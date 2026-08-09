"""
Business logic shared by the web routes (main.py) and the agent's MCP tools
(mcp_tools.py). Neither of those files should contain SQL or HTTP calls
directly — they call into here, and this is the one place that logic lives.
"""

import os

import classify
import embeddings
import lakebase
from adzuna_client import AdzunaClient
from config import COVER_LETTER_MODEL
from lakebase import LakebaseError  # re-exported so callers never import lakebase directly


def ensure_schema() -> None:
    lakebase.ensure_schema()


def close_db_pool() -> None:
    """Releases pooled Lakebase connections. Call on app shutdown."""
    lakebase.close_pool()


def get_profile() -> dict:
    return lakebase.get_profile()


# ----------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------

def search_jobs(query: str, top_k: int = 10) -> list:
    """Semantic search over already-ingested postings."""
    query_vector = embeddings.embed(query)
    return lakebase.search_jobs_semantic(query_vector, top_k=top_k)


def get_recommended_jobs(top_k: int = 20) -> list:
    """Postings ranked against the saved profile — no query needed."""
    return lakebase.rank_jobs_for_profile(top_k=top_k)


def browse_jobs(min_salary=None, work_mode=None, sponsorship_only=False,
                sort_by="newest", limit=50, offset=0) -> tuple:
    """Plain structured browsing — filters and sorting, no embedding model
    needed. Returns (postings, has_more)."""
    return lakebase.browse_jobs(
        min_salary=min_salary, work_mode=work_mode,
        sponsorship_only=sponsorship_only, sort_by=sort_by,
        limit=limit, offset=offset,
    )


def get_job_posting(job_posting_id: str) -> dict:
    return lakebase.get_job_posting(job_posting_id)


# ----------------------------------------------------------------------------
# Live fetch — the on-demand path, distinct from the Spark batch pipeline.
# ----------------------------------------------------------------------------

def fetch_new_postings(what: str, where: str = None, max_results: int = 20) -> dict:
    """
    Fetches postings from Adzuna right now for a specific query, classifies,
    embeds, and stores them — all synchronously in one request.

    This is the ad-hoc counterpart to notebooks/ingest_jobs_spark.py, which
    handles broad scheduled sweeps. Both write to the same Lakebase tables;
    this one just skips Delta and Spark entirely, appropriate at this size
    (tens of postings, not a bulk load).
    """
    client = AdzunaClient(
        app_id=os.environ["ADZUNA_APP_ID"],
        app_key=os.environ["ADZUNA_APP_KEY"],
    )
    raw = client.search(what=what, where=where, results_per_page=max_results, max_pages=1)

    postings = [classify.classify_posting(p) for p in raw]
    written = lakebase.upsert_job_postings(postings)

    embedding_rows = []
    for posting in postings:
        for chunk_index, chunk, vector in embeddings.embed_posting_chunks(posting):
            embedding_rows.append((posting["id"], chunk_index, chunk, vector))
    lakebase.upsert_job_embeddings(embedding_rows)

    return {"fetched": len(postings), "written": written}


# ----------------------------------------------------------------------------
# Profile
# ----------------------------------------------------------------------------

def save_profile(profile: dict) -> None:
    """Saves the profile and (re)computes its embedding from the new text."""
    vector = embeddings.embed(embeddings.build_profile_text(profile))
    lakebase.save_profile(profile, embedding=vector)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def save_to_pipeline(job_posting_id: str, status: str = "saved") -> dict:
    return lakebase.save_to_pipeline(job_posting_id, status=status)


def update_status(job_posting_id: str, new_status: str) -> dict:
    return lakebase.update_application_status(job_posting_id, new_status)


def get_pipeline(status: str = None) -> list:
    return lakebase.get_pipeline(status=status)


def remove_from_pipeline(job_posting_id: str) -> dict:
    """Removes a posting from the pipeline entirely (not a status change)."""
    return lakebase.delete_application(job_posting_id)


def get_stale_applications(days: int = 14) -> list:
    return lakebase.get_stale_applications(days=days)


def add_interview_note(application_id: int, note_text: str, interview_date: str = None) -> dict:
    return lakebase.add_interview_note(application_id, note_text, interview_date)


def get_interview_notes(application_id: int) -> list:
    return lakebase.get_interview_notes(application_id)


def get_pipeline_stats() -> dict:
    return lakebase.get_pipeline_stats()


def get_recent_runs(limit: int = 20) -> list:
    return lakebase.get_recent_runs(limit=limit)


# ----------------------------------------------------------------------------
# Cover letter drafting — the one feature that needs an LLM rather than
# retrieval. Uses a Databricks foundation model serving endpoint via the
# Workspace SDK, which authenticates using the app's own identity — no
# separate API key or secret to manage.
# ----------------------------------------------------------------------------

def draft_cover_letter(job_posting_id: str) -> str:
    """
    Drafts a short cover letter paragraph tailored to one posting, grounded in
    the saved profile and the posting's actual text — not generic filler.

    Raises ValueError if there's no profile or the posting doesn't exist,
    rather than silently drafting a letter with nothing to draw on.
    """
    posting = lakebase.get_job_posting(job_posting_id)
    if not posting:
        raise ValueError(f"No posting found with id {job_posting_id}")

    profile = lakebase.get_profile()
    if not profile:
        raise ValueError("No profile saved yet — set one up before drafting a cover letter")

    prompt = f"""Write a concise, specific cover letter paragraph (120-180 words) for this candidate applying to this job. Reference at least one concrete detail from the job description and one concrete skill or experience from the candidate. No generic filler like "I am excited to apply". Plain text, no markdown.

CANDIDATE
Name: {profile.get('full_name', 'the candidate')}
Target roles: {profile.get('target_roles', '')}
Experience: {profile.get('years_experience', 'unspecified')} years
Skills: {profile.get('tech_stack_musthaves', '')}
Resume summary: {(profile.get('resume_text') or '')[:1000]}

JOB POSTING
Title: {posting['title']}
Company: {posting.get('company', 'the company')}
Description: {posting['description'][:1500]}
"""

    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        openai_client = w.serving_endpoints.get_open_ai_client()
        response = openai_client.chat.completions.create(
            model=COVER_LETTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        draft = response.choices[0].message.content.strip()
    except Exception as e:
        # The most common failure here is the serving endpoint named by
        # COVER_LETTER_MODEL not existing/being enabled in this workspace —
        # surface that directly rather than a raw SDK traceback the caller
        # (the web route or an MCP tool) has no way to interpret.
        raise RuntimeError(
            f"Cover letter drafting failed calling serving endpoint "
            f"'{COVER_LETTER_MODEL}': {e}. Check that this endpoint exists "
            f"and is enabled in your workspace."
        ) from e

    lakebase.save_cover_letter(job_posting_id, draft)
    return draft