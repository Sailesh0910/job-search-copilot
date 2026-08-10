"""
Business logic shared by the web routes (main.py) and the agent's MCP tools
(mcp_tools.py). Neither of those files should contain SQL or HTTP calls
directly — they call into here, and this is the one place that logic lives.
"""

import json
import logging
import os

import requests

import classify
import embeddings
import lakebase
from adzuna_client import AdzunaClient
from config import COVER_LETTER_MODEL, SUPERVISOR_AGENT_ENDPOINT
from lakebase import LakebaseError  # re-exported so callers never import lakebase directly

logger = logging.getLogger(__name__)


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
        response = requests.post(
            f"{w.config.host}/ai-gateway/mlflow/v1/chat/completions",
            headers=w.config.authenticate(),
            json={
                "model": COVER_LETTER_MODEL,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        draft = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # The most common failure here is COVER_LETTER_MODEL not being a
        # valid model id in this workspace's AI Gateway — surface that
        # directly rather than a raw traceback the caller (the web route or
        # an MCP tool) has no way to interpret.
        raise RuntimeError(
            f"Cover letter drafting failed calling model "
            f"'{COVER_LETTER_MODEL}': {e}. Check that this model is listed "
            f"under AI Gateway > Models in your workspace."
        ) from e

    lakebase.save_cover_letter(job_posting_id, draft)
    return draft


# ----------------------------------------------------------------------------
# Chat with the agent — a web UI client for the Supervisor Agent's own
# serving endpoint. The agent handles its own tool-calling (including MCP
# calls back into this same app) internally; this is just the client side of
# that conversation. Distinct from draft_cover_letter above, which calls a
# plain chat-completions model directly with no agent/tool loop involved.
# ----------------------------------------------------------------------------

def _extract_sse_error(raw_text: str):
    """Pulls a human-readable message out of a server-sent-events error
    frame, e.g. 'event: error\\ndata: {"message": "..."}\\n\\ndata: [DONE]\\n\\n'.
    Returns None if no such frame is present."""
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            parsed = json.loads(payload)
        except ValueError:
            continue
        if isinstance(parsed, dict) and "message" in parsed:
            return parsed["message"]
    return None


def _post_invocations(host: str, headers: dict, conversation: list) -> dict:
    """POSTs the full conversation to the agent's serving endpoint and
    returns the parsed JSON body, or raises a RuntimeError with the real
    cause.

    Uses {host}/serving-endpoints/{endpoint}/invocations, not the OpenAI
    Responses API's generic /serving-endpoints/responses route. Confirmed by
    inspecting the Agent Bricks Playground's own network traffic: this
    endpoint has no previous_response_id continuation, every call must
    resend the complete conversation from scratch."""
    try:
        response = requests.post(
            f"{host}/serving-endpoints/{SUPERVISOR_AGENT_ENDPOINT}/invocations",
            headers=headers,
            json={"input": conversation, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Chat request to the agent failed: {e}. Check that the agent "
            f"endpoint '{SUPERVISOR_AGENT_ENDPOINT}' is deployed and "
            f"reachable."
        ) from e

    try:
        return response.json()
    except ValueError as e:
        # response.json() only raises this on a body that isn't valid JSON.
        # Despite stream: False, the endpoint may still reply as
        # server-sent events (e.g. its error frames do) — pull a real
        # message out of that shape if present, since a bare "Expecting
        # value" error can't tell us what actually went wrong.
        sse_error = _extract_sse_error(response.text)
        if sse_error:
            raise RuntimeError(
                f"Agent endpoint '{SUPERVISOR_AGENT_ENDPOINT}' returned an "
                f"error: {sse_error}"
            ) from e
        raise RuntimeError(
            f"Agent endpoint '{SUPERVISOR_AGENT_ENDPOINT}' returned a "
            f"non-JSON response (status {response.status_code}): "
            f"{response.text[:500]!r}"
        ) from e


def chat_with_agent(conversation: list) -> str:
    """
    Sends the full conversation so far to the Supervisor Agent and returns
    its reply text. `conversation` is a list of raw item dicts (user/
    assistant messages, and — once a tool has been called — the
    mcp_approval_request/response and function_call_output items below);
    this function mutates it in place, appending every new item so the
    caller's stored history is exactly what needs to be resent next turn.

    This endpoint (unlike the generic OpenAI Responses API) is stateless
    with no previous_response_id continuation, and does not execute MCP
    tools itself after an approval — the caller has to actually run the
    tool and report the result back as a function_call_output. Both facts
    were confirmed by inspecting Agent Bricks Playground's own network
    traffic (it doesn't work any other way, multiple other request shapes
    were tried and rejected with an "Invalid message sequence" error), not
    assumed from the OpenAI spec.
    """
    from mcp_tools import TOOL_DISPATCH

    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        host = w.config.host
        headers = w.config.authenticate()
    except Exception as e:
        raise RuntimeError(
            f"Chat request to the agent failed: {e}. Check that the agent "
            f"endpoint '{SUPERVISOR_AGENT_ENDPOINT}' is deployed and "
            f"reachable."
        ) from e

    for _ in range(10):
        data = _post_invocations(host, headers, conversation)
        output_items = data.get("output", [])
        conversation.extend(output_items)

        approval_requests = [item for item in output_items if item.get("type") == "mcp_approval_request"]
        if not approval_requests:
            break

        for req in approval_requests:
            conversation.append({
                "type": "mcp_approval_response",
                "approval_request_id": req["id"],
                "approve": True,
            })
            tool_name = req.get("name")
            tool_fn = TOOL_DISPATCH.get(tool_name)
            try:
                arguments = json.loads(req.get("arguments") or "{}")
                result = tool_fn(**arguments) if tool_fn else {"error": f"Unknown tool '{tool_name}'"}
            except Exception as e:
                result = {"error": str(e)}
            logger.info("chat_with_agent ran tool %s -> %s", tool_name, list(result.keys()) if isinstance(result, dict) else type(result))
            conversation.append({
                "type": "function_call_output",
                "call_id": req["id"],
                "name": tool_name,
                # default=str: tool results can carry raw datetime values
                # (e.g. view_pipeline's status_updated_at) straight from the
                # database. FastMCP handles that conversion for us when a
                # tool is called over /mcp; calling the function directly
                # here bypasses that, so it's needed explicitly.
                "output": json.dumps(result, default=str),
            })
    else:
        raise RuntimeError("The agent kept requesting tool calls without producing a final answer (capped at 10 rounds).")

    reply = ""
    for item in reversed(conversation):
        if item.get("type") == "message" and item.get("role") == "assistant":
            reply = " ".join(c.get("text", "") for c in item.get("content", [])).strip()
            break

    if not reply:
        raise RuntimeError(
            f"The agent responded with no text output. Raw response: "
            f"{str(data)[:500]!r}"
        )
    return reply