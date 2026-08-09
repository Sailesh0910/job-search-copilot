"""
MCP tool definitions for the job hunting agent.

Thin wrappers over job_broker.py only — no SQL, no HTTP calls, no business
logic here, same broker-pattern discipline as the Day 3 weather MCP server.

This module defines `mcp` but does not call mcp.run(). It gets mounted into
the FastAPI app in main.py instead, since Free Edition allows only one
Databricks App and this one also serves the web frontend.
"""

import hashlib
import hmac
import secrets

from fastmcp import FastMCP

import job_broker

mcp = FastMCP("Job-Hunting-Copilot")

# Per-process secret backing remove_saved_job's confirmation tokens (see
# below). Regenerated on every restart, which just means a token issued
# before a redeploy stops working, not a real problem for a single-user app.
_REMOVAL_TOKEN_SECRET = secrets.token_bytes(32)


def _removal_token(job_posting_id: str) -> str:
    return hmac.new(_REMOVAL_TOKEN_SECRET, job_posting_id.encode(), hashlib.sha256).hexdigest()[:12]


@mcp.tool()
def search_jobs(query: str, top_k: int = 10) -> dict:
    """
    Semantically searches already-ingested job postings.
    Use this when the user describes what they want in their own words,
    e.g. "remote backend roles that don't need 5+ years of Kubernetes".

    Args:
        query: Natural-language description of the role wanted.
        top_k: Max results to return (1-50).

    Returns:
        Dict with 'results': postings with similarity scores, title, company,
        location, salary, sponsorship_signal, work_mode_signal, and url.
    """
    return {"results": job_broker.search_jobs(query, top_k=top_k)}


@mcp.tool()
def get_recommended_jobs(top_k: int = 10) -> dict:
    """
    Returns postings ranked against the user's saved profile and resume —
    no search query needed. Use this when the user asks for their best
    matches or "what should I apply to" without specifying keywords.

    Args:
        top_k: Max results to return (1-50).

    Returns:
        Dict with 'results', or an empty list plus 'message' if no profile
        has been saved yet.
    """
    results = job_broker.get_recommended_jobs(top_k=top_k)
    if not results:
        return {"results": [], "message": "No profile saved yet, or it has no resume text to match against."}
    return {"results": results}


@mcp.tool()
def find_new_postings(what: str, where: str = None, max_results: int = 20) -> dict:
    """
    Fetches fresh postings from Adzuna right now and stores them. Use this
    when search_jobs or get_recommended_jobs don't have good matches for a
    role or location the user asked about — this goes and gets new data
    rather than searching what's already stored.

    Args:
        what: Role or keywords to search for, e.g. "data engineer".
        where: Optional location filter, e.g. "Austin".
        max_results: How many postings to fetch (1-50).

    Returns:
        Dict with 'fetched' and 'written' counts, or 'error' if the fetch failed.
    """
    try:
        return job_broker.fetch_new_postings(what, where=where, max_results=max_results)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def save_job(job_posting_id: str, status: str = "saved") -> dict:
    """
    Adds a posting to the pipeline, or moves it to a new stage if it's
    already there — the same tool handles both. Use this when the user
    wants to save, track, or update the status of a specific posting.

    Args:
        job_posting_id: The posting's id, from a search result.
        status: One of 'saved', 'applied', 'interviewing', 'rejected', 'offer'.

    Returns:
        Dict with the application's id and current status, or 'error' if the
        posting id doesn't exist.
    """
    try:
        return job_broker.save_to_pipeline(job_posting_id, status=status)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def view_pipeline(status: str = None) -> dict:
    """
    Returns tracked applications, optionally filtered to one stage. Use this
    when the user asks what they've applied to or wants to see their pipeline.

    Args:
        status: Optional filter, one of 'saved', 'applied', 'interviewing',
                'rejected', 'offer'. Omit for everything.

    Returns:
        Dict with 'applications': tracked postings with their status. Each
        entry includes posting_possibly_stale — a heuristic (the posting is
        old) rather than a live recheck of whether the listing is still up;
        present it as a hint to verify, not a fact.
    """
    return {"applications": job_broker.get_pipeline(status=status)}


@mcp.tool()
def remove_saved_job(job_posting_id: str, confirmation_token: str = None) -> dict:
    """
    Removes a posting from the pipeline entirely — not a status change, the
    tracked application and any interview notes on it are deleted. Use this
    only when the user explicitly asks to remove, delete, or un-save a
    posting, never as a side effect of another request.

    This is two calls by design, not a formality: the first call (with no
    confirmation_token) never deletes anything. It returns the posting's
    details plus a confirmation_token. Only call this a second time, passing
    that exact token back, after the user has actually confirmed they want
    it gone — that second call is what performs the deletion. This is
    enforced here, not just requested by instruction, so a single tool call
    can never delete data.

    Args:
        job_posting_id: The posting's id, from view_pipeline.
        confirmation_token: Omit on the first call. Pass back the token from
            that first call's response to actually perform the removal.

    Returns:
        First call: {"confirmation_required": True, "confirmation_token":
        ..., "posting": {...}}. Second call (with the right token):
        {"removed": True/False}. {"error": ...} on failure.
    """
    try:
        expected_token = _removal_token(job_posting_id)
        if confirmation_token != expected_token:
            posting = job_broker.get_job_posting(job_posting_id)
            if not posting:
                return {"error": f"No posting found with id {job_posting_id}"}
            return {
                "confirmation_required": True,
                "confirmation_token": expected_token,
                "posting": {"title": posting.get("title"), "company": posting.get("company")},
                "message": (
                    "Ask the user to confirm before calling remove_saved_job "
                    "again with this exact confirmation_token."
                ),
            }
        result = job_broker.remove_from_pipeline(job_posting_id)
        return {"removed": result is not None}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def check_stale_applications(days: int = 14) -> dict:
    """
    Finds applications with no status change in a while that may need
    follow-up. Use this when the user asks what needs attention.

    Args:
        days: Days of inactivity that counts as stale. Defaults to 14.

    Returns:
        Dict with 'stale': applications and their last-updated date.
    """
    return {"stale": job_broker.get_stale_applications(days=days)}


@mcp.tool()
def draft_cover_letter(job_posting_id: str) -> dict:
    """
    Drafts a short, specific cover letter paragraph for one posting, grounded
    in the saved profile and the posting's actual text. Use this when the
    user asks for help applying or wants application material for a
    specific posting.

    Args:
        job_posting_id: The posting's id.

    Returns:
        Dict with 'draft', or 'error' if there's no profile saved or the
        posting doesn't exist.
    """
    try:
        return {"draft": job_broker.draft_cover_letter(job_posting_id)}
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)}


@mcp.tool()
def log_interview_note(application_id: int, note_text: str, interview_date: str = None) -> dict:
    """
    Records a note against a tracked application. Use this when the user
    wants to log something about an interview or a conversation with an
    employer.

    Args:
        application_id: The application's id, from view_pipeline.
        note_text: The note content.
        interview_date: Optional date string (YYYY-MM-DD).

    Returns:
        Dict with the new note's id, or 'error' on failure.
    """
    try:
        return job_broker.add_interview_note(application_id, note_text, interview_date)
    except Exception as e:
        return {"error": str(e)}