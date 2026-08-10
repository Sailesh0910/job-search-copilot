"""
Job Hunting Copilot — FastAPI app.

Serves the web frontend (profile, job browsing, pipeline board) and mounts
the MCP server at /mcp for the Agent Bricks agent. One process, one
deployment — Free Edition allows only one Databricks App per account, so the
frontend and the agent's tool server have to live together.

Every route below calls into job_broker.py for actual logic. No SQL, no
embedding calls, no LLM calls happen in this file — that discipline is what
keeps the same business logic usable from both a human clicking buttons and
an agent calling tools.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.routing import Route

import embeddings
import job_broker
from mcp_tools import mcp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# path="/" so the MCP endpoint lives at its own root within the sub-app;
# app.mount("/mcp", ...) below then supplies the "/mcp" prefix. Passing
# path="/mcp" here as well (the previous config) made the effective URL
# "/mcp/mcp" — the mount prefix and the sub-app's internal path stacked.
#
# stateless_http=True: FastMCP's default session tracking is in-memory
# per-process. A Databricks App may run multiple replicas or restart the
# process between requests, either of which drops a session created on one
# request before the next one arrives ("Session terminated"). Stateless
# mode makes every request self-contained, no session continuity required.
mcp_app = mcp.http_app(path="/", stateless_http=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup: creates the schema if needed and warms the
    embedding model so the first real request isn't the one paying the
    multi-second model-load cost. Releases pooled database connections on
    shutdown.

    Also enters the MCP server's own lifespan — FastMCP's session manager
    needs this wired into the parent app's lifespan, not just mounted as
    routes, or tool calls fail silently.
    """
    job_broker.ensure_schema()
    embeddings.get_model()
    async with mcp_app.lifespan(app):
        yield
    job_broker.close_db_pool()


app = FastAPI(title="Job Hunting Copilot", lifespan=lifespan)


class _McpBarePassthrough:
    """
    Forwards requests at exactly /mcp (no trailing slash) straight into the
    mounted MCP app, bypassing Starlette's Mount, which otherwise 307s a
    bare-prefix request to /mcp/ before it ever reaches the handler.
    Confirmed necessary, not precautionary: Agent Bricks' MCP client (used
    to register this app's tools) doesn't follow that redirect on POST, so
    tool registration failed outright until this was added.

    A class instance rather than a plain function, since Starlette's Route
    wraps plain functions as request/response handlers; a callable object
    is treated as an ASGI app already, which is what's needed here to
    forward the raw (scope, receive, send) straight through.
    """
    async def __call__(self, scope, receive, send):
        scope = dict(scope)
        scope["path"] = "/"
        scope["root_path"] = scope.get("root_path", "") + "/mcp"
        await mcp_app(scope, receive, send)


# Must be registered before the mount below — Starlette matches routes in
# registration order, and this exact-path route needs to win over Mount's
# own (redirecting) handling of the same bare path.
app.router.routes.insert(0, Route("/mcp", _McpBarePassthrough(), methods=["GET", "POST", "DELETE"]))
app.mount("/mcp", mcp_app)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

STATUS = Literal["saved", "applied", "interviewing", "rejected", "offer"]
STATUSES = list(STATUS.__args__)

JOBS_PAGE_SIZE = 25

# In-memory chat history. Fine for a single-user app with no auth (same
# scope as the rest of the app, see ARCHITECTURE.md); resets on redeploy or
# if the app sleeps, which is an acceptable tradeoff for a demo chat surface.
_chat_history: list = []


def _parse_optional_int(raw: Optional[str]) -> Optional[int]:
    """
    Parses an optional numeric form/query field, treating a blank or
    non-numeric value as "not provided" rather than raising — the previous
    plain int(raw) crashed the whole request (500) on the ordinary case of a
    user leaving an optional number field empty or typing something
    non-numeric into it.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@app.exception_handler(job_broker.LakebaseError)
async def lakebase_error_handler(request: Request, exc: job_broker.LakebaseError):
    """
    Safety net for routes that don't already handle LakebaseError themselves
    (currently just /jobs, which shows the error inline alongside the search
    filters). The raw exception text isn't shown to the browser — it can
    carry connection details from the underlying psycopg2 error — it's
    logged server-side instead.
    """
    logger.error("Unhandled LakebaseError on %s %s", request.method, request.url.path, exc_info=exc)
    return templates.TemplateResponse(
        request, "error.html",
        {"message": "Something went wrong talking to the database. Please try again in a moment."},
        status_code=503,
    )


# ----------------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------

@app.get("/")
def dashboard(request: Request):
    stats = job_broker.get_pipeline_stats()
    recent_runs = job_broker.get_recent_runs(limit=5)
    return templates.TemplateResponse(request, "dashboard.html", {
        "stats": stats, "recent_runs": recent_runs,
    })


# ----------------------------------------------------------------------------
# Profile
# ----------------------------------------------------------------------------

@app.get("/profile")
def profile_form(request: Request):
    profile = job_broker.get_profile() or {}
    return templates.TemplateResponse(request, "profile.html", {"profile": profile})


@app.post("/profile")
def profile_save(
    full_name: str = Form(""),
    target_roles: str = Form(""),
    years_experience: str = Form(""),
    location_preference: str = Form("any"),
    remote_preference: Literal["any", "remote", "hybrid", "onsite"] = Form("any"),
    min_salary: str = Form(""),
    sponsorship_required: bool = Form(False),
    work_authorization: str = Form(""),
    tech_stack_musthaves: str = Form(""),
    company_size_pref: str = Form(""),
    other_notes: str = Form(""),
    resume_text: str = Form(""),
):
    job_broker.save_profile({
        "full_name": full_name or None,
        "target_roles": target_roles or None,
        "years_experience": _parse_optional_int(years_experience),
        "location_preference": location_preference or "any",
        "remote_preference": remote_preference or "any",
        "min_salary": _parse_optional_int(min_salary),
        "sponsorship_required": sponsorship_required,
        "work_authorization": work_authorization or None,
        "tech_stack_musthaves": tech_stack_musthaves or None,
        "company_size_pref": company_size_pref or None,
        "other_notes": other_notes or None,
        "resume_text": resume_text or None,
    })
    return RedirectResponse("/profile?saved=1", status_code=303)


# ----------------------------------------------------------------------------
# Jobs — browse, search, live fetch
# ----------------------------------------------------------------------------

@app.get("/jobs")
def jobs_list(
    request: Request,
    q: str = "",
    min_salary: str = "",
    work_mode: Literal["any", "remote", "hybrid", "onsite"] = "any",
    sponsorship_only: bool = False,
    sort: Literal["newest", "oldest"] = "newest",
    page: int = 1,
    fetch_error: bool = False,
):
    """
    Two modes on one page: a typed query runs semantic search; no query
    falls back to plain structured browsing with the filter controls.

    Pagination only applies to browse mode — semantic search already returns
    a fixed, relevance-ranked top_k list, so "page 2 of a search" isn't a
    meaningful concept here the way "page 2 of newest postings" is.
    """
    error = None
    if fetch_error:
        error = ("Couldn't fetch new postings from Adzuna. Check that ADZUNA_APP_ID "
                  "and ADZUNA_APP_KEY are configured correctly, then try again.")

    page = max(1, page)
    has_more = False
    try:
        if q.strip():
            jobs = job_broker.search_jobs(q.strip(), top_k=30)
        else:
            jobs, has_more = job_broker.browse_jobs(
                min_salary=_parse_optional_int(min_salary),
                work_mode=work_mode, sponsorship_only=sponsorship_only, sort_by=sort,
                limit=JOBS_PAGE_SIZE, offset=(page - 1) * JOBS_PAGE_SIZE,
            )
    except job_broker.LakebaseError as e:
        logger.error("Lakebase error on /jobs: %s", e)
        jobs = []
        error = error or "Couldn't load jobs right now — the database isn't reachable. Try again shortly."

    return templates.TemplateResponse(request, "jobs.html", {
        "jobs": jobs, "q": q, "min_salary": min_salary,
        "work_mode": work_mode, "sponsorship_only": sponsorship_only,
        "sort": sort, "error": error, "mode": "browse",
        "page": page, "has_more": has_more,
    })


@app.get("/recommended")
def recommended(request: Request):
    jobs = job_broker.get_recommended_jobs(top_k=30)
    return templates.TemplateResponse(request, "jobs.html", {
        "jobs": jobs, "q": "", "min_salary": "",
        "work_mode": "any", "sponsorship_only": False, "sort": "newest",
        "error": None, "mode": "recommended", "page": 1, "has_more": False,
    })


@app.post("/jobs/fetch")
def jobs_fetch(what: str = Form(...), where: str = Form("")):
    """Live on-demand fetch from Adzuna — the ad-hoc counterpart to the Spark batch job."""
    what = what.strip()
    try:
        job_broker.fetch_new_postings(what, where=where.strip() or None)
    except Exception as e:
        logger.error("Adzuna fetch failed for %r: %s", what, e)
        return RedirectResponse(f"/jobs?q={quote(what)}&fetch_error=1", status_code=303)
    return RedirectResponse(f"/jobs?q={quote(what)}", status_code=303)


@app.post("/jobs/{job_posting_id}/save")
def job_save(job_posting_id: str, status: STATUS = Form("saved")):
    job_broker.save_to_pipeline(job_posting_id, status=status)
    return RedirectResponse("/pipeline", status_code=303)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

@app.get("/pipeline")
def pipeline_board(request: Request, status: str = "", cover_letter_error: bool = False):
    applications = job_broker.get_pipeline(status=status or None)
    stale = job_broker.get_stale_applications()
    error = (
        "Couldn't draft a cover letter right now. Check that your profile is "
        "saved and the serving endpoint is available, then try again."
        if cover_letter_error else None
    )
    return templates.TemplateResponse(request, "pipeline.html", {
        "applications": applications, "stale": stale,
        "statuses": STATUSES, "current_status": status, "error": error,
    })


@app.post("/pipeline/{job_posting_id}/status")
def pipeline_update_status(job_posting_id: str, new_status: STATUS = Form(...)):
    job_broker.update_status(job_posting_id, new_status)
    return RedirectResponse("/pipeline", status_code=303)


@app.post("/pipeline/{job_posting_id}/cover-letter")
def pipeline_draft_cover_letter(job_posting_id: str):
    """Drafts (or redrafts) a cover letter for a tracked posting from the
    web UI — the same job_broker.draft_cover_letter the MCP tool uses, so
    the agent and a human clicking buttons get identical behavior."""
    try:
        job_broker.draft_cover_letter(job_posting_id)
    except (ValueError, RuntimeError) as e:
        logger.error("Cover letter drafting failed for %s: %s", job_posting_id, e)
        return RedirectResponse("/pipeline?cover_letter_error=1", status_code=303)
    return RedirectResponse("/pipeline", status_code=303)


@app.post("/pipeline/{job_posting_id}/remove")
def pipeline_remove(job_posting_id: str):
    """Removes a posting from the pipeline entirely — not a status change."""
    job_broker.remove_from_pipeline(job_posting_id)
    return RedirectResponse("/pipeline", status_code=303)


@app.post("/pipeline/{application_id}/notes")
def pipeline_add_note(application_id: int, note_text: str = Form(...), interview_date: str = Form("")):
    job_broker.add_interview_note(application_id, note_text, interview_date or None)
    return RedirectResponse("/pipeline", status_code=303)


# ----------------------------------------------------------------------------
# Chat — talk to the Supervisor Agent from inside this app, instead of only
# through the Agent Bricks Playground. The agent still does its own tool
# calls back into this app's /mcp; this is purely a client for its endpoint.
# ----------------------------------------------------------------------------

def _chat_display_messages(history: list) -> list:
    """
    _chat_history stores the raw conversation items chat_with_agent needs to
    resend every turn (user/assistant messages, plus mcp_approval_request/
    response and function_call_output items once a tool gets called) — not
    something to render directly. This picks out just the user and
    assistant text turns, in order, for the chat bubble UI.
    """
    display = []
    for item in history:
        if item.get("role") == "user" and "content" in item and isinstance(item["content"], str):
            display.append({"role": "user", "content": item["content"]})
        elif item.get("type") == "message" and item.get("role") == "assistant":
            text = " ".join(c.get("text", "") for c in item.get("content", [])).strip()
            if text:
                display.append({"role": "assistant", "content": text})
    return display


@app.get("/chat")
def chat_page(request: Request, error: bool = False):
    message = (
        "Couldn't reach the agent. Check that its endpoint is deployed and "
        "reachable, then try again."
        if error else None
    )
    return templates.TemplateResponse(request, "chat.html", {
        "history": _chat_display_messages(_chat_history), "error": message,
    })


@app.post("/chat")
def chat_send(message: str = Form(...)):
    message = message.strip()
    if not message:
        return RedirectResponse("/chat", status_code=303)

    _chat_history.append({"role": "user", "content": message})
    try:
        job_broker.chat_with_agent(_chat_history)
    except RuntimeError as e:
        logger.error("Chat with agent failed: %s", e)
        return RedirectResponse("/chat?error=1", status_code=303)

    return RedirectResponse("/chat", status_code=303)


@app.post("/chat/clear")
def chat_clear():
    _chat_history.clear()
    return RedirectResponse("/chat", status_code=303)


if __name__ == "__main__":
    import uvicorn
    # Databricks Apps injects the real port via DATABRICKS_APP_PORT — hardcoding
    # one caused the Day 1 Bad Gateway, so read it explicitly rather than assuming.
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
