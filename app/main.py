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

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import embeddings
import job_broker
from mcp_tools import mcp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mcp_app = mcp.http_app(path="/mcp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup: creates the schema if needed and warms the
    embedding model so the first real request isn't the one paying the
    multi-second model-load cost.

    Also enters the MCP server's own lifespan — FastMCP's session manager
    needs this wired into the parent app's lifespan, not just mounted as
    routes, or tool calls fail silently.
    """
    job_broker.ensure_schema()
    embeddings.get_model()
    async with mcp_app.lifespan(app):
        yield


app = FastAPI(title="Job Hunting Copilot", lifespan=lifespan)
app.mount("/mcp", mcp_app)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

STATUSES = ["saved", "applied", "interviewing", "rejected", "offer"]


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------

@app.get("/")
def dashboard(request: Request):
    stats = job_broker.get_pipeline_stats()
    recent_runs = job_broker.get_recent_runs(limit=5)
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "stats": stats, "recent_runs": recent_runs,
    })


# ----------------------------------------------------------------------------
# Profile
# ----------------------------------------------------------------------------

@app.get("/profile")
def profile_form(request: Request):
    profile = job_broker.get_profile() or {}
    return templates.TemplateResponse("profile.html", {"request": request, "profile": profile})


@app.post("/profile")
def profile_save(
    full_name: str = Form(""),
    target_roles: str = Form(""),
    years_experience: str = Form(""),
    location_preference: str = Form("any"),
    remote_preference: str = Form("any"),
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
        "years_experience": int(years_experience) if years_experience.strip() else None,
        "location_preference": location_preference or "any",
        "remote_preference": remote_preference or "any",
        "min_salary": int(min_salary) if min_salary.strip() else None,
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
    work_mode: str = "any",
    sponsorship_only: bool = False,
    sort: str = "newest",
):
    """
    Two modes on one page: a typed query runs semantic search; no query
    falls back to plain structured browsing with the filter controls.
    """
    error = None
    try:
        if q.strip():
            jobs = job_broker.search_jobs(q.strip(), top_k=30)
        else:
            jobs = job_broker.browse_jobs(
                min_salary=int(min_salary) if min_salary.strip() else None,
                work_mode=work_mode, sponsorship_only=sponsorship_only, sort_by=sort,
            )
    except job_broker.LakebaseError as e:
        jobs, error = [], str(e)

    return templates.TemplateResponse("jobs.html", {
        "request": request, "jobs": jobs, "q": q, "min_salary": min_salary,
        "work_mode": work_mode, "sponsorship_only": sponsorship_only,
        "sort": sort, "error": error, "mode": "browse",
    })


@app.get("/recommended")
def recommended(request: Request):
    jobs = job_broker.get_recommended_jobs(top_k=30)
    return templates.TemplateResponse("jobs.html", {
        "request": request, "jobs": jobs, "q": "", "min_salary": "",
        "work_mode": "any", "sponsorship_only": False, "sort": "newest",
        "error": None, "mode": "recommended",
    })


@app.post("/jobs/fetch")
def jobs_fetch(what: str = Form(...), where: str = Form("")):
    """Live on-demand fetch from Adzuna — the ad-hoc counterpart to the Spark batch job."""
    try:
        job_broker.fetch_new_postings(what.strip(), where=where.strip() or None)
    except Exception:
        pass  # keep the UI moving; the redirect target just shows whatever was found
    return RedirectResponse(f"/jobs?q={what}", status_code=303)


@app.post("/jobs/{job_posting_id}/save")
def job_save(job_posting_id: str, status: str = Form("saved")):
    job_broker.save_to_pipeline(job_posting_id, status=status)
    return RedirectResponse("/pipeline", status_code=303)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

@app.get("/pipeline")
def pipeline_board(request: Request, status: str = ""):
    applications = job_broker.get_pipeline(status=status or None)
    stale = job_broker.get_stale_applications()
    return templates.TemplateResponse("pipeline.html", {
        "request": request, "applications": applications, "stale": stale,
        "statuses": STATUSES, "current_status": status,
    })


@app.post("/pipeline/{job_posting_id}/status")
def pipeline_update_status(job_posting_id: str, new_status: str = Form(...)):
    job_broker.update_status(job_posting_id, new_status)
    return RedirectResponse("/pipeline", status_code=303)


@app.post("/pipeline/{application_id}/notes")
def pipeline_add_note(application_id: int, note_text: str = Form(...), interview_date: str = Form("")):
    job_broker.add_interview_note(application_id, note_text, interview_date or None)
    return RedirectResponse("/pipeline", status_code=303)


if __name__ == "__main__":
    import uvicorn
    # Databricks Apps injects the real port via DATABRICKS_APP_PORT — hardcoding
    # one caused the Day 1 Bad Gateway, so read it explicitly rather than assuming.
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)