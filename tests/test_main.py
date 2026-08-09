"""
Exercises the actual FastAPI app (routes, validation, the lifespan, the
centralized error handler) via TestClient. job_broker's functions are
monkeypatched at the seam main.py calls through — no database, no Adzuna
call, and no real embedding model load happen here. The lifespan itself
still runs for real (schema creation and model warmup are stubbed to
no-ops), which is what exercises the MCP mount wiring and shutdown pool
cleanup along with everything else.
"""

import pytest
from fastapi.testclient import TestClient

import embeddings
import job_broker
import main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(job_broker, "ensure_schema", lambda: None)
    monkeypatch.setattr(embeddings, "get_model", lambda: None)
    monkeypatch.setattr(job_broker, "close_db_pool", lambda: None)
    with TestClient(main.app) as c:
        yield c


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_dashboard_loads(client, monkeypatch):
    monkeypatch.setattr(job_broker, "get_pipeline_stats", lambda: {"saved": 2, "applied": 1})
    monkeypatch.setattr(job_broker, "get_recent_runs", lambda limit: [])
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


# ----------------------------------------------------------------------------
# Profile — regression tests for the crash-on-blank-numeric-field bug
# ----------------------------------------------------------------------------

def test_profile_get_renders_empty_form_when_no_profile_saved(client, monkeypatch):
    monkeypatch.setattr(job_broker, "get_profile", lambda: None)
    resp = client.get("/profile")
    assert resp.status_code == 200
    assert "None" not in resp.text  # years_experience/min_salary shouldn't leak literal "None"


def test_profile_post_with_blank_optional_numbers_does_not_crash(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(job_broker, "save_profile", lambda profile: captured.update(profile))

    resp = client.post("/profile", data={
        "full_name": "Alex", "years_experience": "", "min_salary": "",
    }, follow_redirects=False)

    assert resp.status_code == 303
    assert captured["years_experience"] is None
    assert captured["min_salary"] is None


def test_profile_post_with_non_numeric_input_does_not_crash(client, monkeypatch):
    """The original bug: int(years_experience) on non-numeric input raised
    an unhandled ValueError -> 500. Non-numeric input should be treated as
    'not provided', same as blank."""
    captured = {}
    monkeypatch.setattr(job_broker, "save_profile", lambda profile: captured.update(profile))

    resp = client.post("/profile", data={"years_experience": "not-a-number"}, follow_redirects=False)

    assert resp.status_code == 303
    assert captured["years_experience"] is None


def test_profile_post_with_valid_numbers_parses_correctly(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(job_broker, "save_profile", lambda profile: captured.update(profile))

    resp = client.post("/profile", data={"years_experience": "5", "min_salary": "90000"},
                        follow_redirects=False)

    assert resp.status_code == 303
    assert captured["years_experience"] == 5
    assert captured["min_salary"] == 90000


# ----------------------------------------------------------------------------
# Jobs — Literal-typed query param validation
# ----------------------------------------------------------------------------

def test_jobs_list_invalid_work_mode_returns_422(client):
    resp = client.get("/jobs?work_mode=bogus")
    assert resp.status_code == 422


def test_jobs_list_browse_mode(client, monkeypatch):
    monkeypatch.setattr(job_broker, "browse_jobs", lambda **kwargs: ([], False))
    resp = client.get("/jobs")
    assert resp.status_code == 200


def test_jobs_fetch_failure_redirects_with_error_flag_and_shows_message(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("ADZUNA_APP_ID not set")

    monkeypatch.setattr(job_broker, "fetch_new_postings", boom)
    monkeypatch.setattr(job_broker, "browse_jobs", lambda **kwargs: ([], False))
    monkeypatch.setattr(job_broker, "search_jobs", lambda *a, **kw: [])

    resp = client.post("/jobs/fetch", data={"what": "data engineer"}, follow_redirects=False)
    assert resp.status_code == 303
    assert "fetch_error=1" in resp.headers["location"]

    follow = client.get(resp.headers["location"])
    assert follow.status_code == 200
    assert "Adzuna" in follow.text


# ----------------------------------------------------------------------------
# Jobs — pagination
# ----------------------------------------------------------------------------

def test_jobs_list_shows_next_link_when_has_more(client, monkeypatch):
    monkeypatch.setattr(job_broker, "browse_jobs", lambda **kwargs: ([{
        "id": "1", "title": "DE", "company": "Acme", "location": "Austin",
        "sponsorship_signal": "not_mentioned", "work_mode_signal": "not_mentioned",
        "description": "d", "salary_min": None, "salary_max": None, "url": None,
    }], True))
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert "page=2" in resp.text
    assert "Next" in resp.text


def test_jobs_list_page_param_computes_offset(client, monkeypatch):
    captured = {}

    def fake_browse(**kwargs):
        captured.update(kwargs)
        return ([], False)

    monkeypatch.setattr(job_broker, "browse_jobs", fake_browse)
    resp = client.get("/jobs?page=3")
    assert resp.status_code == 200
    assert captured["offset"] == 2 * main.JOBS_PAGE_SIZE
    assert captured["limit"] == main.JOBS_PAGE_SIZE


def test_jobs_list_invalid_page_defaults_to_first_page(client, monkeypatch):
    captured = {}

    def fake_browse(**kwargs):
        captured.update(kwargs)
        return ([], False)

    monkeypatch.setattr(job_broker, "browse_jobs", fake_browse)
    resp = client.get("/jobs?page=0")
    assert resp.status_code == 200
    assert captured["offset"] == 0


# ----------------------------------------------------------------------------
# Pipeline — Literal-typed status validation
# ----------------------------------------------------------------------------

def test_pipeline_status_update_rejects_invalid_status(client):
    resp = client.post("/pipeline/job-1/status", data={"new_status": "bogus"})
    assert resp.status_code == 422


def test_pipeline_status_update_accepts_valid_status(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(job_broker, "update_status",
                         lambda job_posting_id, new_status: captured.update(id=job_posting_id, status=new_status))
    resp = client.post("/pipeline/job-1/status", data={"new_status": "applied"}, follow_redirects=False)
    assert resp.status_code == 303
    assert captured == {"id": "job-1", "status": "applied"}


def test_pipeline_board_loads(client, monkeypatch):
    monkeypatch.setattr(job_broker, "get_pipeline", lambda status=None: [])
    monkeypatch.setattr(job_broker, "get_stale_applications", lambda: [])
    resp = client.get("/pipeline")
    assert resp.status_code == 200


def test_pipeline_board_shows_possibly_stale_badge(client, monkeypatch):
    monkeypatch.setattr(job_broker, "get_pipeline", lambda status=None: [{
        "application_id": 1, "job_posting_id": "job-1", "status": "saved",
        "status_updated_at": "2024-01-01", "applied_at": None, "cover_letter_draft": None,
        "title": "DE", "company": "Acme", "location": "Austin", "url": None,
        "salary_min": None, "salary_max": None, "sponsorship_signal": "not_mentioned",
        "work_mode_signal": "not_mentioned", "posting_possibly_stale": True,
    }])
    monkeypatch.setattr(job_broker, "get_stale_applications", lambda: [])
    resp = client.get("/pipeline")
    assert resp.status_code == 200
    assert "Possibly stale" in resp.text


# ----------------------------------------------------------------------------
# Pipeline — cover letter drafting
# ----------------------------------------------------------------------------

def test_pipeline_draft_cover_letter_success_redirects_without_error(client, monkeypatch):
    monkeypatch.setattr(job_broker, "draft_cover_letter", lambda job_posting_id: "a draft")
    resp = client.post("/pipeline/job-1/cover-letter", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/pipeline"


def test_pipeline_draft_cover_letter_failure_redirects_with_error_and_shows_message(client, monkeypatch):
    def boom(job_posting_id):
        raise ValueError("No profile saved yet")

    monkeypatch.setattr(job_broker, "draft_cover_letter", boom)
    monkeypatch.setattr(job_broker, "get_pipeline", lambda status=None: [])
    monkeypatch.setattr(job_broker, "get_stale_applications", lambda: [])

    resp = client.post("/pipeline/job-1/cover-letter", follow_redirects=False)
    assert resp.status_code == 303
    assert "cover_letter_error=1" in resp.headers["location"]

    follow = client.get(resp.headers["location"])
    assert follow.status_code == 200
    assert "cover letter" in follow.text.lower()


# ----------------------------------------------------------------------------
# Pipeline — remove
# ----------------------------------------------------------------------------

def test_pipeline_remove_calls_job_broker_and_redirects(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(job_broker, "remove_from_pipeline",
                         lambda job_posting_id: captured.update(id=job_posting_id))
    resp = client.post("/pipeline/job-1/remove", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/pipeline"
    assert captured == {"id": "job-1"}


# ----------------------------------------------------------------------------
# Centralized LakebaseError handler — must not leak raw exception text
# ----------------------------------------------------------------------------

def test_lakebase_error_handler_hides_raw_detail_and_returns_503(client, monkeypatch):
    def boom():
        raise job_broker.LakebaseError("connection to host db.internal.example port 5432 failed: secret-detail")

    monkeypatch.setattr(job_broker, "get_pipeline_stats", boom)

    resp = client.get("/")
    assert resp.status_code == 503
    assert "secret-detail" not in resp.text
    assert "db.internal.example" not in resp.text
    assert "went wrong" in resp.text.lower()
