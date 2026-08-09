"""
job_broker.py is pure orchestration — it has no SQL of its own (that's the
whole point of the broker pattern: main.py and mcp_tools.py both call into
here instead of lakebase/adzuna_client/embeddings directly). The one
exception is draft_cover_letter's direct call to the AI Gateway, which these
tests monkeypatch alongside lakebase/embeddings/adzuna_client, so nothing
here touches a database, the network, or a real Databricks workspace.
"""

import pytest
import requests as requests_module

import job_broker


# ----------------------------------------------------------------------------
# fetch_new_postings
# ----------------------------------------------------------------------------

class _FakeAdzunaClient:
    def __init__(self, app_id, app_key):
        self.app_id = app_id
        self.app_key = app_key

    def search(self, what, where, results_per_page, max_pages):
        return [
            {"id": "1", "title": f"{what} I", "description": "desc", "location": where or "Unknown",
             "company": "Acme"},
        ]


def test_fetch_new_postings_classifies_embeds_and_stores(monkeypatch):
    monkeypatch.setenv("ADZUNA_APP_ID", "id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "key")
    monkeypatch.setattr(job_broker, "AdzunaClient", _FakeAdzunaClient)

    upserted_postings = []
    upserted_embeddings = []
    monkeypatch.setattr(job_broker.lakebase, "upsert_job_postings",
                         lambda postings: upserted_postings.extend(postings) or len(postings))
    monkeypatch.setattr(job_broker.lakebase, "upsert_job_embeddings",
                         lambda rows: upserted_embeddings.extend(rows) or len(rows))
    monkeypatch.setattr(job_broker.embeddings, "embed_posting_chunks",
                         lambda posting: [(0, "chunk", [0.1, 0.2])])

    result = job_broker.fetch_new_postings("data engineer", where="Austin", max_results=5)

    assert result == {"fetched": 1, "written": 1}
    assert len(upserted_postings) == 1
    # classify.classify_posting should have run — postings get signal fields.
    assert "sponsorship_signal" in upserted_postings[0]
    assert "work_mode_signal" in upserted_postings[0]
    assert upserted_embeddings == [("1", 0, "chunk", [0.1, 0.2])]


# ----------------------------------------------------------------------------
# search / recommend / browse — thin delegations
# ----------------------------------------------------------------------------

def test_search_jobs_embeds_query_then_delegates_to_lakebase(monkeypatch):
    monkeypatch.setattr(job_broker.embeddings, "embed", lambda text: [9.0])
    captured = {}

    def fake_search(vector, top_k):
        captured["vector"] = vector
        captured["top_k"] = top_k
        return ["result"]

    monkeypatch.setattr(job_broker.lakebase, "search_jobs_semantic", fake_search)

    result = job_broker.search_jobs("backend roles", top_k=7)

    assert result == ["result"]
    assert captured == {"vector": [9.0], "top_k": 7}


def test_get_recommended_jobs_delegates_to_rank_jobs_for_profile(monkeypatch):
    monkeypatch.setattr(job_broker.lakebase, "rank_jobs_for_profile", lambda top_k: ["ranked"])
    assert job_broker.get_recommended_jobs(top_k=3) == ["ranked"]


def test_browse_jobs_passes_through_filters_and_returns_tuple(monkeypatch):
    captured = {}

    def fake_browse(**kwargs):
        captured.update(kwargs)
        return (["job"], True)

    monkeypatch.setattr(job_broker.lakebase, "browse_jobs", fake_browse)

    jobs, has_more = job_broker.browse_jobs(min_salary=1000, work_mode="remote",
                                             sponsorship_only=True, sort_by="oldest",
                                             limit=25, offset=25)

    assert jobs == ["job"]
    assert has_more is True
    assert captured == {
        "min_salary": 1000, "work_mode": "remote", "sponsorship_only": True,
        "sort_by": "oldest", "limit": 25, "offset": 25,
    }


# ----------------------------------------------------------------------------
# pipeline — remove
# ----------------------------------------------------------------------------

def test_remove_from_pipeline_delegates_to_lakebase(monkeypatch):
    captured = {}
    monkeypatch.setattr(job_broker.lakebase, "delete_application",
                         lambda job_posting_id: captured.update(id=job_posting_id) or {"id": 1})

    result = job_broker.remove_from_pipeline("job-1")

    assert result == {"id": 1}
    assert captured == {"id": "job-1"}


# ----------------------------------------------------------------------------
# save_profile — embeds the assembled profile text
# ----------------------------------------------------------------------------

def test_save_profile_embeds_profile_text_before_saving(monkeypatch):
    monkeypatch.setattr(job_broker.embeddings, "build_profile_text", lambda p: "assembled text")
    monkeypatch.setattr(job_broker.embeddings, "embed", lambda text: {"assembled text": [1.0]}[text])

    captured = {}
    monkeypatch.setattr(job_broker.lakebase, "save_profile",
                         lambda profile, embedding=None: captured.update(profile=profile, embedding=embedding))

    job_broker.save_profile({"full_name": "Alex"})

    assert captured["embedding"] == [1.0]
    assert captured["profile"] == {"full_name": "Alex"}


# ----------------------------------------------------------------------------
# draft_cover_letter
# ----------------------------------------------------------------------------

def test_draft_cover_letter_raises_when_posting_missing(monkeypatch):
    monkeypatch.setattr(job_broker.lakebase, "get_job_posting", lambda pid: None)
    with pytest.raises(ValueError, match="No posting found"):
        job_broker.draft_cover_letter("missing-id")


def test_draft_cover_letter_raises_when_no_profile(monkeypatch):
    monkeypatch.setattr(job_broker.lakebase, "get_job_posting",
                         lambda pid: {"id": pid, "title": "DE", "company": "Acme", "description": "desc"})
    monkeypatch.setattr(job_broker.lakebase, "get_profile", lambda: None)
    with pytest.raises(ValueError, match="No profile saved"):
        job_broker.draft_cover_letter("job-1")


class _FakeConfig:
    host = "https://example.cloud.databricks.com"

    def authenticate(self):
        return {"Authorization": "Bearer fake-token"}


class _FakeWorkspaceClient:
    """Stand-in for databricks.sdk.WorkspaceClient — only .config is used
    by draft_cover_letter now, since the actual chat call goes through
    requests.post directly against the AI Gateway."""
    def __init__(self):
        self.config = _FakeConfig()


class _FakeResponse:
    def __init__(self, content, status_ok=True):
        self._content = content
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise requests_module.HTTPError("404 model not found")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_draft_cover_letter_saves_and_returns_draft(monkeypatch):
    monkeypatch.setattr(job_broker.lakebase, "get_job_posting",
                         lambda pid: {"id": pid, "title": "DE", "company": "Acme", "description": "desc"})
    monkeypatch.setattr(job_broker.lakebase, "get_profile", lambda: {"full_name": "Alex"})
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setattr(job_broker.requests, "post",
                         lambda *a, **k: _FakeResponse("Dear hiring manager, ..."))

    saved = {}
    monkeypatch.setattr(job_broker.lakebase, "save_cover_letter",
                         lambda pid, draft: saved.update(job_posting_id=pid, draft=draft))

    draft = job_broker.draft_cover_letter("job-1")

    assert draft == "Dear hiring manager, ..."
    assert saved == {"job_posting_id": "job-1", "draft": "Dear hiring manager, ..."}


def test_draft_cover_letter_wraps_serving_endpoint_failure(monkeypatch):
    """A raw HTTP/SDK exception (e.g. model not found/enabled) should surface
    as a clear, actionable RuntimeError — not an opaque traceback the MCP
    tool or web route has no way to interpret."""
    monkeypatch.setattr(job_broker.lakebase, "get_job_posting",
                         lambda pid: {"id": pid, "title": "DE", "company": "Acme", "description": "desc"})
    monkeypatch.setattr(job_broker.lakebase, "get_profile", lambda: {"full_name": "Alex"})
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setattr(job_broker.requests, "post",
                         lambda *a, **k: _FakeResponse(None, status_ok=False))

    with pytest.raises(RuntimeError, match="Cover letter drafting failed"):
        job_broker.draft_cover_letter("job-1")


# ----------------------------------------------------------------------------
# chat_with_agent
# ----------------------------------------------------------------------------

class _FakeResponsesResponse:
    """Stands in for the agent endpoint's Responses-API-shaped reply:
    {"output": [{"content": [{"text": "..."}]}]}."""
    def __init__(self, text, status_ok=True):
        self._text = text
        self._status_ok = status_ok
        self.status_code = 200 if status_ok else 503
        self.text = text if isinstance(text, str) else ""

    def raise_for_status(self):
        if not self._status_ok:
            raise requests_module.HTTPError("503 agent endpoint unavailable")

    def json(self):
        return {"output": [{"content": [{"text": self._text}]}]}


class _FakeNonJsonResponse:
    """A 200 whose body isn't valid JSON (e.g. empty), which is what a plain
    response.json() call turns into an opaque 'Expecting value' error."""
    status_code = 200
    text = ""

    def raise_for_status(self):
        pass

    def json(self):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


def test_chat_with_agent_returns_reply(monkeypatch):
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponsesResponse("Here are three roles that match your profile.")

    monkeypatch.setattr(job_broker.requests, "post", fake_post)

    messages = [{"role": "user", "content": "What should I apply to?"}]
    reply = job_broker.chat_with_agent(messages)

    assert reply == "Here are three roles that match your profile."
    assert captured["url"].endswith("/serving-endpoints/responses")
    assert captured["json"]["input"] == messages


def test_chat_with_agent_wraps_failure(monkeypatch):
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setattr(job_broker.requests, "post",
                         lambda *a, **k: _FakeResponsesResponse(None, status_ok=False))

    with pytest.raises(RuntimeError, match="Chat request to the agent failed"):
        job_broker.chat_with_agent([{"role": "user", "content": "hi"}])


def test_chat_with_agent_extracts_message_from_sse_error_frame(monkeypatch):
    """Regression test for the real failure hit in production: the agent
    endpoint replied 200 but as an SSE error frame (tool registration
    failing against this app's own /mcp), not plain JSON."""
    sse_body = (
        'event: error\n'
        'data: {"error_code": "INVALID_PARAMETER_VALUE", '
        '"message": "Failed to register tools from Databricks App MCP '
        'server \'mcp-job-hunting-copilot\': HTTP 401"}\n\n'
        'data: [DONE]\n\n'
    )

    class _FakeSseResponse:
        status_code = 200
        text = sse_body

        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setattr(job_broker.requests, "post", lambda *a, **k: _FakeSseResponse())

    with pytest.raises(RuntimeError, match="Failed to register tools.*HTTP 401"):
        job_broker.chat_with_agent([{"role": "user", "content": "hi"}])


def test_chat_with_agent_wraps_non_json_response_with_status_and_body(monkeypatch):
    """A 200 with an unparseable (often empty) body previously surfaced as
    a bare 'Expecting value' error with no way to tell what actually came
    back. This should include the real status code and raw body instead."""
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setattr(job_broker.requests, "post", lambda *a, **k: _FakeNonJsonResponse())

    with pytest.raises(RuntimeError, match="non-JSON response \\(status 200\\)"):
        job_broker.chat_with_agent([{"role": "user", "content": "hi"}])


def test_chat_with_agent_raises_on_empty_reply(monkeypatch):
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setattr(job_broker.requests, "post",
                         lambda *a, **k: _FakeResponsesResponse(""))

    with pytest.raises(RuntimeError, match="no text output"):
        job_broker.chat_with_agent([{"role": "user", "content": "hi"}])
