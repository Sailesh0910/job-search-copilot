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
    """Stands in for the agent endpoint's invocations reply:
    {"output": [{"type": "message", "role": "assistant", "content": [{"text": "..."}]}]}."""
    def __init__(self, text, status_ok=True):
        self._text = text
        self._status_ok = status_ok
        self.status_code = 200 if status_ok else 503
        self.text = text if isinstance(text, str) else ""

    def raise_for_status(self):
        if not self._status_ok:
            raise requests_module.HTTPError("503 agent endpoint unavailable")

    def json(self):
        return {"output": [{"type": "message", "role": "assistant", "content": [{"text": self._text}]}]}


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
        # chat_with_agent mutates the same conversation list in place after
        # this call returns — snapshot the input list now, or the assertion
        # below would see later appends too.
        captured["json"] = {**json, "input": list(json["input"])}
        return _FakeResponsesResponse("Here are three roles that match your profile.")

    monkeypatch.setattr(job_broker.requests, "post", fake_post)

    conversation = [{"role": "user", "content": "What should I apply to?"}]
    reply = job_broker.chat_with_agent(conversation)

    assert reply == "Here are three roles that match your profile."
    assert captured["url"].endswith(f"/serving-endpoints/{job_broker.SUPERVISOR_AGENT_ENDPOINT}/invocations")
    # The endpoint has no previous_response_id continuation — every call
    # resends the whole conversation, confirmed against Playground's own
    # network traffic.
    assert "previous_response_id" not in captured["json"]
    assert captured["json"]["input"] == [{"role": "user", "content": "What should I apply to?"}]
    # chat_with_agent mutates the conversation list in place, appending the
    # agent's reply — the caller (main.py) relies on this to build next
    # turn's full history.
    assert conversation[-1]["type"] == "message"


def test_chat_with_agent_wraps_failure(monkeypatch):
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setattr(job_broker.requests, "post",
                         lambda *a, **k: _FakeResponsesResponse(None, status_ok=False))

    with pytest.raises(RuntimeError, match="Chat request to the agent failed"):
        job_broker.chat_with_agent([{"role": "user", "content": "hi"}])


class _FakeApprovalFlowResponse:
    """A canned .json()-returning response used to script a fixed sequence
    of raw response bodies for the tool-execution loop."""
    def __init__(self, body):
        self._body = body
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_chat_with_agent_runs_approved_tool_and_resends_full_history(monkeypatch):
    """Regression test for the real stuck-chat bug, and for the real fix:
    inspecting Agent Bricks Playground's own network traffic showed this
    endpoint (a) has no previous_response_id continuation — the full
    conversation is resent every call — and (b) never executes MCP tools
    itself after approval; the caller has to run the tool and report back a
    function_call_output. Earlier attempts assuming the OpenAI Responses API
    spec (server-side execution, previous_response_id chaining) all failed
    against the real endpoint with "Invalid message sequence" errors."""
    import mcp_tools

    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setitem(mcp_tools.TOOL_DISPATCH, "check_stale_applications",
                         lambda days=14: {"stale": []})

    first_response = {
        "id": "resp_1",
        "output": [
            {"type": "message", "role": "assistant", "content": [{"text": "I'll check that for you."}]},
            {"type": "mcp_approval_request", "id": "mcpr_1", "name": "check_stale_applications",
             "arguments": '{"days": 14}'},
        ],
    }
    final_response = {
        "id": "resp_2",
        "output": [{"type": "message", "role": "assistant", "content": [{"text": "Nothing needs follow-up."}]}],
    }
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({**json, "input": list(json["input"])})
        body = first_response if len(calls) == 1 else final_response
        return _FakeApprovalFlowResponse(body)

    monkeypatch.setattr(job_broker.requests, "post", fake_post)

    conversation = [{"role": "user", "content": "Do I have pending work?"}]
    reply = job_broker.chat_with_agent(conversation)

    assert reply == "Nothing needs follow-up."
    assert len(calls) == 2
    # No previous_response_id — the second call resends everything: the
    # original user turn, the first response's own output items, plus the
    # approval response and the tool's real output this function ran itself.
    assert "previous_response_id" not in calls[1]
    assert calls[1]["input"] == [
        {"role": "user", "content": "Do I have pending work?"},
        {"type": "message", "role": "assistant", "content": [{"text": "I'll check that for you."}]},
        {"type": "mcp_approval_request", "id": "mcpr_1", "name": "check_stale_applications",
         "arguments": '{"days": 14}'},
        {"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": True},
        {"type": "function_call_output", "call_id": "mcpr_1", "name": "check_stale_applications",
         "output": '{"stale": []}'},
    ]


def test_chat_with_agent_resolves_parallel_approvals_one_at_a_time(monkeypatch):
    """Regression test for a real failure: when the model requests two tool
    calls in one turn (both mcp_approval_request items in the same
    response), submitting both approval+output pairs together in one
    follow-up call was rejected by the agent backend ("Invalid approval
    response. The approval response ID does not match the request."), even
    though each tool had actually already run successfully. Each pending
    approval must go out in its own round-trip."""
    import mcp_tools

    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setitem(mcp_tools.TOOL_DISPATCH, "check_stale_applications", lambda days=14: {"stale": []})
    monkeypatch.setitem(mcp_tools.TOOL_DISPATCH, "view_pipeline", lambda status=None: {"applications": []})

    first_response = {
        "output": [
            {"type": "message", "role": "assistant", "content": [{"text": "I'll check both."}]},
            {"type": "mcp_approval_request", "id": "mcpr_1", "name": "check_stale_applications", "arguments": "{}"},
            {"type": "mcp_approval_request", "id": "mcpr_2", "name": "view_pipeline", "arguments": "{}"},
        ],
    }
    # After resolving mcpr_1, the server has no reason to repeat mcpr_2 in
    # its next reply — it already told us about it once, in the first
    # response. This empty body simulates that; chat_with_agent still has
    # to find mcpr_2 by scanning the accumulated conversation, not by
    # looking at only the latest response.
    empty_response = {"output": []}
    final_response = {
        "output": [{"type": "message", "role": "assistant", "content": [{"text": "All clear."}]}],
    }
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({**json, "input": list(json["input"])})
        body = [first_response, empty_response, final_response][len(calls) - 1]
        return _FakeApprovalFlowResponse(body)

    monkeypatch.setattr(job_broker.requests, "post", fake_post)

    reply = job_broker.chat_with_agent([{"role": "user", "content": "Do I have pending work?"}])

    assert reply == "All clear."
    # 1 initial call + 1 per pending approval, each resolved in its own
    # round-trip rather than batched together.
    assert len(calls) == 3
    types_in_call_2 = [item.get("type") for item in calls[1]["input"]]
    assert types_in_call_2[-2:] == ["mcp_approval_response", "function_call_output"]
    # mcpr_2 must still be unresolved after call 2 — only mcpr_1 was handled.
    assert not any(
        item.get("type") == "mcp_approval_response" and item.get("approval_request_id") == "mcpr_2"
        for item in calls[1]["input"]
    )
    types_in_call_3 = [item.get("type") for item in calls[2]["input"]]
    assert types_in_call_3[-2:] == ["mcp_approval_response", "function_call_output"]
    assert calls[2]["input"][-2]["approval_request_id"] == "mcpr_2"


def test_chat_with_agent_tool_loop_is_capped(monkeypatch):
    """If the endpoint somehow keeps requesting tool calls forever, this
    must not hang the request indefinitely."""
    import mcp_tools

    monkeypatch.setattr("databricks.sdk.WorkspaceClient", _FakeWorkspaceClient)
    monkeypatch.setitem(mcp_tools.TOOL_DISPATCH, "check_stale_applications",
                         lambda days=14: {"stale": []})

    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        # A fresh id every round — a real server would too, since it's a
        # new tool call each time, not a repeat of the last one. Reusing
        # one id would make chat_with_agent's "already answered" check
        # correctly treat it as resolved after round 1, which isn't the
        # scenario this test is for.
        body = {
            "id": f"resp_loop_{len(calls)}",
            "output": [{"type": "mcp_approval_request", "id": f"mcpr_loop_{len(calls)}",
                        "name": "check_stale_applications", "arguments": "{}"}],
        }
        return _FakeApprovalFlowResponse(body)

    monkeypatch.setattr(job_broker.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="capped at 10 rounds"):
        job_broker.chat_with_agent([{"role": "user", "content": "hi"}])

    assert len(calls) == 11  # 1 initial + 10 capped approval rounds


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
