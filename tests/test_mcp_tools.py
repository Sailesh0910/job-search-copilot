"""
mcp_tools.py wraps job_broker for the agent, mostly thin (see test_job_broker.py
for the underlying logic). remove_saved_job is the one tool with its own real
logic worth testing directly: a two-call confirmation gate so a single tool
call, however it was triggered, can never delete data. job_broker is
monkeypatched throughout, no database involved.
"""

import mcp_tools


def _posting(job_posting_id="job-1"):
    return {"id": job_posting_id, "title": "Data Engineer", "company": "Acme"}


def test_remove_saved_job_first_call_returns_confirmation_and_does_not_delete(monkeypatch):
    monkeypatch.setattr(mcp_tools.job_broker, "get_job_posting", lambda pid: _posting(pid))
    called = {"removed": False}
    monkeypatch.setattr(mcp_tools.job_broker, "remove_from_pipeline",
                         lambda pid: called.update(removed=True))

    result = mcp_tools.remove_saved_job("job-1")

    assert result["confirmation_required"] is True
    assert "confirmation_token" in result
    assert result["posting"] == {"title": "Data Engineer", "company": "Acme"}
    assert called["removed"] is False


def test_remove_saved_job_missing_posting_returns_error_without_a_token(monkeypatch):
    monkeypatch.setattr(mcp_tools.job_broker, "get_job_posting", lambda pid: None)

    result = mcp_tools.remove_saved_job("missing-id")

    assert result == {"error": "No posting found with id missing-id"}
    assert "confirmation_token" not in result


def test_remove_saved_job_wrong_token_does_not_delete(monkeypatch):
    monkeypatch.setattr(mcp_tools.job_broker, "get_job_posting", lambda pid: _posting(pid))
    called = {"removed": False}
    monkeypatch.setattr(mcp_tools.job_broker, "remove_from_pipeline",
                         lambda pid: called.update(removed=True))

    result = mcp_tools.remove_saved_job("job-1", confirmation_token="guessed-wrong")

    assert result["confirmation_required"] is True
    assert called["removed"] is False


def test_remove_saved_job_correct_token_deletes(monkeypatch):
    monkeypatch.setattr(mcp_tools.job_broker, "get_job_posting", lambda pid: _posting(pid))
    monkeypatch.setattr(mcp_tools.job_broker, "remove_from_pipeline", lambda pid: {"id": 1})

    first = mcp_tools.remove_saved_job("job-1")
    second = mcp_tools.remove_saved_job("job-1", confirmation_token=first["confirmation_token"])

    assert second == {"removed": True}


def test_remove_saved_job_token_is_specific_to_the_posting_id(monkeypatch):
    monkeypatch.setattr(mcp_tools.job_broker, "get_job_posting", lambda pid: _posting(pid))

    token_for_job_1 = mcp_tools.remove_saved_job("job-1")["confirmation_token"]
    monkeypatch.setattr(mcp_tools.job_broker, "remove_from_pipeline", lambda pid: {"id": 1})

    result = mcp_tools.remove_saved_job("job-2", confirmation_token=token_for_job_1)

    assert result["confirmation_required"] is True


def test_remove_saved_job_wraps_unexpected_exception_as_error(monkeypatch):
    def boom(pid):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(mcp_tools.job_broker, "get_job_posting", boom)

    result = mcp_tools.remove_saved_job("job-1")

    assert result == {"error": "db unreachable"}
