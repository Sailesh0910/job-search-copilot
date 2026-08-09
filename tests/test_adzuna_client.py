"""
No real HTTP call ever leaves this test file — client.session.get and
client._make_request are monkeypatched directly, so no Adzuna credentials
are needed to run these.
"""

import pytest
import requests

from adzuna_client import AdzunaClient


def _client():
    return AdzunaClient(app_id="test-id", app_key="test-key")


def test_normalize_posting_maps_fields_and_rounds_salary_to_int():
    raw = {
        "id": "123", "title": "DE", "company": {"display_name": "Acme"},
        "location": {"display_name": "Austin"}, "description": "desc",
        "salary_min": 89999.6, "salary_max": 120000.4,
        "redirect_url": "http://x", "created": "2024-01-01T00:00:00Z",
    }
    n = _client()._normalize_posting(raw)
    assert n["id"] == "123"
    assert n["title"] == "DE"
    assert n["company"] == "Acme"
    assert n["location"] == "Austin"
    assert n["salary_min"] == 90000
    assert n["salary_max"] == 120000
    assert isinstance(n["salary_min"], int)
    assert isinstance(n["salary_max"], int)


def test_normalize_posting_handles_missing_salary_and_fields():
    n = _client()._normalize_posting({"id": "1", "title": "DE"})
    assert n["salary_min"] is None
    assert n["salary_max"] is None
    assert n["company"] == "Unknown"
    assert n["location"] == "Unknown"


def test_normalize_posting_fallback_id_is_deterministic_and_location_sensitive():
    client = _client()
    raw_austin = {"title": "DE", "company": {"display_name": "Acme"}, "location": {"display_name": "Austin"}}
    raw_raleigh = {"title": "DE", "company": {"display_name": "Acme"}, "location": {"display_name": "Raleigh"}}

    n1a = client._normalize_posting(dict(raw_austin))
    n1b = client._normalize_posting(dict(raw_austin))
    n2 = client._normalize_posting(raw_raleigh)

    assert n1a["id"] == n1b["id"]  # same input -> same fallback id (dedup works)
    assert n1a["id"] != n2["id"]   # different location -> different id


def test_make_request_retries_then_succeeds(monkeypatch):
    client = _client()
    calls = {"n": 0}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": []}

    def fake_get(url, params, timeout):
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.exceptions.ConnectionError("boom")
        return FakeResponse()

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr("adzuna_client.time.sleep", lambda s: None)

    data = client._make_request(1, {})
    assert data == {"results": []}
    assert calls["n"] == 2


def test_make_request_raises_after_exhausting_retries(monkeypatch):
    client = _client()

    def fake_get(url, params, timeout):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr("adzuna_client.time.sleep", lambda s: None)

    with pytest.raises(requests.exceptions.ConnectionError):
        client._make_request(1, {}, max_retries=2)


def test_search_stops_paging_when_a_page_returns_no_results(monkeypatch):
    client = _client()
    call_count = {"n": 0}

    def fake_make_request(page, params):
        call_count["n"] += 1
        if page == 1:
            return {"results": [{"id": "1", "title": "DE", "company": {}, "location": {}, "description": "d"}]}
        return {"results": []}

    monkeypatch.setattr(client, "_make_request", fake_make_request)
    postings = client.search(what="data engineer", max_pages=5)

    assert len(postings) == 1
    assert call_count["n"] == 2  # stopped after the second (empty) page


def test_search_continues_past_a_failed_page(monkeypatch):
    """A single page failing shouldn't kill the whole sync — the loop should
    stop paging (can't know what page N+1 holds) but return what it has."""
    client = _client()

    def fake_make_request(page, params):
        if page == 1:
            return {"results": [{"id": "1", "title": "DE", "company": {}, "location": {}, "description": "d"}]}
        raise requests.exceptions.RequestException("network blip")

    monkeypatch.setattr(client, "_make_request", fake_make_request)
    postings = client.search(what="data engineer", max_pages=3)

    assert len(postings) == 1
