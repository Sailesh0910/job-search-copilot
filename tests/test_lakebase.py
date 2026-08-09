"""
No real Postgres connection is ever made here. Two layers are tested
separately:

  - get_connection()/_checkout_healthy_connection(): tested against a fake
    pool object, verifying the resilience logic itself (retry-once on a
    stale connection, discard only on OperationalError).
  - Everything built on read_cursor()/transaction() (queries, upserts,
    profile save): tested by monkeypatching get_connection() to yield a
    FakeConnection/FakeCursor, so the real SQL-building logic in lakebase.py
    runs, but against an in-memory double instead of psycopg2.
"""

import os
from contextlib import contextmanager

import psycopg2
import pytest

import lakebase


# ----------------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, rows=None, one_row=None):
        self.executed = []
        self._rows = rows if rows is not None else []
        self._one_row = one_row

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._one_row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


@pytest.fixture
def fake_conn(monkeypatch):
    cursor = FakeCursor()
    conn = FakeConnection(cursor)

    @contextmanager
    def fake_get_connection():
        yield conn

    monkeypatch.setattr(lakebase, "get_connection", fake_get_connection)
    return conn


# ----------------------------------------------------------------------------
# schema.sql resolution — this is the exact bug that blocked the app from
# starting at all: ensure_schema() looked for app/schema.sql while the file
# lived at sql/schema.sql.
# ----------------------------------------------------------------------------

def test_schema_sql_lives_beside_lakebase_py():
    assert os.path.exists(lakebase._SCHEMA_PATH)


def test_ensure_schema_executes_the_real_schema_file(fake_conn):
    lakebase.ensure_schema()
    sql, _params = fake_conn._cursor.executed[-1]
    assert "CREATE SCHEMA IF NOT EXISTS job_copilot" in sql
    assert fake_conn.committed


# ----------------------------------------------------------------------------
# search_jobs_semantic — the fixed ranking/dedup logic
# ----------------------------------------------------------------------------

def test_search_jobs_semantic_dedupes_keeping_best_chunk_per_posting(fake_conn, monkeypatch):
    monkeypatch.setattr(lakebase, "get_profile", lambda: None)
    # Rows already best-to-worst, as ORDER BY embedding <=> vector returns.
    fake_conn._cursor._rows = [
        {"id": "A", "similarity": 0.9, "chunk_text": "best A"},
        {"id": "B", "similarity": 0.85, "chunk_text": "best B"},
        {"id": "A", "similarity": 0.7, "chunk_text": "worse A chunk"},
        {"id": "C", "similarity": 0.6, "chunk_text": "best C"},
    ]
    results = lakebase.search_jobs_semantic([0.1, 0.2], top_k=2, apply_profile_filters=False)
    assert [r["id"] for r in results] == ["A", "B"]
    assert results[0]["chunk_text"] == "best A"


def test_search_jobs_semantic_query_orders_by_distance_and_clamps_top_k(fake_conn, monkeypatch):
    monkeypatch.setattr(lakebase, "get_profile", lambda: None)
    fake_conn._cursor._rows = []
    lakebase.search_jobs_semantic([0.1], top_k=1000, apply_profile_filters=False)
    sql, params = fake_conn._cursor.executed[-1]
    assert params[-1] == 250  # top_k clamped to 50, fetch_limit = min(50*5, 250)
    assert sql.index("ORDER BY e.embedding") < sql.index("LIMIT")


def test_search_jobs_semantic_applies_profile_filters(fake_conn, monkeypatch):
    monkeypatch.setattr(lakebase, "get_profile", lambda: {
        "min_salary": 100000, "remote_preference": "remote",
        "location_preference": "any", "sponsorship_required": True,
    })
    fake_conn._cursor._rows = []
    lakebase.search_jobs_semantic([0.1], top_k=5)
    sql, params = fake_conn._cursor.executed[-1]
    assert "salary_max" in sql
    assert "work_mode_signal" in sql
    assert "sponsorship_signal" in sql
    assert 100000 in params


def test_search_jobs_semantic_uses_passed_profile_without_refetching(fake_conn, monkeypatch):
    """rank_jobs_for_profile() already has the profile in hand (it needs it
    for the resume embedding) — passing it through should skip the second
    get_profile() call search_jobs_semantic would otherwise make."""
    def fail_if_called():
        raise AssertionError("get_profile() should not be called when profile= is passed")

    monkeypatch.setattr(lakebase, "get_profile", fail_if_called)
    fake_conn._cursor._rows = []

    lakebase.search_jobs_semantic(
        [0.1], top_k=5,
        profile={"min_salary": 50000, "remote_preference": "any",
                 "location_preference": "any", "sponsorship_required": False},
    )
    sql, params = fake_conn._cursor.executed[-1]
    assert "salary_max" in sql
    assert 50000 in params


# ----------------------------------------------------------------------------
# browse_jobs
# ----------------------------------------------------------------------------

def test_browse_jobs_clamps_limit(fake_conn):
    fake_conn._cursor._rows = []
    lakebase.browse_jobs(limit=99999)
    params = fake_conn._cursor.executed[-1][1]
    assert params[-2] == 201  # clamped limit (200) + 1, to probe for a next page
    assert params[-1] == 0    # default offset

    lakebase.browse_jobs(limit=0)
    assert fake_conn._cursor.executed[-1][1][-2] == 2  # clamped limit (1) + 1


def test_browse_jobs_returns_has_more_when_extra_row_present(fake_conn):
    fake_conn._cursor._rows = [{"id": str(i)} for i in range(6)]
    jobs, has_more = lakebase.browse_jobs(limit=5)
    assert len(jobs) == 5
    assert has_more is True


def test_browse_jobs_no_more_when_fewer_rows_than_limit(fake_conn):
    fake_conn._cursor._rows = [{"id": str(i)} for i in range(3)]
    jobs, has_more = lakebase.browse_jobs(limit=5)
    assert len(jobs) == 3
    assert has_more is False


def test_browse_jobs_passes_offset(fake_conn):
    fake_conn._cursor._rows = []
    lakebase.browse_jobs(limit=10, offset=20)
    sql, params = fake_conn._cursor.executed[-1]
    assert "OFFSET %s" in sql
    assert params[-2:] == [11, 20]


def test_browse_jobs_builds_where_clause_from_filters(fake_conn):
    fake_conn._cursor._rows = []
    lakebase.browse_jobs(min_salary=80000, work_mode="remote", sponsorship_only=True, sort_by="oldest")
    sql, params = fake_conn._cursor.executed[-1]
    assert "salary_max" in sql
    assert "work_mode_signal = %s" in sql
    assert "sponsorship_signal != 'no_sponsorship_stated'" in sql
    assert "ORDER BY posted_at ASC" in sql
    assert 80000 in params and "remote" in params


def test_browse_jobs_no_filters_omits_where_clause(fake_conn):
    fake_conn._cursor._rows = []
    lakebase.browse_jobs()
    sql, _params = fake_conn._cursor.executed[-1]
    assert "WHERE" not in sql
    assert "ORDER BY posted_at DESC" in sql


# ----------------------------------------------------------------------------
# save_profile — insert vs. update branch
# ----------------------------------------------------------------------------

def test_save_profile_inserts_when_no_existing_profile(fake_conn):
    fake_conn._cursor._one_row = None
    lakebase.save_profile({"full_name": "Alex"}, embedding=[0.1, 0.2])
    sql, _params = fake_conn._cursor.executed[-1]
    assert sql.strip().startswith("INSERT INTO job_copilot.profile")
    assert fake_conn.committed


def test_save_profile_updates_when_existing_profile(fake_conn):
    fake_conn._cursor._one_row = {"id": 7}
    lakebase.save_profile({"full_name": "Alex"}, embedding=None)
    sql, params = fake_conn._cursor.executed[-1]
    assert sql.strip().startswith("UPDATE job_copilot.profile")
    assert params[-1] == 7


# ----------------------------------------------------------------------------
# upserts
# ----------------------------------------------------------------------------

def test_upsert_job_postings_empty_list_short_circuits(fake_conn):
    assert lakebase.upsert_job_postings([]) == 0
    assert fake_conn._cursor.executed == []


def test_upsert_job_postings_calls_execute_values(fake_conn, monkeypatch):
    calls = []
    monkeypatch.setattr(lakebase, "execute_values", lambda cur, sql, rows, **kw: calls.append((sql, rows, kw)))
    postings = [{"id": "1", "title": "DE", "description": "d"}]
    n = lakebase.upsert_job_postings(postings)
    assert n == 1
    sql, rows, _kw = calls[0]
    assert "ON CONFLICT (id) DO UPDATE" in sql
    assert rows[0][0] == "1"
    assert fake_conn.committed


def test_upsert_job_embeddings_formats_vector_literal(fake_conn, monkeypatch):
    calls = []
    monkeypatch.setattr(lakebase, "execute_values", lambda cur, sql, rows, **kw: calls.append((sql, rows, kw)))
    n = lakebase.upsert_job_embeddings([("job-1", 0, "chunk text", [0.1, 0.2, 0.3])])
    assert n == 1
    _sql, formatted, kw = calls[0]
    assert formatted[0][3] == "[0.1,0.2,0.3]"
    assert f"vector({lakebase.EMBEDDING_DIM})" in kw["template"]


def test_upsert_job_embeddings_empty_list_short_circuits(fake_conn):
    assert lakebase.upsert_job_embeddings([]) == 0


# ----------------------------------------------------------------------------
# get_pipeline / delete_application
# ----------------------------------------------------------------------------

def test_get_pipeline_selects_posted_at_and_stale_flag_with_default_threshold(fake_conn):
    fake_conn._cursor._rows = []
    lakebase.get_pipeline()
    sql, params = fake_conn._cursor.executed[-1]
    assert "posting_possibly_stale" in sql
    assert "p.posted_at" in sql
    assert params[0] == 30  # default stale_posting_days


def test_get_pipeline_custom_stale_threshold_and_status_filter(fake_conn):
    fake_conn._cursor._rows = []
    lakebase.get_pipeline(status="applied", stale_posting_days=45)
    sql, params = fake_conn._cursor.executed[-1]
    assert "WHERE a.status = %s" in sql
    assert params == [45, "applied"]


def test_delete_application_returns_deleted_id(fake_conn):
    fake_conn._cursor._one_row = {"id": 7}
    result = lakebase.delete_application("job-1")
    sql, params = fake_conn._cursor.executed[-1]
    assert sql.strip().startswith("DELETE FROM job_copilot.applications")
    assert params == ("job-1",)
    assert result == {"id": 7}
    assert fake_conn.committed


def test_delete_application_returns_none_when_nothing_to_remove(fake_conn):
    fake_conn._cursor._one_row = None
    assert lakebase.delete_application("missing-id") is None


# ----------------------------------------------------------------------------
# Connection pool resilience — the actual retry/discard logic
# ----------------------------------------------------------------------------

class FakePool:
    def __init__(self, connections):
        self._connections = list(connections)
        self.putconn_calls = []

    def getconn(self):
        return self._connections.pop(0)

    def putconn(self, conn, close=False):
        self.putconn_calls.append((conn, close))


class _StubCursor:
    def __init__(self, raise_exc=None):
        self._raise_exc = raise_exc

    def execute(self, sql):
        if self._raise_exc:
            raise self._raise_exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _StubConn:
    def __init__(self, healthy=True):
        self.healthy = healthy

    def cursor(self):
        return _StubCursor(raise_exc=None if self.healthy else psycopg2.OperationalError("stale"))


def test_checkout_healthy_connection_returns_immediately_when_healthy():
    conn = _StubConn(healthy=True)
    pool = FakePool([conn])
    assert lakebase._checkout_healthy_connection(pool) is conn
    assert pool.putconn_calls == []


def test_checkout_healthy_connection_discards_stale_connection_and_retries():
    dead = _StubConn(healthy=False)
    healthy = _StubConn(healthy=True)
    pool = FakePool([dead, healthy])
    assert lakebase._checkout_healthy_connection(pool) is healthy
    assert pool.putconn_calls == [(dead, True)]


def test_checkout_healthy_connection_raises_after_two_failures():
    dead1, dead2 = _StubConn(healthy=False), _StubConn(healthy=False)
    pool = FakePool([dead1, dead2])
    with pytest.raises(psycopg2.OperationalError):
        lakebase._checkout_healthy_connection(pool)
    assert pool.putconn_calls == [(dead1, True), (dead2, True)]


def test_get_connection_returns_conn_to_pool_on_success(monkeypatch):
    conn = _StubConn(healthy=True)
    pool = FakePool([conn])
    monkeypatch.setattr(lakebase, "_get_pool", lambda: pool)
    with lakebase.get_connection() as c:
        assert c is conn
    assert pool.putconn_calls == [(conn, False)]


def test_get_connection_discards_conn_on_operational_error(monkeypatch):
    conn = _StubConn(healthy=True)
    pool = FakePool([conn])
    monkeypatch.setattr(lakebase, "_get_pool", lambda: pool)
    with pytest.raises(psycopg2.OperationalError):
        with lakebase.get_connection():
            raise psycopg2.OperationalError("dropped")
    assert pool.putconn_calls == [(conn, True)]


def test_get_connection_keeps_conn_on_ordinary_query_error(monkeypatch):
    """A duplicate-key or bad-data error doesn't mean the connection itself
    is broken — it should stay in the pool for reuse after the caller's own
    rollback, not be thrown away."""
    conn = _StubConn(healthy=True)
    pool = FakePool([conn])
    monkeypatch.setattr(lakebase, "_get_pool", lambda: pool)
    with pytest.raises(psycopg2.IntegrityError):
        with lakebase.get_connection():
            raise psycopg2.IntegrityError("duplicate key")
    assert pool.putconn_calls == [(conn, False)]
