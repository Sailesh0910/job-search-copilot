"""
Lakebase (Postgres) connection and data access for the job hunting copilot.

Connection comes from a single JOB_COPILOT_CONNECTION_STRING env var, wired
in via app.yaml's valueFrom pointing at a Databricks secret. No dbutils —
deployed apps run outside a notebook context, so os.environ is the only
thing that works in both the app and the batch scripts.

Connections come from a small pooled ThreadedConnectionPool rather than one
psycopg2.connect() per call — opening a fresh TCP+TLS connection for every
query is real, avoidable latency. The pool is small (1-5 connections) since
this is a single-user app on shared, free-tier compute — concurrency here is
inherently low, so a large pool would just hold idle connections against
whatever cap the Lakebase instance has.

Error handling: every write goes through transaction(), which commits on
success and rolls back on any failure. Failures are re-raised as LakebaseError
so callers never see raw psycopg2 tracebacks.
"""

import logging
import os
import threading
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool as psycopg2_pool
from psycopg2.extras import RealDictCursor, execute_values

from config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

MODEL_NAME = EMBEDDING_MODEL_NAME

# schema.sql lives next to this file. Resolving via __file__ rather than a
# relative path from the caller's working directory means this works whether
# lakebase.py is imported from a deployed app or from a notebook — wherever
# this file physically is, schema.sql is right beside it.
_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

_POOL_MIN_CONN = 1
_POOL_MAX_CONN = 5

_pool = None
_pool_lock = threading.Lock()


class LakebaseError(Exception):
    """
    Raised for any database failure, wrapping the underlying psycopg2 error.

    Callers (web routes, MCP tools) catch this rather than psycopg2 exceptions
    directly, so the database driver stays an implementation detail and error
    messages reaching a user or an agent are meaningful rather than raw
    driver tracebacks.
    """


def _connection_string() -> str:
    conn_string = os.environ.get("JOB_COPILOT_CONNECTION_STRING")
    if not conn_string:
        raise LakebaseError(
            "JOB_COPILOT_CONNECTION_STRING is not set. In Databricks Apps this "
            "should be wired in app.yaml via a 'valueFrom' pointing at your secret."
        )
    return conn_string


def _get_pool() -> psycopg2_pool.ThreadedConnectionPool:
    """
    Lazily creates the connection pool on first use (double-checked locking,
    since FastAPI can call this from multiple request threads). Lazy rather
    than created at import time so importing lakebase.py doesn't require
    JOB_COPILOT_CONNECTION_STRING to already be set — e.g. a notebook that
    only calls track_run() shouldn't need it.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                try:
                    _pool = psycopg2_pool.ThreadedConnectionPool(
                        _POOL_MIN_CONN, _POOL_MAX_CONN,
                        _connection_string(), cursor_factory=RealDictCursor,
                    )
                except psycopg2.Error as e:
                    raise LakebaseError(f"Could not connect to Lakebase: {e}") from e
    return _pool


def close_pool() -> None:
    """Closes all pooled connections. Call on app shutdown; safe to call even
    if the pool was never created."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


def _checkout_healthy_connection(pool):
    """
    Gets a connection from the pool, validating it with a cheap round trip
    before handing it back. A connection sitting idle in the pool can go
    stale server-side (idle timeout, network blip, an expired credential) —
    a real risk for a low-traffic app that may sit idle between requests for
    a while. Catching that here means one retry gets a fresh connection
    instead of the caller's actual query failing with a confusing error.
    """
    for attempt in range(2):
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            return conn
        except psycopg2.Error:
            pool.putconn(conn, close=True)
            if attempt == 1:
                raise


@contextmanager
def get_connection():
    """Yields a pooled connection with dict-style rows; always returns it to
    the pool. A connection an OperationalError passed through is discarded
    instead of reused, since the connection itself is what's suspect —
    ordinary query errors (bad data, a constraint violation) don't indicate a
    broken connection and leave it safe to reuse after the caller's own
    rollback."""
    pool = _get_pool()
    try:
        conn = _checkout_healthy_connection(pool)
    except psycopg2.Error as e:
        raise LakebaseError(f"Could not connect to Lakebase: {e}") from e

    discard = False
    try:
        yield conn
    except psycopg2.OperationalError:
        discard = True
        raise
    finally:
        pool.putconn(conn, close=discard)


@contextmanager
def read_cursor():
    """Yields a cursor for read-only queries. No commit needed."""
    with get_connection() as conn:
        try:
            with conn.cursor() as cur:
                yield cur
        except psycopg2.Error as e:
            raise LakebaseError(f"Database read failed: {e}") from e


@contextmanager
def transaction():
    """
    Yields a cursor inside a transaction. Commits on success, rolls back on any
    exception, always returns the connection to the pool.

    Every write goes through this rather than hand-rolled commit calls, so a
    failure partway through can't leave a transaction dangling on a connection
    that then gets reused — which silently discards the work and can leave the
    database holding locks.
    """
    with get_connection() as conn:
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise LakebaseError(f"Database write failed: {e}") from e
        except Exception:
            conn.rollback()
            raise


def ensure_schema():
    """
    Applies schema.sql (every statement in it is IF NOT EXISTS, so this is
    safe to call on every app startup). Reading the DDL from one file rather
    than duplicating it here in Python means there's only one schema
    definition to keep correct — the alternative already caused one real
    drift bug during this project (pipeline_runs existed here but not in
    schema.sql) before this fix.
    """
    with open(_SCHEMA_PATH) as f:
        ddl = f.read()

    with transaction() as cur:
        # psycopg2 sends a plain string via the simple query protocol, which
        # (unlike the prepared-statement path used when parameters are passed)
        # supports multiple semicolon-separated statements in one call, so the
        # whole file executes as-is.
        cur.execute(ddl)


def upsert_job_postings(postings):
    """Batch upsert postings. Re-running a sync refreshes rows rather than
    duplicating them, since Adzuna's own id is the primary key."""
    if not postings:
        return 0

    rows = [
        (
            p["id"], p["title"], p.get("company"), p.get("location"),
            p["description"], p.get("salary_min"), p.get("salary_max"),
            p.get("url"), p.get("sponsorship_signal", "not_mentioned"),
            p.get("work_mode_signal", "not_mentioned"),
            p.get("posted_at"), p.get("synced_at"),
        )
        for p in postings
    ]

    sql = """
        INSERT INTO job_copilot.job_postings
            (id, title, company, location, description, salary_min, salary_max,
             url, sponsorship_signal, work_mode_signal, posted_at, synced_at)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            company = EXCLUDED.company,
            location = EXCLUDED.location,
            description = EXCLUDED.description,
            salary_min = EXCLUDED.salary_min,
            salary_max = EXCLUDED.salary_max,
            url = EXCLUDED.url,
            sponsorship_signal = EXCLUDED.sponsorship_signal,
            work_mode_signal = EXCLUDED.work_mode_signal,
            posted_at = EXCLUDED.posted_at,
            synced_at = EXCLUDED.synced_at;
    """

    with transaction() as cur:
        execute_values(cur, sql, rows)
    return len(rows)


def upsert_job_embeddings(rows):
    """rows: list of (job_posting_id, chunk_index, chunk_text, embedding_list)."""
    if not rows:
        return 0

    formatted = [
        (pid, idx, text, "[" + ",".join(map(str, emb)) + "]", MODEL_NAME)
        for (pid, idx, text, emb) in rows
    ]

    sql = f"""
        INSERT INTO job_copilot.job_embeddings
            (job_posting_id, chunk_index, chunk_text, embedding, model_name)
        VALUES %s
        ON CONFLICT (job_posting_id, chunk_index) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name;
    """

    with transaction() as cur:
        execute_values(cur, sql, formatted, template=f"(%s, %s, %s, %s::vector({EMBEDDING_DIM}), %s)")
    return len(formatted)


# ----------------------------------------------------------------------------
# Profile
# ----------------------------------------------------------------------------

def get_profile():
    """Returns the single profile row, or None if not set up yet."""
    with read_cursor() as cur:
        cur.execute("SELECT * FROM job_copilot.profile ORDER BY id LIMIT 1;")
        row = cur.fetchone()
        return dict(row) if row else None


def save_profile(profile, embedding=None):
    """Inserts or updates the single profile row."""
    embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None
    existing = get_profile()

    with transaction() as cur:
        if existing:
            cur.execute(f"""
                UPDATE job_copilot.profile SET
                    full_name = %s, target_roles = %s, years_experience = %s,
                    location_preference = %s, remote_preference = %s,
                    min_salary = %s, sponsorship_required = %s,
                    work_authorization = %s, tech_stack_musthaves = %s,
                    company_size_pref = %s, other_notes = %s,
                    resume_text = %s,
                    resume_embedding = COALESCE(%s::vector({EMBEDDING_DIM}), resume_embedding),
                    updated_at = NOW()
                WHERE id = %s;
            """, (
                profile.get("full_name"), profile.get("target_roles"),
                profile.get("years_experience"), profile.get("location_preference", "any"),
                profile.get("remote_preference", "any"), profile.get("min_salary"),
                profile.get("sponsorship_required", False), profile.get("work_authorization"),
                profile.get("tech_stack_musthaves"), profile.get("company_size_pref"),
                profile.get("other_notes"), profile.get("resume_text"),
                embedding_str, existing["id"],
            ))
        else:
            cur.execute(f"""
                INSERT INTO job_copilot.profile
                    (full_name, target_roles, years_experience, location_preference,
                     remote_preference, min_salary, sponsorship_required,
                     work_authorization, tech_stack_musthaves, company_size_pref,
                     other_notes, resume_text, resume_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector({EMBEDDING_DIM}));
            """, (
                profile.get("full_name"), profile.get("target_roles"),
                profile.get("years_experience"), profile.get("location_preference", "any"),
                profile.get("remote_preference", "any"), profile.get("min_salary"),
                profile.get("sponsorship_required", False), profile.get("work_authorization"),
                profile.get("tech_stack_musthaves"), profile.get("company_size_pref"),
                profile.get("other_notes"), profile.get("resume_text"), embedding_str,
            ))


# ----------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------

def search_jobs_semantic(query_embedding, top_k=10, apply_profile_filters=True, profile=None):
    """
    Semantic search over job postings, with the profile's hard filters applied
    first and cosine similarity ranking on top.

    profile: pass an already-fetched profile dict to skip re-querying it —
    rank_jobs_for_profile() already has one in hand from computing the query
    vector, so it passes it through rather than this function fetching the
    same row a second time. Callers that don't already have one (e.g. a plain
    text search) can omit it and it's fetched here as before.

    Sponsorship logic worth noting: when sponsorship is required we exclude only
    postings that EXPLICITLY say they don't sponsor. 'not_mentioned' is silence,
    not a no, and most postings never mention it either way — filtering those out
    would discard the majority of viable roles.

    Ranking approach: this ranks individual CHUNKS with an index-friendly
    `ORDER BY embedding <=> vector LIMIT`, then keeps the best chunk per
    posting in Python, in similarity order, until top_k distinct postings are
    collected. The alternative — DISTINCT ON (p.id) to collapse chunks per
    posting before ranking — forces Postgres to sort by posting id first and
    similarity second, which both breaks top-k ranking (you'd get the top_k
    postings with the lowest ids, not the best matches) and makes the HNSW
    vector index unusable, since it can only be used when the vector-distance
    ORDER BY leads the query. fetch_limit over-fetches (5x top_k, capped) so
    postings with several high-ranking chunks don't crowd out other postings
    — a small, bounded cost for correctness.
    """
    top_k = max(1, min(top_k, 50))
    fetch_limit = min(top_k * 5, 250)
    vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

    where_clauses = []
    params = [vector_str]

    if apply_profile_filters:
        if profile is None:
            profile = get_profile()
        if profile:
            if profile.get("min_salary"):
                where_clauses.append("(p.salary_max IS NULL OR p.salary_max >= %s)")
                params.append(profile["min_salary"])

            remote_pref = profile.get("remote_preference", "any")
            if remote_pref and remote_pref != "any":
                where_clauses.append("(p.work_mode_signal = %s OR p.work_mode_signal = 'not_mentioned')")
                params.append(remote_pref)

            location_pref = profile.get("location_preference", "any")
            if location_pref and location_pref != "any":
                where_clauses.append("p.location ILIKE %s")
                params.append(f"%{location_pref}%")

            if profile.get("sponsorship_required"):
                where_clauses.append("p.sponsorship_signal != 'no_sponsorship_stated'")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    params.extend([vector_str, fetch_limit])

    sql = f"""
        SELECT p.id, p.title, p.company, p.location, p.description,
               p.salary_min, p.salary_max, p.url,
               p.sponsorship_signal, p.work_mode_signal, p.posted_at,
               e.chunk_text,
               1 - (e.embedding <=> %s::vector({EMBEDDING_DIM})) AS similarity
        FROM job_copilot.job_embeddings e
        JOIN job_copilot.job_postings p ON p.id = e.job_posting_id
        {where_sql}
        ORDER BY e.embedding <=> %s::vector({EMBEDDING_DIM})
        LIMIT %s;
    """

    with read_cursor() as cur:
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]

    # Rows are already best-to-worst (chunk-level) since the query orders by
    # vector distance directly. Keep the first (best) occurrence per posting.
    seen_postings = set()
    results = []
    for row in rows:
        if row["id"] in seen_postings:
            continue
        seen_postings.add(row["id"])
        results.append(row)
        if len(results) >= top_k:
            break

    return results


def rank_jobs_for_profile(top_k=20):
    """
    Ranks every posting by how well it matches the saved profile, using the
    profile's own resume_embedding as the query vector.

    This is the "show me my best matches" view — no search query needed, since
    the resume and preference text already describe what a good match looks like.
    Returns an empty list if no profile has been saved or embedded yet.
    """
    profile = get_profile()
    if not profile or profile.get("resume_embedding") is None:
        return []

    # pgvector hands the embedding back as a string like '[0.1,0.2,...]'.
    raw = profile["resume_embedding"]
    embedding = (
        [float(x) for x in raw.strip("[]").split(",")]
        if isinstance(raw, str) else list(raw)
    )

    return search_jobs_semantic(embedding, top_k=top_k, apply_profile_filters=True, profile=profile)


def browse_jobs(min_salary=None, work_mode=None, sponsorship_only=False,
                sort_by="newest", limit=50, offset=0):
    """
    Plain structured browsing — no embedding model needed, so this stays fast
    even on a cold start.

    Returns (postings, has_more). has_more is derived by fetching one extra
    row past `limit` rather than a separate COUNT(*) query, which would be a
    second full scan of whatever the filters matched just to size a "next
    page" link.

    sort_by: 'newest' (default) or 'oldest'. For match-based ranking use
             rank_jobs_for_profile() instead.
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    where_clauses = []
    params = []

    if min_salary:
        where_clauses.append("(salary_max IS NULL OR salary_max >= %s)")
        params.append(min_salary)
    if work_mode and work_mode != "any":
        where_clauses.append("work_mode_signal = %s")
        params.append(work_mode)
    if sponsorship_only:
        where_clauses.append("sponsorship_signal != 'no_sponsorship_stated'")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    order_sql = (
        "ORDER BY posted_at ASC NULLS LAST" if sort_by == "oldest"
        else "ORDER BY posted_at DESC NULLS LAST"
    )
    params.extend([limit + 1, offset])

    with read_cursor() as cur:
        cur.execute(f"""
            SELECT * FROM job_copilot.job_postings
            {where_sql}
            {order_sql}
            LIMIT %s OFFSET %s;
        """, params)
        rows = [dict(r) for r in cur.fetchall()]

    has_more = len(rows) > limit
    return rows[:limit], has_more


def get_job_posting(job_posting_id):
    """Fetches a single posting by id."""
    with read_cursor() as cur:
        cur.execute("SELECT * FROM job_copilot.job_postings WHERE id = %s;", (job_posting_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_existing_posting_ids() -> set:
    """All posting ids currently in Lakebase — used to skip already-loaded
    rows on a repeat load run rather than re-upserting unchanged data."""
    with read_cursor() as cur:
        cur.execute("SELECT id FROM job_copilot.job_postings;")
        return {r["id"] for r in cur.fetchall()}


def get_postings_without_embeddings() -> list:
    """
    Postings with no embedding rows yet.

    One known limitation: this checks whether ANY embedding exists for a
    posting, not whether every expected chunk does. A run that died halfway
    through one posting's chunks would leave it partially embedded and this
    wouldn't catch it. Acceptable given upsert_job_embeddings is idempotent —
    re-running over everything fixes it — but worth knowing.
    """
    with read_cursor() as cur:
        cur.execute("""
            SELECT p.id, p.title, p.company, p.location, p.description
            FROM job_copilot.job_postings p
            LEFT JOIN job_copilot.job_embeddings e ON e.job_posting_id = p.id
            WHERE e.job_posting_id IS NULL
            ORDER BY p.posted_at DESC NULLS LAST;
        """)
        return [dict(r) for r in cur.fetchall()]


# ----------------------------------------------------------------------------
# Applications pipeline
# ----------------------------------------------------------------------------

def save_to_pipeline(job_posting_id, status="saved"):
    """Adds a posting to the pipeline, or updates its status if already there."""
    with transaction() as cur:
        cur.execute("""
            INSERT INTO job_copilot.applications (job_posting_id, status, status_updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (job_posting_id) DO UPDATE SET
                status = EXCLUDED.status,
                status_updated_at = NOW()
            RETURNING id, status;
        """, (job_posting_id, status))
        result = cur.fetchone()
    return dict(result) if result else None


def update_application_status(job_posting_id, new_status):
    """
    Moves an application to a new pipeline stage.

    applied_at is set the first time status becomes 'applied' and never
    overwritten after — so it records when you actually applied, not the last
    time the row was touched.
    """
    with transaction() as cur:
        cur.execute("""
            UPDATE job_copilot.applications
            SET status = %s,
                status_updated_at = NOW(),
                applied_at = CASE WHEN %s = 'applied' AND applied_at IS NULL
                                  THEN NOW() ELSE applied_at END
            WHERE job_posting_id = %s
            RETURNING id, status;
        """, (new_status, new_status, job_posting_id))
        row = cur.fetchone()
    return dict(row) if row else None


def get_pipeline(status=None, stale_posting_days=30):
    """
    Returns applications joined to their postings, optionally filtered by
    stage. Each row includes posting_possibly_stale — a lightweight
    heuristic ("this listing was posted a long time ago") rather than a live
    recheck of whether the posting is still up. This app never re-fetches
    external URLs to verify a listing is still active; it's a hint to check
    manually before assuming a saved posting is still open, not a fact.
    """
    where_sql = "WHERE a.status = %s" if status else ""
    params = [stale_posting_days]
    if status:
        params.append(status)

    with read_cursor() as cur:
        cur.execute(f"""
            SELECT a.id AS application_id, a.status, a.status_updated_at,
                   a.applied_at, a.cover_letter_draft,
                   p.id AS job_posting_id, p.title, p.company, p.location,
                   p.url, p.salary_min, p.salary_max, p.posted_at,
                   p.sponsorship_signal, p.work_mode_signal,
                   (p.posted_at IS NOT NULL
                    AND p.posted_at < NOW() - (%s || ' days')::INTERVAL) AS posting_possibly_stale
            FROM job_copilot.applications a
            JOIN job_copilot.job_postings p ON p.id = a.job_posting_id
            {where_sql}
            ORDER BY a.status_updated_at DESC;
        """, params)
        return [dict(r) for r in cur.fetchall()]


def delete_application(job_posting_id):
    """
    Removes a posting from the pipeline entirely — not a status change. Its
    interview_notes cascade-delete via the FK (ON DELETE CASCADE in
    schema.sql). The underlying job_postings/job_embeddings rows are left
    alone — those are the shared catalog, not pipeline-tracking data.
    """
    with transaction() as cur:
        cur.execute(
            "DELETE FROM job_copilot.applications WHERE job_posting_id = %s RETURNING id;",
            (job_posting_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def get_stale_applications(days=14):
    """Applications sitting in a non-terminal stage with no movement recently."""
    with read_cursor() as cur:
        cur.execute("""
            SELECT a.id AS application_id, a.status, a.status_updated_at,
                   p.title, p.company, p.url
            FROM job_copilot.applications a
            JOIN job_copilot.job_postings p ON p.id = a.job_posting_id
            WHERE a.status IN ('saved', 'applied', 'interviewing')
              AND a.status_updated_at < NOW() - (%s || ' days')::INTERVAL
            ORDER BY a.status_updated_at ASC;
        """, (days,))
        return [dict(r) for r in cur.fetchall()]


def save_cover_letter(job_posting_id, draft_text):
    """Stores a generated cover letter draft against an application."""
    with transaction() as cur:
        cur.execute("""
            INSERT INTO job_copilot.applications (job_posting_id, cover_letter_draft)
            VALUES (%s, %s)
            ON CONFLICT (job_posting_id) DO UPDATE SET
                cover_letter_draft = EXCLUDED.cover_letter_draft
            RETURNING id;
        """, (job_posting_id, draft_text))
        row = cur.fetchone()
    return dict(row) if row else None


def add_interview_note(application_id, note_text, interview_date=None):
    """Adds a note against an application. One application can have several."""
    with transaction() as cur:
        cur.execute("""
            INSERT INTO job_copilot.interview_notes (application_id, note_text, interview_date)
            VALUES (%s, %s, %s)
            RETURNING id;
        """, (application_id, note_text, interview_date))
        row = cur.fetchone()
    return dict(row) if row else None


def get_interview_notes(application_id):
    """Returns all notes for an application, newest first."""
    with read_cursor() as cur:
        cur.execute("""
            SELECT * FROM job_copilot.interview_notes
            WHERE application_id = %s
            ORDER BY interview_date DESC NULLS LAST, created_at DESC;
        """, (application_id,))
        return [dict(r) for r in cur.fetchall()]


def get_pipeline_stats():
    """Counts by pipeline stage, for the dashboard."""
    with read_cursor() as cur:
        cur.execute("""
            SELECT status, COUNT(*) AS count
            FROM job_copilot.applications
            GROUP BY status;
        """)
        return {r["status"]: r["count"] for r in cur.fetchall()}


# ----------------------------------------------------------------------------
# Pipeline run auditing
# ----------------------------------------------------------------------------

class _Run:
    """Handle yielded by track_run, so a job can report how much it processed."""
    def __init__(self, run_id):
        self.id = run_id
        self.rows = None


@contextmanager
def track_run(job_name):
    """
    Records a pipeline run: start time, outcome, rows processed, and the error
    if it failed. Wrap a job's main() in this and it's audited.

        with lakebase.track_run("load_to_lakebase") as run:
            written = do_the_work()
            run.rows = written

    The exception is re-raised after recording, so failures still surface
    normally in the notebook rather than being swallowed.
    """
    with transaction() as cur:
        cur.execute(
            "INSERT INTO job_copilot.pipeline_runs (job_name) VALUES (%s) RETURNING id;",
            (job_name,),
        )
        run = _Run(cur.fetchone()["id"])

    try:
        yield run
    except Exception as e:
        with transaction() as cur:
            cur.execute("""
                UPDATE job_copilot.pipeline_runs
                SET status = 'failed', finished_at = NOW(), error_message = %s
                WHERE id = %s;
            """, (str(e)[:2000], run.id))
        raise
    else:
        with transaction() as cur:
            cur.execute("""
                UPDATE job_copilot.pipeline_runs
                SET status = 'success', finished_at = NOW(), rows_processed = %s
                WHERE id = %s;
            """, (run.rows, run.id))


def get_recent_runs(limit=20):
    """Recent pipeline runs, newest first."""
    with read_cursor() as cur:
        cur.execute("""
            SELECT * FROM job_copilot.pipeline_runs
            ORDER BY started_at DESC LIMIT %s;
        """, (limit,))
        return [dict(r) for r in cur.fetchall()]
