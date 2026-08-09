-- ============================================================================
-- AI Job Hunting Copilot — Lakebase (Postgres) schema
--
-- Design notes:
--   * Single-user tool: no users table, no auth. Adding multi-tenancy later
--     would mean a users table plus a user_id FK on profile and applications;
--     job_postings and job_embeddings are a shared catalog and would not change.
--   * Hard filters (salary, remote, sponsorship) live in explicit columns so
--     they can be queried with plain SQL WHERE clauses.
--   * Soft preferences (tech stack, company size, free-form notes) are folded
--     into the profile's embedding instead, since semantic similarity handles
--     nuance better than exact-match columns.
--   * sponsorship_signal and work_mode_signal are heuristics derived from
--     posting text during the Spark ingestion step. Adzuna's API exposes
--     neither as a structured field, so these are best-effort classifications,
--     not verified employer statements.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS job_copilot;
SET search_path TO job_copilot;

CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
-- profile — exactly one row. Holds both hard filters and soft preference text.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_copilot.profile (
    id                    SERIAL PRIMARY KEY,
    full_name             TEXT,

    -- Hard filters: queried directly in SQL.
    target_roles          TEXT,
    years_experience      INTEGER,
    location_preference   TEXT DEFAULT 'any',   -- 'any' means do not filter
    remote_preference     TEXT DEFAULT 'any',   -- 'any' | 'remote' | 'hybrid' | 'onsite'
    min_salary            INTEGER,
    sponsorship_required  BOOLEAN DEFAULT FALSE,

    -- Soft preferences: embedded, not filtered on.
    work_authorization    TEXT,                 -- e.g. 'F1 STEM OPT, will need H1B'
    tech_stack_musthaves  TEXT,
    company_size_pref     TEXT,
    other_notes           TEXT,

    resume_text           TEXT,
    resume_embedding      vector(768),          -- all-mpnet-base-v2 = 768 dims
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- job_postings — the shared catalog, synced from Adzuna.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_copilot.job_postings (
    id                  TEXT PRIMARY KEY,       -- Adzuna's own posting id
    title               TEXT NOT NULL,
    company             TEXT,
    location            TEXT,
    description         TEXT NOT NULL,
    salary_min          INTEGER,
    salary_max          INTEGER,
    url                 TEXT,

    -- Heuristic signals derived from description text during ingestion.
    sponsorship_signal  TEXT DEFAULT 'not_mentioned'
        CHECK (sponsorship_signal IN ('mentions_sponsorship', 'no_sponsorship_stated', 'not_mentioned')),
    work_mode_signal    TEXT DEFAULT 'not_mentioned'
        CHECK (work_mode_signal IN ('remote', 'hybrid', 'onsite', 'not_mentioned')),

    posted_at           TIMESTAMPTZ,
    synced_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- job_embeddings — chunked posting text with vectors, for semantic search.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_copilot.job_embeddings (
    id              SERIAL PRIMARY KEY,
    job_posting_id  TEXT NOT NULL REFERENCES job_copilot.job_postings(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       vector(768) NOT NULL,
    model_name      TEXT NOT NULL DEFAULT 'sentence-transformers/all-mpnet-base-v2',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_posting_id, chunk_index)
);

-- ----------------------------------------------------------------------------
-- applications — one row per posting you have engaged with.
-- status_updated_at powers the "stale application" check.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_copilot.applications (
    id                  SERIAL PRIMARY KEY,
    job_posting_id      TEXT NOT NULL REFERENCES job_copilot.job_postings(id) ON DELETE CASCADE,
    status              TEXT NOT NULL DEFAULT 'saved'
        CHECK (status IN ('saved', 'applied', 'interviewing', 'rejected', 'offer')),
    status_updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    applied_at          TIMESTAMPTZ,
    cover_letter_draft  TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_posting_id)
);

-- ----------------------------------------------------------------------------
-- interview_notes — separate table since one application can have several rounds.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_copilot.interview_notes (
    id              SERIAL PRIMARY KEY,
    application_id  INTEGER NOT NULL REFERENCES job_copilot.applications(id) ON DELETE CASCADE,
    note_text       TEXT NOT NULL,
    interview_date  DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- pipeline_runs — audit log for the batch jobs. Records what ran, when, whether
-- it succeeded, how much it processed, and the error if it failed.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_copilot.pipeline_runs (
    id              SERIAL PRIMARY KEY,
    job_name        TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'success', 'failed')),
    rows_processed  INTEGER,
    error_message   TEXT
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Vector indexes: HNSW with cosine distance, matching the <=> operator used
-- in retrieval queries. HNSW is an approximate nearest-neighbour index — it
-- trades a small amount of recall for a large speed gain, which is the right
-- tradeoff here and stays correct as the catalog grows.
CREATE INDEX IF NOT EXISTS idx_job_embeddings_vector
    ON job_copilot.job_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_profile_resume_vector
    ON job_copilot.profile USING hnsw (resume_embedding vector_cosine_ops);

-- FK index: every semantic search joins embeddings back to their posting.
CREATE INDEX IF NOT EXISTS idx_job_embeddings_posting_id
    ON job_copilot.job_embeddings (job_posting_id);

-- Hard-filter columns: these are what the structured WHERE clauses hit.
CREATE INDEX IF NOT EXISTS idx_job_postings_salary_min
    ON job_copilot.job_postings (salary_min);

CREATE INDEX IF NOT EXISTS idx_job_postings_sponsorship
    ON job_copilot.job_postings (sponsorship_signal);

CREATE INDEX IF NOT EXISTS idx_job_postings_work_mode
    ON job_copilot.job_postings (work_mode_signal);

CREATE INDEX IF NOT EXISTS idx_job_postings_posted_at
    ON job_copilot.job_postings (posted_at DESC);

-- Pipeline board and stale-application queries.
CREATE INDEX IF NOT EXISTS idx_applications_status
    ON job_copilot.applications (status);

CREATE INDEX IF NOT EXISTS idx_applications_status_updated
    ON job_copilot.applications (status_updated_at);

CREATE INDEX IF NOT EXISTS idx_interview_notes_application_id
    ON job_copilot.interview_notes (application_id);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_job_started
    ON job_copilot.pipeline_runs (job_name, started_at DESC);