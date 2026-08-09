# Databricks notebook source
"""
Embedding ingestion: Lakebase postings -> Lakebase vectors.

Third and final step of the batch pipeline. Finds postings that don't have
embeddings yet, chunks their text, embeds each chunk, and writes the vectors
into job_copilot.job_embeddings.

Why this is its own step rather than folded into the loader:
  - It's the slow part. Model inference over every chunk dominates the runtime,
    while row loading is fast. Keeping them separate means a slow embed run
    doesn't hold up getting postings into the app.
  - It's the part most likely to be re-run alone. Switching embedding models
    means regenerating every vector without touching the postings themselves —
    truncate job_embeddings, run this, done.
  - No Spark needed here at all. This is plain psycopg2 plus the model, which
    matches the constraint that Spark JDBC writes against Lakebase don't work.

Run after load_jobs_to_lakebase.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

import embeddings
import lakebase

SECRET_SCOPE = "job-copilot"


def _load_secrets_into_env():
    """
    Populates the env vars this module's functions read directly
    (LAKEBASE_CONNECTION_STRING, ADZUNA_APP_ID, ADZUNA_APP_KEY) from the same
    secret scope setup_secrets.py creates. The deployed App gets these
    automatically through app.yaml's valueFrom; a notebook session doesn't,
    so this is the notebook-side equivalent. No-ops if dbutils isn't
    available (e.g. running this file directly outside Databricks) or a
    variable is already set.
    """
    try:
        for key in ("LAKEBASE_CONNECTION_STRING", "ADZUNA_APP_ID", "ADZUNA_APP_KEY"):
            if not os.environ.get(key):
                os.environ[key] = dbutils.secrets.get(SECRET_SCOPE, key)
    except NameError:
        pass


_load_secrets_into_env()

BATCH_SIZE = 50


def embed_profile_if_needed():
    """
    Embeds the saved profile if it has text but no vector yet.

    Without this the "rank jobs for my profile" view silently returns nothing,
    since it needs a profile embedding to use as its query vector. The app
    embeds on save too, but this covers a profile written directly to the DB.
    """
    profile = lakebase.get_profile()
    if not profile:
        print("  No profile saved yet — skipping.")
        return

    if profile.get("resume_embedding") is not None:
        print("  Profile already embedded.")
        return

    text = embeddings.build_profile_text(profile)
    if not text.strip():
        print("  Profile has no text to embed — skipping.")
        return

    vector = embeddings.embed(text)
    lakebase.save_profile(profile, embedding=vector)
    print("  Profile embedded.")


def _run(run):
    print("=" * 70)
    print("EMBEDDING INGESTION")
    print("=" * 70)
    print(f"Model: {embeddings.MODEL_NAME}")
    print(f"Chunk size: {embeddings.CHUNK_SIZE}, overlap: {embeddings.CHUNK_OVERLAP}")

    lakebase.ensure_schema()

    print("\nChecking profile...")
    embed_profile_if_needed()

    print("\nFinding postings without embeddings...")
    to_embed = lakebase.get_postings_without_embeddings()
    print(f"  {len(to_embed)} postings need embeddings")

    if not to_embed:
        print("\nNothing to do.")
        return

    # Loading the model takes a few seconds, so warm it once up front rather
    # than having the first posting pay the cost mid-loop.
    print("\nLoading embedding model...")
    embeddings.get_model()
    print("  Model ready.")

    batch = []
    total_chunks = 0
    total_written = 0

    for i, posting in enumerate(to_embed, 1):
        for chunk_index, chunk, vector in embeddings.embed_posting_chunks(posting):
            batch.append((posting["id"], chunk_index, chunk, vector))
            total_chunks += 1

            if len(batch) >= BATCH_SIZE:
                total_written += lakebase.upsert_job_embeddings(batch)
                batch = []

        if i % 25 == 0:
            print(f"  Processed {i}/{len(to_embed)} postings...")

    if batch:
        total_written += lakebase.upsert_job_embeddings(batch)

    run.rows = total_written

    print("\n" + "=" * 70)
    print(f"Done. {len(to_embed)} postings -> {total_chunks} chunks -> {total_written} vectors written.")
    print("=" * 70)


def main(**kwargs):
    """Runs the job, recording start/finish/failure in job_copilot.pipeline_runs."""
    with lakebase.track_run("ingest_jobs_embeddings") as run:
        _run(run, **kwargs)


if __name__ == "__main__":
    main()