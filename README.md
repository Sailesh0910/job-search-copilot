# AI Job Hunting Copilot

A personal job search tool. Semantic search over live job postings, a
tracked application pipeline, and an AI agent that can search, recommend,
and draft cover letters. Built as the capstone for the Rise of the AI Data
Engineer boot camp.

See `ARCHITECTURE.md` for the full design rationale and diagrams, and
`AGENT_CONFIG.md` for the Agent Bricks system prompt and tool list.

## What it does

- Ingests real job postings from the Adzuna API, both in scheduled bulk
  batches (Spark, to Delta, then Lakebase) and live on-demand single
  searches (straight to Lakebase).
- Classifies each posting for visa sponsorship language and remote/hybrid/
  onsite work mode, heuristically, since neither is a structured field
  Adzuna exposes.
- Embeds postings and your resume/preferences for semantic matching, so you
  can search in plain language or just ask for your best matches.
- Tracks a pipeline (saved, applied, interviewing, rejected, offer) with
  interview notes and stale-application flagging.
- Drafts a tailored cover letter paragraph for a specific posting using a
  model served through the workspace's AI Gateway.
- Exposes all of the above as MCP tools, so an Agent Bricks agent can do it
  conversationally, not just through the web UI.

## Project layout

```
app/            The single deployed Databricks App (FastAPI + mounted MCP server)
notebooks/      Batch pipeline scripts. Run in Databricks, not deployed.
```

`app/` has to be self-contained since Databricks Apps deploy one folder as
one process. `notebooks/` are thin scripts that import from `../app`.

## Setup

### 1. Lakebase

Create a Lakebase project (a fresh one, not shared with other assignments,
so schemas don't collide). The app creates its own tables and indexes
automatically on first startup, so there's no manual schema step. `app/schema.sql`
is there to read directly if you want to see what gets created.

### 2. Secrets

Three secrets, same pattern as every other assignment this week:

```bash
databricks secrets create-scope job-copilot
databricks secrets put-secret job-copilot LAKEBASE_CONNECTION_STRING
databricks secrets put-secret job-copilot ADZUNA_APP_ID
databricks secrets put-secret job-copilot ADZUNA_APP_KEY
```

Get an Adzuna app id/key free at developer.adzuna.com. No credential is
needed for the cover letter feature, since it authenticates via the app's
own Databricks identity through `WorkspaceClient()`.

Add all three as resources in the Databricks Apps UI, matching the
`valueFrom` keys already in `app/app.yaml`.

### 3. Deploy

Point a Databricks App at the `app/` folder. The app name should start with
`mcp-` if you want it auto-recognized as a Custom MCP server. Otherwise
you'll need to register it manually as an external MCP.

### 4. Ingest some data

Run the three notebooks in `notebooks/`, in order, from a Databricks
notebook. They use widgets, so you'll be prompted for which roles to
search instead of editing code:

1. `ingest_jobs_spark.py`: Adzuna to Delta
2. `load_jobs_to_lakebase.py`: Delta to Lakebase
3. `ingest_jobs_embeddings.py`: embeds postings and your profile

Or skip straight to the app and use the "Fetch from Adzuna" box on the
Jobs page for a quick single-role live fetch instead.

### 5. Set up your profile

Visit `/profile` on the deployed app and fill it in. Semantic matching and
cover letter drafting are both built on this.

### 6. Register the agent

Agent Bricks, then Create Agent, then Supervisor Agent, then Add a
Databricks App (not Add a UC MCP Service), then select this app. See
`AGENT_CONFIG.md` for the system prompt to paste in.

Once it's created, note the agent's own serving endpoint name (shown on its
endpoint page in Serving) and set it as `SUPERVISOR_AGENT_ENDPOINT` in
`app/app.yaml` if it differs from the default. The `/chat` page in this app
talks to that endpoint directly, so you don't need the Agent Bricks
Playground to try the agent out, though the Playground still works too.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite is entirely self-contained. It never touches a real
Postgres database, makes a real HTTP call to Adzuna, or downloads the real
embedding model. Every external dependency (Lakebase, Adzuna, the
sentence-transformers model, the AI Gateway call the cover letter feature
makes) is faked at the seam the app itself calls through, so `pytest` runs
the same whether or not `LAKEBASE_CONNECTION_STRING`, `ADZUNA_APP_ID`, or
`ADZUNA_APP_KEY` are set. That's useful for verifying the logic before
you've provisioned anything.

## Known limitations

- **Sponsorship and work-mode signals are heuristic**, derived from posting
  text via regex, not verified employer data. The agent is instructed to
  present them as signals to confirm, not facts.
- **Single-user design.** No auth, one profile row. See `ARCHITECTURE.md`
  for what changes if this became multi-tenant.
- **Batch ingestion, not CDC.** The Spark pipeline is a scheduled/on-demand
  batch job, not a continuous change-data-capture stream. See
  `ARCHITECTURE.md` for the fuller explanation of that distinction.
- **Cover letter drafting depends on a model being available** through your
  workspace's AI Gateway under the name in `COVER_LETTER_MODEL` (defaults
  to `system.ai.llama-4-maverick`). Check AI Gateway > Models for what's
  actually available in your workspace if drafting fails.
