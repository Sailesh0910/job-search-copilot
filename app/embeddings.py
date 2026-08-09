"""
Embedding helpers, shared by the web routes, the MCP tools, and the batch
ingestion script.

The model loads once at import time. That's deliberate: SentenceTransformer
loads several hundred MB into memory and takes a few seconds to initialize,
so doing it per-request would make every search painfully slow. Importing this
module anywhere in the app gets the same already-loaded instance.
"""

from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazily loads the model on first use, then reuses it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> List[float]:
    """Embeds a single string into a 768-dim vector."""
    return get_model().encode(text).tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embeds many strings at once. Meaningfully faster than looping over embed()
    because the model batches them through the network in one pass rather than
    one forward pass per string.
    """
    return [vec.tolist() for vec in get_model().encode(texts)]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Sliding-window chunking, snapping the window edge back to the nearest
    sentence or newline break when one falls in the last half of the window.

    Why chunk at all: embedding models have a token limit, and a long job
    description embedded as one vector averages everything together — the
    specific requirement you care about gets diluted by boilerplate about
    company culture. Chunking keeps distinct parts of a posting separately
    searchable.

    Why overlap: a requirement split across a chunk boundary would otherwise
    be half-represented in two vectors and fully represented in neither.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            last_break = max(last_period, last_newline)
            if last_break > chunk_size * 0.5:
                end = start + last_break + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap

    return chunks


def build_posting_text(posting: Dict) -> str:
    """
    Assembles the text that represents a job posting for embedding.

    Title and company go in alongside the description because a search for
    "senior data engineer at a startup" should match on the title, not only
    on whatever the description body happens to say.
    """
    parts = [
        posting.get("title", ""),
        posting.get("company", ""),
        posting.get("location", ""),
        posting.get("description", ""),
    ]
    return "\n".join(p for p in parts if p)


def build_profile_text(profile: Dict) -> str:
    """
    Assembles the text that represents you, for matching against postings.

    This is where the soft preferences earn their place. Work authorization,
    tech stack must-haves, company size preference and free-form notes all get
    folded in here, so semantic matching naturally favours postings that
    discuss them — without needing a rigid filter column for every preference
    you might ever have.

    Labelled prefixes ("Target roles:", "Work authorization:") rather than bare
    concatenation, since the embedding picks up on the framing, not just the
    keywords.
    """
    parts = []

    if profile.get("target_roles"):
        parts.append(f"Target roles: {profile['target_roles']}")
    if profile.get("years_experience") is not None:
        parts.append(f"Years of experience: {profile['years_experience']}")
    if profile.get("tech_stack_musthaves"):
        parts.append(f"Core skills and technologies: {profile['tech_stack_musthaves']}")
    if profile.get("work_authorization"):
        parts.append(f"Work authorization: {profile['work_authorization']}")
    if profile.get("remote_preference") and profile["remote_preference"] != "any":
        parts.append(f"Work arrangement preference: {profile['remote_preference']}")
    if profile.get("location_preference") and profile["location_preference"] != "any":
        parts.append(f"Preferred location: {profile['location_preference']}")
    if profile.get("company_size_pref"):
        parts.append(f"Company size preference: {profile['company_size_pref']}")
    if profile.get("other_notes"):
        parts.append(f"Additional preferences: {profile['other_notes']}")
    if profile.get("resume_text"):
        parts.append(f"Resume:\n{profile['resume_text']}")

    return "\n".join(parts)


def embed_profile(profile: Dict) -> List[float]:
    """Builds the profile text and embeds it into a single vector."""
    return embed(build_profile_text(profile))


def embed_posting_chunks(posting: Dict):
    """
    Returns (chunk_index, chunk_text, embedding) tuples for one posting,
    ready to hand to lakebase.upsert_job_embeddings().
    """
    chunks = chunk_text(build_posting_text(posting))
    if not chunks:
        return []
    vectors = embed_batch(chunks)
    return [(i, chunk, vec) for i, (chunk, vec) in enumerate(zip(chunks, vectors))]