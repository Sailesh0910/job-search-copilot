"""
The real SentenceTransformer model is never loaded here — it's a ~420MB
download and several seconds of init, exactly the cost main.py's lifespan
pays once at startup so real requests don't have to. get_model() is
monkeypatched to a deterministic fake so these tests run instantly and need
no network access.
"""

import pytest

import embeddings


class _FakeVector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class FakeSentenceTransformer:
    """Deterministic stand-in: each text maps to a fixed-length vector
    derived from its length, so different inputs produce different (but
    reproducible) vectors without real inference."""

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts):
        single = isinstance(texts, str)
        inputs = [texts] if single else texts
        vectors = [_FakeVector([float(len(t))] * 4) for t in inputs]
        return vectors[0] if single else vectors


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(embeddings, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(embeddings, "_model", None)
    yield
    monkeypatch.setattr(embeddings, "_model", None)


def test_get_model_is_lazy_and_cached():
    m1 = embeddings.get_model()
    m2 = embeddings.get_model()
    assert m1 is m2


def test_embed_returns_list_of_floats():
    vec = embeddings.embed("hello")
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)


def test_embed_batch_returns_one_vector_per_text():
    vecs = embeddings.embed_batch(["a", "bb", "ccc"])
    assert len(vecs) == 3


# ----------------------------------------------------------------------------
# chunk_text — pure logic, no model involved
# ----------------------------------------------------------------------------

def test_chunk_text_empty_returns_empty_list():
    assert embeddings.chunk_text("") == []
    assert embeddings.chunk_text(None) == []


def test_chunk_text_short_text_is_a_single_chunk():
    text = "short description"
    assert embeddings.chunk_text(text) == [text]


def test_chunk_text_long_text_splits_into_multiple_nonempty_chunks():
    text = "Sentence one. " * 200
    chunks = embeddings.chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_text_overlap_repeats_boundary_content():
    text = "AAAA. " * 100
    chunks = embeddings.chunk_text(text, chunk_size=50, overlap=10)
    # With overlap > 0, consecutive chunks should share some trailing/leading
    # content rather than cutting cleanly at chunk_size with no repetition.
    assert len(chunks) > 1


# ----------------------------------------------------------------------------
# Text assembly
# ----------------------------------------------------------------------------

def test_build_posting_text_joins_nonempty_fields_only():
    posting = {"title": "DE", "company": "Acme", "location": "", "description": "desc"}
    assert embeddings.build_posting_text(posting) == "DE\nAcme\ndesc"


def test_build_profile_text_includes_labelled_fields():
    profile = {"target_roles": "Data Engineer", "years_experience": 5, "remote_preference": "any"}
    text = embeddings.build_profile_text(profile)
    assert "Target roles: Data Engineer" in text
    assert "Years of experience: 5" in text
    assert "remote" not in text.lower()  # 'any' is deliberately excluded


def test_build_profile_text_empty_profile_is_empty_string():
    assert embeddings.build_profile_text({}) == ""


# ----------------------------------------------------------------------------
# embed_posting_chunks — ties chunking + embedding together
# ----------------------------------------------------------------------------

def test_embed_posting_chunks_returns_index_text_vector_tuples():
    posting = {"title": "DE", "description": "Some description text.", "location": "", "company": "Acme"}
    result = embeddings.embed_posting_chunks(posting)
    assert len(result) == 1
    idx, chunk, vec = result[0]
    assert idx == 0
    assert isinstance(chunk, str)
    assert isinstance(vec, list)


def test_embed_posting_chunks_empty_posting_returns_empty_list():
    assert embeddings.embed_posting_chunks({}) == []
