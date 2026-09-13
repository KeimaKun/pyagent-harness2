import pytest
from harness.memory import MemoryStore


def test_memory_store_creation_and_insertion():
    mem = MemoryStore(db_path=":memory:", dim=4)
    row_id = mem.store_memory(
        title="Test Memory",
        content="This is test knowledge content.",
        embedding=[0.1, 0.2, 0.3, 0.4],
        metadata={"category": "test"},
    )
    assert row_id > 0


def test_memory_text_fallback_retrieval():
    mem = MemoryStore(db_path=":memory:", dim=4)
    mem.store_memory(title="Python Info", content="Python is a programming language.", metadata={})
    mem.store_memory(title="Database Info", content="SQLite is a lightweight database.", metadata={})

    # Search query
    results = mem.retrieve_relevant(query="Python", top_k=2)
    assert len(results) >= 1
    assert any("Python" in r["title"] or "Python" in r["content"] for r in results)


def test_memory_vector_retrieval_if_available():
    mem = MemoryStore(db_path=":memory:", dim=4)
    mem.store_memory(
        title="Vector Item A",
        content="Alpha content",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    mem.store_memory(
        title="Vector Item B",
        content="Beta content",
        embedding=[0.0, 1.0, 0.0, 0.0],
    )

    results = mem.retrieve_relevant(query="Alpha", query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["title"] == "Vector Item A"
