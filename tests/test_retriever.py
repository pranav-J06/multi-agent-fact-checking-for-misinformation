"""Unit tests for FAISS Evidence Retriever."""
from src.retriever.faiss_retriever import FAISSEvidenceRetriever

def test_retriever():
    retriever = FAISSEvidenceRetriever(top_k=3)
    results = retriever.retrieve(["Albert Einstein photoelectric effect"])
    assert len(results) == 1
    item = results[0]
    assert "sub_claim" in item
    assert "top_evidence" in item
    assert len(item["top_evidence"]) <= 3
    assert item["top_score"] > 0.4
    assert any("photoelectric" in ev["text"].lower() for ev in item["top_evidence"])

if __name__ == "__main__":
    test_retriever()
    print("Retriever tests passed!")
