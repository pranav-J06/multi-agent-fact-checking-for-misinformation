"""Unit tests for Claim Extractor Agent and heuristic fallback."""
from src.agents.claim_extractor import ClaimExtractorAgent

def test_heuristic_decomposition():
    agent = ClaimExtractorAgent()
    claim = "Marie Curie was a Polish physicist and she won the Nobel Prize in Physics in 1903 as well as Chemistry in 1911."
    subclaims = agent._heuristic_decomposition(claim)
    assert len(subclaims) >= 2
    for s in subclaims:
        assert len(s.split()) >= 2

if __name__ == "__main__":
    test_heuristic_decomposition()
    print("Claim extractor tests passed!")
