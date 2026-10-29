"""
Claim Extractor Agent module.
Decomposes complex input claims into 2 to 6 atomic, verifiable sub-claims.
Backbone: Llama-3.1-8B-Instruct (4-bit GGUF via Ollama)
"""

import re
from typing import List, Dict, Any, Tuple
from src.agents.base_agent import BaseAgent

SYSTEM_PROMPT = """You are an expert Claim Extractor Agent in a multi-agent fact-checking system.
Your sole job is to extract atomic, verifiable sub-claims from factual statements.

Universal Extraction & Canonical Proposition Principles:
1. Proposition Atomicity:
   - If the statement contains multiple distinct factual assertions (e.g. joined by conjunctions, dependent clauses, or compound predicates), decompose them into 2 to 4 atomic sub-claims.
   - If the statement expresses a single clear factual assertion, maintain it as EXACTLY ONE atomic sub-claim.
2. Canonical Semantic Normalization & Entity Disambiguation:
   - Formulate sub-claims using clear, canonical, encyclopedic declarative language.
   - Disambiguate and expand widely recognized entity monikers, sports team nicknames, and informal shorthands to their canonical full names where unambiguous in context (e.g., 'the Lakers' -> 'the Los Angeles Lakers'; 'the Warriors' -> 'the Golden State Warriors'; 'Man United' -> 'Manchester United'; 'Barca' -> 'FC Barcelona').
   - Map informal, colloquial, metaphorical, or domain-specific phrasing into its objective semantic equivalent (e.g., organizational presence, geographical location, role establishment, biological mechanism, scientific metric) without altering the claim's core truth-conditions.
   - Ensure the extracted proposition evaluates the real-world factual event or property, not idiosyncratic conversational word choices.
3. Anti-Tautology Constraint:
   - NEVER generate trivial tautologies (e.g. do NOT invent 'X is a person', 'The country exists', 'X participated'). Every sub-claim must be an independently verifiable factual assertion.

Respond STRICTLY with a JSON object:
{
  "sub_claims": [
    "Canonical atomic sub-claim"
  ]
}"""

class ClaimExtractorAgent(BaseAgent):
    def extract_subclaims(self, claim: str) -> Tuple[List[str], float]:
        """
        Decomposes a raw claim string into atomic sub-claims.
        
        Returns:
            sub_claims: List[str] of length 1-6
            latency: float in seconds
        """
        prompt = f"""Extract atomic verifiable sub-claims from the following claim (keep as exactly 1 claim if it is already an atomic assertion):
Claim: "{claim.strip()}"

JSON response:"""

        data, latency = self.generate_json(prompt, system_prompt=SYSTEM_PROMPT)

        sub_claims: List[str] = []
        if isinstance(data, dict):
            if "sub_claims" in data and isinstance(data["sub_claims"], list):
                sub_claims = [str(s).strip() for s in data["sub_claims"] if str(s).strip()]
            elif "sub-claims" in data and isinstance(data["sub-claims"], list):
                sub_claims = [str(s).strip() for s in data["sub-claims"] if str(s).strip()]
            elif "claims" in data and isinstance(data["claims"], list):
                sub_claims = [str(s).strip() for s in data["claims"] if str(s).strip()]
            elif "items" in data and isinstance(data["items"], list):
                sub_claims = [str(s).strip() for s in data["items"] if str(s).strip()]
        elif isinstance(data, list):
            sub_claims = [str(s).strip() for s in data if str(s).strip()]

        # Fallback only if LLM returned 0 sub-claims
        if len(sub_claims) == 0:
            sub_claims = self._heuristic_decomposition(claim)

        # Enforce range [2, 6] as specified in system design
        if len(sub_claims) > 6:
            sub_claims = sub_claims[:6]

        return sub_claims, latency

    def _heuristic_decomposition(self, claim: str) -> List[str]:
        """Splits compound sentences using clausal conjunctions without breaking geographic names or creating tautologies."""
        claim_clean = claim.strip().rstrip(".")
        # Split on coordinating clausal conjunctions or semicolons, but NOT plain commas (which breaks 'Paris, France')
        parts = re.split(r';\s+|\s+\b(?:and|but|while|as well as|additionally)\b\s+', claim_clean, flags=re.IGNORECASE)
        # Keep clauses that have at least 3 words and look like independent phrases
        valid_parts = [p.strip() for p in parts if len(p.strip().split()) >= 3]
        if len(valid_parts) >= 2:
            return [p + "." if not p.endswith(".") else p for p in valid_parts[:6]]
        # Return atomic claim as a single sub-claim without fabricating tautologies
        return [claim.strip()]
