"""
Claim Extractor Agent module.
Decomposes complex input claims into 1 to 6 atomic, verifiable sub-claims.
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
2. Strict Speaker Fidelity & Zero-Inversion Rule (CRITICAL):
   - Extract EXACTLY what the input claim asserts. Do NOT attempt to fact-check, verify, or correct the claim in this stage!
   - If the input claim asserts that 'X is Y', your sub-claims MUST assert that 'X is Y' (even if you know it is factually false).
   - NEVER invert an affirmative assertion into a negative assertion (e.g. if the claim is "Brazil is the largest continent", NEVER write "Brazil is not a continent" or "The largest continent is not Brazil").
   - Only include negation words (e.g. 'not', 'never', 'neither') if the original input claim explicitly used negation!
3. Canonical Semantic Normalization & Entity Disambiguation:
   - Formulate sub-claims using clear, canonical, encyclopedic declarative language.
   - Disambiguate and expand widely recognized entity monikers, sports team nicknames, and informal shorthands to their canonical full names where unambiguous in context (e.g., 'the Lakers' -> 'the Los Angeles Lakers'; 'the Warriors' -> 'the Golden State Warriors'; 'Man United' -> 'Manchester United'; 'Barca' -> 'FC Barcelona').
   - Map informal or colloquial phrasing into its objective semantic equivalent without altering the claim's core truth-conditions.
4. Anti-Tautology Constraint:
   - NEVER generate trivial tautologies (e.g. do NOT invent 'X is a person', 'The country exists', 'X participated'). Every sub-claim must be an independently verifiable factual assertion.

Respond STRICTLY with a JSON object:
{
  "sub_claims": [
    "Canonical atomic sub-claim"
  ]
}"""

class ClaimExtractorAgent(BaseAgent):
    @staticmethod
    def has_explicit_negation(text: str) -> bool:
        """Checks if text contains explicit grammatical negation words."""
        tokens = set(re.findall(r"\b\w+(?:'\w+)?\b", text.lower()))
        neg_words = {
            "not", "never", "no", "neither", "nor", "none", "nobody", "nothing",
            "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
            "doesn't", "don't", "didn't", "won't", "wouldn't", "can't", "cannot",
            "couldn't", "shouldn't"
        }
        return bool(tokens.intersection(neg_words))

    @staticmethod
    def fix_hallucinated_inversion(sub_claim: str) -> str:
        """Restores affirmative polarity if LLM injected unwanted negation into an affirmative claim."""
        s = sub_claim
        s = re.sub(r"\b(is|was|are|were)\s+not\s+", r"\1 ", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(does|did|do)\s+not\s+(\w+)\b", lambda m: m.group(2), s, flags=re.IGNORECASE)
        s = re.sub(r"\b(has|have|had)\s+not\s+", r"\1 ", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(can|could|would|will)\s+not\s+", r"\1 ", s, flags=re.IGNORECASE)
        s = re.sub(r"\bnever\s+", "", s, flags=re.IGNORECASE)
        return s.strip()

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

        raw_sub_claims: List[str] = []
        if isinstance(data, dict):
            if "sub_claims" in data and isinstance(data["sub_claims"], list):
                raw_sub_claims = [str(s).strip() for s in data["sub_claims"] if str(s).strip()]
            elif "sub-claims" in data and isinstance(data["sub-claims"], list):
                raw_sub_claims = [str(s).strip() for s in data["sub-claims"] if str(s).strip()]
            elif "claims" in data and isinstance(data["claims"], list):
                raw_sub_claims = [str(s).strip() for s in data["claims"] if str(s).strip()]
            elif "items" in data and isinstance(data["items"], list):
                raw_sub_claims = [str(s).strip() for s in data["items"] if str(s).strip()]
        elif isinstance(data, list):
            raw_sub_claims = [str(s).strip() for s in data if str(s).strip()]

        # -------------------------------------------------------------
        # Universal Polarity & Fidelity Guard
        # -------------------------------------------------------------
        claim_has_neg = self.has_explicit_negation(claim)
        cleaned_subs: List[str] = []

        for s in raw_sub_claims:
            s_has_neg = self.has_explicit_negation(s)
            if not claim_has_neg and s_has_neg:
                # LLM pre-fact-checked and inverted the proposition: restore affirmative statement
                restored = self.fix_hallucinated_inversion(s)
                if restored and len(restored.split()) >= 3:
                    cleaned_subs.append(restored)
            else:
                cleaned_subs.append(s)

        # Deduplicate while preserving order
        seen = set()
        sub_claims: List[str] = []
        for s in cleaned_subs:
            canonical = re.sub(r"[^\w\s]", "", s.lower()).strip()
            if canonical not in seen and len(s.split()) >= 3:
                seen.add(canonical)
                sub_claims.append(s)

        # Subject coverage guarantee: Ensure primary subject of the claim is preserved
        subject_tokens = [w for w in re.findall(r"\b\w+\b", claim) if len(w) >= 4]
        has_subject_mention = any(
            any(t.lower() in s.lower() for t in subject_tokens)
            for s in sub_claims
        ) if subject_tokens else True

        if not has_subject_mention or len(sub_claims) == 0:
            heuristic_subs = self._heuristic_decomposition(claim)
            if not has_subject_mention and sub_claims:
                sub_claims = heuristic_subs + [s for s in sub_claims if s not in heuristic_subs]
            else:
                sub_claims = heuristic_subs

        # Enforce range [1, 6] as specified in system design
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
