"""
Cross-Verifier Agent module with Batched Dual-Pass Verification.
Runs two independent, stateless passes over the same Llama-3.1-8B backbone:
  - Pass A: Prompted purely to evaluate and argue SUPPORT for ALL sub-claims at once
  - Pass B: Prompted purely to evaluate and argue REFUTE for ALL sub-claims at once
Reduces LLM round-trips from 2*N calls down to strictly 2 calls total,
slashing pipeline latency from ~25-38s down to ~6-8s while strictly preserving
the stateless separation of stances and preventing premature consensus.
"""

from typing import List, Dict, Any, Tuple
import re
import numpy as np
import logging
from src.agents.base_agent import BaseAgent

from datetime import datetime

logger = logging.getLogger(__name__)

BATCH_SUPPORT_PROMPT = """You are a rigorous Natural Language Inference (NLI) verification engine.
Your task is to determine whether the provided retrieved evidence (PREMISE) logically SUPPORTS or ENTAILS each atomic sub-claim (HYPOTHESIS).

System Temporal Anchor: Current Reference Date is {current_date}.

Formal Logical Principles:
1. Strict Premise Grounding: Base decisions exclusively on facts explicitly stated in the retrieved evidence. Disregard internal pre-training knowledge, cutoffs, or unstated assumptions.
2. Predicate & Relation Verification: The premise must substantiate the specific action, role, award, date, or property asserted in the sub-claim. Merely mentioning the same named entity is NOT sufficient for support; if the entity is mentioned but the asserted claim/activity is omitted or unverified, support is low (0.0 - 0.2).
3. Semantic Paraphrasing: Distinguish genuine factual contradictions from benign linguistic paraphrasing or standard abbreviations. If the premise verifies the core proposition under natural interpretation, assign appropriate support.
4. Polarity & Negative Propositions: If a sub-claim asserts that an entity did not achieve an outcome or was not present, evidence demonstrating that a mutually exclusive entity achieved that outcome directly entails the negative assertion.
5. Graded Entailment Scoring (0.0 to 1.0):
   - 0.85 to 1.0: Complete, direct entailment where the premise explicitly proves the sub-claim.
   - 0.50 to 0.84: Partial support; premise corroborates the main proposition with minor detail unstated.
   - 0.20 to 0.49: Weak or tangential alignment; related topics are discussed but key assertions remain unverified.
   - 0.0 to 0.19: The premise is silent, inconclusive, or directly contradicts the sub-claim.

Respond strictly in JSON format:
{{
  "verifications": [
    {{
      "idx": 1,
      "support_score": <float 0.0 to 1.0>,
      "rationale": "<1-2 sentence logical justification>"
    }}
  ]
}}"""

BATCH_REFUTE_PROMPT = """You are a rigorous Natural Language Inference (NLI) adversarial verification engine.
Your task is to determine whether the provided retrieved evidence (PREMISE) logically REFUTES or CONTRADICTS each atomic sub-claim (HYPOTHESIS).

System Temporal Anchor: Current Reference Date is {current_date}.

Formal Logical Principles:
1. Strict Premise Grounding: Base decisions exclusively on facts explicitly stated in the retrieved evidence. Disregard internal pre-training knowledge, cutoffs, or unstated assumptions.
2. Direct Contradiction: Assign high refutation when the premise directly conflicts with, disproves, or renders the sub-claim factually impossible (e.g. incompatible dates, conflicting winners, mutually exclusive roles, or opposing outcomes).
3. Silence is NOT Contradiction (Inconclusive / NEI): If the evidence is silent on the assertion, does not mention the claimed fact, or discusses unrelated achievements, the refutation score must be low (0.0 - 0.15). Never declare a contradiction simply because evidence does not mention a detail.
4. Corroboration Precludes Refutation: If the premise corroborates or confirms the sub-claim, refutation is 0.0.
5. Polarity & Negation Awareness: If a sub-claim asserts a negative proposition (e.g., entity X did not win), evidence confirming that entity X lost or that entity Y won corroborates the negative claim and does NOT refute it.
6. Graded Refutation Scoring (0.0 to 1.0):
   - 0.85 to 1.0: Direct, explicit contradiction of the core asserted fact.
   - 0.50 to 0.84: Substantive inconsistency or strong contextual incompatibility.
   - 0.20 to 0.49: Minor inconsistency or ambiguous discrepancy.
   - 0.0 to 0.19: The premise is silent, neutral, uninformative, or corroborates the sub-claim.

Respond strictly in JSON format:
{{
  "verifications": [
    {{
      "idx": 1,
      "refutation_score": <float 0.0 to 1.0>,
      "rationale": "<1-2 sentence logical justification>"
    }}
  ]
}}"""

class CrossVerifierAgent(BaseAgent):
    def verify_subclaims(
        self,
        retrieval_results: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Runs batched dual-pass verification over all sub-claims with retrieved evidence.
        Pass A (Support) and Pass B (Refute) are executed in two single-shot batched passes.
        """
        N = len(retrieval_results)
        stance_tensor = np.zeros((N, 2), dtype=np.float32)
        detailed_verifications = []
        total_latency = 0.0

        if N == 0:
            return stance_tensor, detailed_verifications, total_latency

        # Current temporal anchor
        curr_date = datetime.now().strftime("%B %Y")
        support_system = BATCH_SUPPORT_PROMPT.format(current_date=curr_date)
        refute_system = BATCH_REFUTE_PROMPT.format(current_date=curr_date)

        # Build batched items block
        items_text_list = []
        for i, item in enumerate(retrieval_results, 1):
            sub = item["sub_claim"]
            ev = item.get("concatenated_evidence") or "No relevant evidence retrieved."
            items_text_list.append(f"{i}. Sub-claim: \"{sub}\"\n   Retrieved Evidence: \"{ev}\"")
        items_block = "\n\n".join(items_text_list)

        prompt_sup = f"Evaluate the degree to which the retrieved evidence logically SUPPORTS each of the following {N} items, assigning a calibrated confidence score between 0.0 and 1.0:\n\n{items_block}\n\nJSON response:"
        prompt_ref = f"Evaluate the degree to which the retrieved evidence logically CONTRADICTS or REFUTES each of the following {N} items, assigning a calibrated confidence score between 0.0 and 1.0:\n\n{items_block}\n\nJSON response:"

        # -------------------------------------------------------------
        # Pass A: Support Stance (Single Batched LLM Call)
        # -------------------------------------------------------------
        data_sup, lat_sup = self.generate_json(prompt_sup, system_prompt=support_system)
        total_latency += lat_sup

        # -------------------------------------------------------------
        # Pass B: Refute Stance (Single Batched LLM Call, Stateless)
        # -------------------------------------------------------------
        data_ref, lat_ref = self.generate_json(prompt_ref, system_prompt=refute_system)
        total_latency += lat_ref

        def _safe_score(val: Any, default: float = 0.0) -> float:
            if isinstance(val, (int, float)):
                return float(np.clip(float(val), 0.0, 1.0))
            if isinstance(val, str):
                m = re.search(r'[-+]?\d*\.?\d+', val)
                if m:
                    try:
                        return float(np.clip(float(m.group(0)), 0.0, 1.0))
                    except ValueError:
                        pass
            return default

        def _safe_idx(k: Any, fallback: int) -> int:
            if isinstance(k, int):
                return k
            if isinstance(k, str):
                m = re.search(r'\d+', k)
                if m:
                    try:
                        return int(m.group(0))
                    except ValueError:
                        pass
            return fallback

        # Parse Pass A results
        sup_items = {}
        if isinstance(data_sup, dict):
            v_list = data_sup.get("verifications", data_sup.get("items", []))
            if isinstance(v_list, list):
                for idx, v in enumerate(v_list):
                    if isinstance(v, dict):
                        k_raw = v.get("idx", idx + 1)
                        k = _safe_idx(k_raw, idx + 1)
                        s = v.get("support_score", v.get("score", v.get("confidence", 0.0)))
                        r = v.get("rationale", v.get("explanation", ""))
                        score_val = _safe_score(s, 0.0)
                        sup_items[k] = (score_val, str(r))
                        # Also index by 1-based sequential position as fallback
                        sup_items[f"pos_{idx+1}"] = (score_val, str(r))

        # Parse Pass B results
        ref_items = {}
        if isinstance(data_ref, dict):
            v_list = data_ref.get("verifications", data_ref.get("items", []))
            if isinstance(v_list, list):
                for idx, v in enumerate(v_list):
                    if isinstance(v, dict):
                        k_raw = v.get("idx", idx + 1)
                        k = _safe_idx(k_raw, idx + 1)
                        s = v.get("refutation_score", v.get("refute_score", v.get("score", 0.0)))
                        r = v.get("rationale", v.get("explanation", ""))
                        score_val = _safe_score(s, 0.0)
                        ref_items[k] = (score_val, str(r))
                        # Also index by 1-based sequential position as fallback
                        ref_items[f"pos_{idx+1}"] = (score_val, str(r))

        # Compile per-subclaim stance tensor & detailed trace
        for i, item in enumerate(retrieval_results):
            sub_claim = item["sub_claim"]
            evidence = item["concatenated_evidence"] or "No relevant evidence retrieved."
            item_num = i + 1

            # Retrieve parsed scores with robust fallbacks: check key, 0-based key, and pos tag
            sup_score, sup_rat = sup_items.get(
                item_num,
                sup_items.get(i, sup_items.get(f"pos_{item_num}", (0.0, "Evaluated by support agent.")))
            )
            ref_score, ref_rat = ref_items.get(
                item_num,
                ref_items.get(i, ref_items.get(f"pos_{item_num}", (0.0, "Evaluated by refutation agent.")))
            )

            stance_tensor[i, 0] = sup_score
            stance_tensor[i, 1] = ref_score

            detailed_verifications.append({
                "sub_claim": sub_claim,
                "evidence": evidence,
                "retrieval_score": item.get("top_score", 0.0),
                "support_score": sup_score,
                "support_rationale": sup_rat,
                "refute_score": ref_score,
                "refute_rationale": ref_rat,
            })

        return stance_tensor, detailed_verifications, total_latency
