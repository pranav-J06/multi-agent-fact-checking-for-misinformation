"""
Quick-start Demonstration Script for VeritasAI Multi-Agent Fact-Checking.
Executes verification across 3 representative claims (SUPPORTED, REFUTED, NOT ENOUGH INFO).
Demonstrates the sub-50ms Verdict Aggregator head and dual-pass cross-verification.
"""

from src.pipeline import FactCheckingPipeline
from src.cli import print_result_card

DEMO_CLAIMS = [
    ("SUPPORTED", "Marie Curie was the first woman to win a Nobel Prize and also won prizes in two distinct scientific fields."),
    ("REFUTED", "Marie Curie won the Nobel Prize in Literature in 1911."),
    ("NOT ENOUGH INFO", "Marie Curie was an accomplished violinist who played classical sonatas every Sunday afternoon.")
]

def main():
    print("=" * 70)
    print("VERITAS-AI: MULTI-AGENT FACT-CHECKING SYSTEM DEMONSTRATION")
    print("Paper: Multi-Agent LLM Frameworks for Fact-Checking (DA1)")
    print("Authors: Pranav Jahagirdar [24BCE1503] & Umang Raj [24BCE1470]")
    print("Hardware: RTX 4060 (8GB VRAM) • Backbone: Llama-3.1-8B-Instruct (4-bit GGUF)")
    print("=" * 70)

    pipeline = FactCheckingPipeline()

    for expected_label, claim in DEMO_CLAIMS:
        print(f"\n>>> TESTING EXPECTED '{expected_label}' CLAIM:")
        print(f"Claim: \"{claim}\"")
        res = pipeline.verify_claim(claim)
        print_result_card(res)

    print("\nDemonstration complete! To explore the interactive Web UI, visit http://127.0.0.1:8000")

if __name__ == "__main__":
    main()
