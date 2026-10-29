"""
Command-Line Interface (CLI) for Multi-Agent Fact-Checking.
Usage:
  python -m src.cli --claim "Marie Curie won two Nobel Prizes."
  python -m src.cli --benchmark
  python -m src.cli --interactive
"""

import sys
import argparse
import json
from src.pipeline import FactCheckingPipeline
from src.evaluation.benchmark import run_benchmarks

def print_result_card(res: dict):
    verdict = res["verdict"]
    conf = res["confidence"] * 100
    lat = res["latencies"]

    color = "\033[92m" if verdict == "SUPPORTED" else ("\033[91m" if verdict == "REFUTED" else "\033[93m")
    reset = "\033[0m"
    bold = "\033[1m"

    print("\n" + "=" * 65)
    print(f"{bold}FACT-CHECKING MULTI-AGENT VERIFICATION REPORT{reset}")
    print("=" * 65)
    print(f"Claim:        {res['claim']}")
    print(f"Verdict:      {color}{bold}{verdict}{reset} ({conf:.1f}% confidence)")
    print(f"Explanation:  {res['explanation']}")
    print("-" * 65)
    print(f"{bold}Stage Latencies:{reset}")
    print(f"  - Claim Extractor:       {lat.get('claim_extractor_s', 0):.2f}s")
    print(f"  - FAISS Retriever:       {lat.get('evidence_retriever_s', 0):.3f}s")
    print(f"  - Cross-Verifier (2x):   {lat.get('cross_verifier_s', 0):.2f}s")
    print(f"  - Verdict Aggregator:    {bold}{lat.get('verdict_aggregator_ms', 0):.2f}ms{reset} (<50ms budget)")
    print(f"  - Total Pipeline Time:   {lat.get('total_pipeline_s', 0):.2f}s")
    print("-" * 65)
    print(f"{bold}Atomic Sub-claims & Dual-Pass Rationales:{reset}")
    for idx, c in enumerate(res["evidence_citations"], 1):
        print(f"  [{idx}] Sub-claim: {c['sub_claim']}")
        print(f"      Support Score: {c['support_score']:.2f} | Rationale: {c['support_rationale']}")
        print(f"      Refute Score:  {c['refute_score']:.2f} | Rationale: {c['refute_rationale']}")
        if c.get("citations"):
            print(f"      Cited: {c['citations'][0]}")
    print("=" * 65 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Fact-Checking System CLI")
    parser.add_argument("--claim", type=str, help="Claim string to verify")
    parser.add_argument("--benchmark", action="store_true", help="Run FEVER & LIAR benchmark evaluation")
    parser.add_argument("--interactive", action="store_true", help="Start interactive claim checking shell")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmarks(num_fever_samples=6, num_liar_samples=3)
        return

    pipeline = FactCheckingPipeline()

    if args.claim:
        res = pipeline.verify_claim(args.claim)
        print_result_card(res)
    elif args.interactive or len(sys.argv) == 1:
        print("\n=== Multi-Agent Fact-Checking Interactive Shell ===")
        print("Type any factual claim and press Enter. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                claim = input("\nEnter Claim > ").strip()
                if not claim:
                    continue
                if claim.lower() in ["exit", "quit"]:
                    break
                res = pipeline.verify_claim(claim)
                print_result_card(res)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
