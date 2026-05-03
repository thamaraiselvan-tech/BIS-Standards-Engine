from __future__ import annotations
"""
inference.py — BIS Standards Recommendation Engine
Mandatory judge entry point.

Usage:
    python inference.py --input hidden_private_dataset.json --output team_results.json

Output JSON schema (strict — must match eval_script.py expectations):
[
  {
    "id": "...",
    "query": "...",
    "retrieved_standards": ["IS 269: 1989", ...],
    "latency_seconds": 0.003
  }
]
"""

import argparse
import json
import os
import sys
import time

# ── Add src to Python path ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from rag_pipeline import get_index, query_pipeline

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.pdf')


def run_inference(input_path: str, output_path: str) -> None:
    # ── Load input ────────────────────────────────────────────────────────────
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
    except Exception as e:
        print(f"[ERROR] Cannot read input file '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # ── Warm up index ONCE (build before timing individual queries) ───────────
    pdf = PDF_PATH if os.path.exists(PDF_PATH) else None
    print(f"[inference] Building BIS index{' with PDF enrichment' if pdf else ''}...")
    get_index(pdf_path=pdf)
    print(f"[inference] Index ready. Processing {len(queries)} queries...\n")

    # ── Run each query — latency measured per-query only (index already built) ─
    results = []
    for item in queries:
        qid   = item.get('id', '')
        query = item.get('query', '')

        t0 = time.perf_counter()
        out = query_pipeline(query, top_k=5, pdf_path=None)  # index already warm
        latency = round(time.perf_counter() - t0, 4)

        # Strict output schema required by judges
        result = {
            'id':                  qid,
            'query':               query,
            'retrieved_standards': out['retrieved_standards'],
            'latency_seconds':     latency,
        }

        # Convenience: pass through expected_standards if present (public test set)
        if 'expected_standards' in item:
            result['expected_standards'] = item['expected_standards']

        results.append(result)
        top3_display = out['retrieved_standards'][:3]
        print(f"  [{qid}] {top3_display}  ({latency:.4f}s)")

    # ── Write output ──────────────────────────────────────────────────────────
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[inference] Done — {len(results)} results saved to '{output_path}'")
    except Exception as e:
        print(f"[ERROR] Cannot write output file '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BIS Standards Recommendation Engine — Inference Script'
    )
    parser.add_argument('--input',  required=True, help='Path to input JSON file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()
    run_inference(args.input, args.output)
