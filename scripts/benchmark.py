#!/usr/bin/env python3
"""
Prompt Repetition Entity Extraction Benchmark

Tests whether repeating the input prompt improves entity extraction accuracy
on a complex legal contract, following Leviathan et al. (Dec 2025).

Usage:
    python scripts/benchmark.py --dry-run          # cost estimate only
    python scripts/benchmark.py                    # full run with confirmation
    python scripts/benchmark.py -y --runs 3        # skip confirm, 3 runs
    python scripts/benchmark.py --models haiku-4.5 # single model
"""

import anthropic
import json
import csv
import time
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Load .env file from project root if it exists
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _val = _val.strip().strip("'\"")
                os.environ.setdefault(_key.strip(), _val)

sys.path.insert(0, str(Path(__file__).parent))
from scoring import score_answer

# ── Configuration ──────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

CONTRACT_PATH = DATA_DIR / "contract.md"
DATASET_PATH = DATA_DIR / "golden_dataset.json"

MODELS = {
    "haiku-4.5": {
        "model_id": "claude-haiku-4-5-20251001",
        "thinking": False,
        "input_cost_per_mtok": 1.00,
        "output_cost_per_mtok": 5.00,
    },
    "sonnet-4.5": {
        "model_id": "claude-sonnet-4-5-20250929",
        "thinking": False,
        "input_cost_per_mtok": 3.00,
        "output_cost_per_mtok": 15.00,
    },
    "sonnet-4.5-thinking": {
        "model_id": "claude-sonnet-4-5-20250929",
        "thinking": True,
        "thinking_budget": 10000,
        "input_cost_per_mtok": 3.00,
        "output_cost_per_mtok": 15.00,
        "runs_override": 2,  # control group — only need directional signal
    },
}

PROMPT_TEMPLATE = (
    "Given the following contract, answer this question: {question}\n\n"
    "Provide only the specific answer, no explanation.\n\n"
    "---BEGIN CONTRACT---\n"
    "{contract}\n"
    "---END CONTRACT---"
)

VARIANT_BUILDERS = {
    "baseline": lambda q: q,
    "vanilla": lambda q: f"{q}\n{q}",
    "verbose": lambda q: f"{q}\nLet me repeat that:\n{q}",
    "x3": lambda q: (
        f"{q}\nLet me repeat that:\n{q}"
        f"\nLet me repeat that one more time:\n{q}"
    ),
}

VARIANT_TOKEN_MULTIPLIERS = {
    "baseline": 1,
    "vanilla": 2,
    "verbose": 2,
    "x3": 3,
}

RUNS_PER_COMBO = 5
DELAY_BETWEEN_CALLS = 1.5
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds


# ── Data loading ───────────────────────────────────────────────────

def load_data():
    contract = CONTRACT_PATH.read_text()
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    return contract, dataset


# ── Cost estimation ────────────────────────────────────────────────

def estimate_cost(contract: str, dataset: dict, models: dict, runs: int):
    contract_tokens = int(len(contract.split()) * 1.33)
    question_overhead = 80
    base_tokens = contract_tokens + question_overhead
    n_questions = len(dataset["questions"])

    total_cost = 0.0
    details = {}

    for model_name, cfg in models.items():
        inp_tok = 0
        out_tok = 0
        n_calls = 0

        model_runs = cfg.get("runs_override", runs)
        for variant, mult in VARIANT_TOKEN_MULTIPLIERS.items():
            calls = n_questions * model_runs
            inp_tok += base_tokens * mult * calls
            out_tok += 50 * calls
            if cfg.get("thinking"):
                out_tok += 1000 * calls  # thinking tokens
            n_calls += calls

        inp_cost = inp_tok / 1e6 * cfg["input_cost_per_mtok"]
        out_cost = out_tok / 1e6 * cfg["output_cost_per_mtok"]
        model_cost = inp_cost + out_cost

        details[model_name] = {
            "n_calls": n_calls,
            "input_tokens": inp_tok,
            "output_tokens": out_tok,
            "cost": model_cost,
        }
        total_cost += model_cost

    return total_cost, details


# ── Prompt construction ────────────────────────────────────────────

def build_prompt(contract: str, question: str, variant: str) -> str:
    base = PROMPT_TEMPLATE.format(contract=contract, question=question)
    return VARIANT_BUILDERS[variant](base)


# ── API call ───────────────────────────────────────────────────────

def call_api(client, model_config: dict, prompt: str) -> dict:
    kwargs = {
        "model": model_config["model_id"],
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

    if model_config.get("thinking"):
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": model_config.get("thinking_budget", 10000),
        }
        kwargs["max_tokens"] = 16000
        kwargs["temperature"] = 1  # required for extended thinking

    start = time.time()
    response = client.messages.create(**kwargs)
    latency_ms = int((time.time() - start) * 1000)

    # Extract text (skip thinking blocks)
    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text
            break

    return {
        "raw_response": text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": latency_ms,
    }


def call_api_with_retry(client, model_config, prompt, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            return call_api(client, model_config, prompt)
        except anthropic.RateLimitError:
            wait = RETRY_BACKOFF * (attempt + 1)
            print(f" [rate limit, retry in {wait}s]", end="", flush=True)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # overloaded
                wait = RETRY_BACKOFF * (attempt + 1)
                print(f" [overloaded, retry in {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                raise
    # Final attempt without catching
    return call_api(client, model_config, prompt)


# ── Summary analysis ───────────────────────────────────────────────

def mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else 0.0


def print_summary(results: list):
    print("\n" + "=" * 70)
    print("SUMMARY ANALYSIS")
    print("=" * 70)

    valid = [r for r in results if r["score"] is not None]
    if not valid:
        print("No valid results to analyze.")
        return

    def acc(rows):
        scores = [r["score"] for r in rows if r["score"] is not None]
        return sum(scores) / len(scores) if scores else 0.0

    variants = ["baseline", "vanilla", "verbose", "x3"]
    difficulties = ["easy", "medium", "hard"]
    positions = ["early", "middle", "late"]
    model_names = sorted(set(r["model"] for r in valid))

    # ── Accuracy by variant ──
    print("\n── Accuracy by Variant ──")
    for v in variants:
        rows = [r for r in valid if r["variant"] == v]
        print(f"  {v:12s}  {acc(rows):.3f}  (n={len(rows)})")

    # ── Accuracy by variant x difficulty ──
    print("\n── Accuracy by Variant x Difficulty ──")
    print(f"  {'':12s}  " + "  ".join(f"{d:>8s}" for d in difficulties))
    for v in variants:
        cells = []
        for d in difficulties:
            rows = [r for r in valid if r["variant"] == v and r["difficulty"] == d]
            cells.append(f"{acc(rows):8.3f}" if rows else "     N/A")
        print(f"  {v:12s}  " + "  ".join(cells))

    # ── Accuracy by variant x position ──
    print("\n── Accuracy by Variant x Position ──")
    print(f"  {'':12s}  " + "  ".join(f"{p:>8s}" for p in positions))
    for v in variants:
        cells = []
        for p in positions:
            rows = [r for r in valid if r["variant"] == v and r["position"] == p]
            cells.append(f"{acc(rows):8.3f}" if rows else "     N/A")
        print(f"  {v:12s}  " + "  ".join(cells))

    # ── Accuracy by model ──
    print("\n── Accuracy by Model ──")
    for m in model_names:
        rows = [r for r in valid if r["model"] == m]
        print(f"  {m:24s}  {acc(rows):.3f}  (n={len(rows)})")

    # ── Accuracy by model x variant ──
    print("\n── Accuracy by Model x Variant ──")
    print(f"  {'':24s}  " + "  ".join(f"{v:>10s}" for v in variants))
    for m in model_names:
        cells = []
        for v in variants:
            rows = [r for r in valid if r["model"] == m and r["variant"] == v]
            cells.append(f"{acc(rows):10.3f}" if rows else "       N/A")
        print(f"  {m:24s}  " + "  ".join(cells))

    # ── Key comparison: Haiku + repetition vs Sonnet baseline ──
    print("\n── Key Comparison: Haiku + Repetition vs Sonnet Baseline ──")
    haiku_bl = [r for r in valid if r["model"] == "haiku-4.5" and r["variant"] == "baseline"]
    sonnet_bl = [r for r in valid if r["model"] == "sonnet-4" and r["variant"] == "baseline"]

    if haiku_bl:
        print(f"  Haiku baseline:            {acc(haiku_bl):.3f}")

    best_v, best_s = None, 0.0
    for v in ["vanilla", "verbose", "x3"]:
        rows = [r for r in valid if r["model"] == "haiku-4.5" and r["variant"] == v]
        s = acc(rows)
        if s > best_s:
            best_s, best_v = s, v
    if best_v:
        print(f"  Haiku best ({best_v:>8s}):    {best_s:.3f}")

    if sonnet_bl:
        print(f"  Sonnet baseline:           {acc(sonnet_bl):.3f}")

    sonnet_think = [r for r in valid if r["model"] == "sonnet-4-thinking" and r["variant"] == "baseline"]
    if sonnet_think:
        print(f"  Sonnet thinking baseline:  {acc(sonnet_think):.3f}")

    # ── Latency & tokens by variant ──
    print("\n── Mean Latency & Tokens by Variant ──")
    print(f"  {'':12s}  {'latency_ms':>10s}  {'input_tok':>10s}  {'output_tok':>10s}")
    for v in variants:
        rows = [r for r in valid if r["variant"] == v]
        lat = mean([r["latency_ms"] for r in rows])
        inp = mean([r["input_tokens"] for r in rows])
        out = mean([r["output_tokens"] for r in rows])
        print(f"  {v:12s}  {lat:10.0f}  {inp:10.0f}  {out:10.0f}")

    # ── Per-question breakdown ──
    print("\n── Per-Question Accuracy (all models/variants) ──")
    q_ids = sorted(set(r["question_id"] for r in valid),
                   key=lambda x: int(x.replace("Q", "")))
    for qid in q_ids:
        rows = [r for r in valid if r["question_id"] == qid]
        diff = rows[0]["difficulty"] if rows else "?"
        pos = rows[0]["position"] if rows else "?"
        print(f"  {qid}  ({diff:6s}, {pos:6s})  {acc(rows):.3f}  (n={len(rows)})")

    # ── Manual review count ──
    review = [r for r in results if r.get("needs_manual_review")]
    if review:
        print(f"\n  {len(review)} results flagged for manual review")


# ── Main benchmark loop ────────────────────────────────────────────

def run_benchmark(args):
    contract, dataset = load_data()
    questions = dataset["questions"]

    # Filter models
    models = MODELS
    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
        models = {k: v for k, v in MODELS.items() if k in model_list}
        if not models:
            print(f"No matching models. Available: {', '.join(MODELS.keys())}")
            return

    runs = args.runs
    variants = list(VARIANT_BUILDERS.keys())

    # Cost estimate
    total_cost, cost_details = estimate_cost(contract, dataset, models, runs)
    total_calls = sum(d["n_calls"] for d in cost_details.values())
    est_minutes = total_calls * args.delay / 60

    print("=" * 70)
    print("  PROMPT REPETITION ENTITY EXTRACTION BENCHMARK")
    print("=" * 70)
    print(f"\n  Questions:       {len(questions)}")
    print(f"  Variants:        {len(variants)} ({', '.join(variants)})")
    print(f"  Models:          {len(models)} ({', '.join(models.keys())})")
    print(f"  Runs/combo:      {runs}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Est. time:       {est_minutes:.0f} min")
    print(f"\n  Cost breakdown:")
    for name, d in cost_details.items():
        print(f"    {name:24s}  {d['n_calls']:4d} calls  "
              f"~{d['input_tokens']/1e6:.1f}M in  ${d['cost']:.2f}")
    print(f"    {'':24s}  {'─' * 30}")
    print(f"    {'TOTAL':24s}  {total_calls:4d} calls  "
          f"{'':>10s}  ${total_cost:.2f}")

    if args.dry_run:
        print("\n  [DRY RUN] Exiting before API calls.\n")
        return

    if not args.yes:
        confirm = input(f"\n  Proceed with {total_calls} calls (~${total_cost:.2f})? [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("  Aborted.")
            return

    # Setup output
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"raw_results_{timestamp}.csv"

    fieldnames = [
        "model", "variant", "question_id", "difficulty", "position",
        "run", "raw_response", "extracted_answer", "score",
        "needs_manual_review", "input_tokens", "output_tokens", "latency_ms",
    ]

    client = anthropic.Anthropic()
    results = []
    completed = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name, model_config in models.items():
            model_runs = model_config.get("runs_override", runs)
            for variant_name in variants:
                for q in questions:
                    prompt = build_prompt(contract, q["question"], variant_name)

                    for run_idx in range(1, model_runs + 1):
                        completed += 1
                        tag = (f"[{completed}/{total_calls}] "
                               f"{model_name} | {variant_name:8s} | "
                               f"{q['id']} | run {run_idx}")
                        print(f"  {tag}", end="", flush=True)

                        try:
                            api_result = call_api_with_retry(
                                client, model_config, prompt
                            )
                            scoring = score_answer(
                                q["id"], api_result["raw_response"]
                            )

                            row = {
                                "model": model_name,
                                "variant": variant_name,
                                "question_id": q["id"],
                                "difficulty": q["difficulty"],
                                "position": q["position"],
                                "run": run_idx,
                                "raw_response": api_result["raw_response"],
                                "extracted_answer": scoring["extracted_answer"],
                                "score": scoring["score"],
                                "needs_manual_review": scoring["needs_manual_review"],
                                "input_tokens": api_result["input_tokens"],
                                "output_tokens": api_result["output_tokens"],
                                "latency_ms": api_result["latency_ms"],
                            }

                            flag = " [REVIEW]" if scoring["needs_manual_review"] else ""
                            print(f" -> {scoring['score']}{flag}")

                        except Exception as e:
                            print(f" -> ERROR: {e}")
                            row = {
                                "model": model_name,
                                "variant": variant_name,
                                "question_id": q["id"],
                                "difficulty": q["difficulty"],
                                "position": q["position"],
                                "run": run_idx,
                                "raw_response": f"ERROR: {e}",
                                "extracted_answer": "",
                                "score": None,
                                "needs_manual_review": True,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "latency_ms": 0,
                            }

                        writer.writerow(row)
                        csvfile.flush()
                        results.append(row)

                        time.sleep(args.delay)

    print(f"\n  Results saved to: {csv_path}")
    print_summary(results)

    # Save summary JSON alongside CSV
    summary_path = csv_path.with_suffix(".summary.json")
    save_summary_json(results, summary_path)
    print(f"  Summary saved to: {summary_path}")


def save_summary_json(results: list, path: Path):
    """Save structured summary for downstream analysis."""
    valid = [r for r in results if r["score"] is not None]
    if not valid:
        return

    def acc(rows):
        scores = [r["score"] for r in rows if r["score"] is not None]
        return round(sum(scores) / len(scores), 4) if scores else None

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_results": len(results),
        "valid_results": len(valid),
        "manual_review_count": sum(
            1 for r in results if r.get("needs_manual_review")
        ),
        "by_variant": {},
        "by_model": {},
        "by_difficulty": {},
        "by_position": {},
        "by_model_variant": {},
        "by_question": {},
    }

    for v in ["baseline", "vanilla", "verbose", "x3"]:
        rows = [r for r in valid if r["variant"] == v]
        summary["by_variant"][v] = {
            "accuracy": acc(rows),
            "n": len(rows),
            "mean_latency_ms": round(mean([r["latency_ms"] for r in rows])),
            "mean_input_tokens": round(mean([r["input_tokens"] for r in rows])),
        }

    for m in sorted(set(r["model"] for r in valid)):
        summary["by_model"][m] = {"accuracy": acc([r for r in valid if r["model"] == m])}
        summary["by_model_variant"][m] = {}
        for v in ["baseline", "vanilla", "verbose", "x3"]:
            rows = [r for r in valid if r["model"] == m and r["variant"] == v]
            summary["by_model_variant"][m][v] = acc(rows)

    for d in ["easy", "medium", "hard"]:
        summary["by_difficulty"][d] = acc([r for r in valid if r["difficulty"] == d])

    for p in ["early", "middle", "late"]:
        summary["by_position"][p] = acc([r for r in valid if r["position"] == p])

    for qid in sorted(set(r["question_id"] for r in valid),
                       key=lambda x: int(x.replace("Q", ""))):
        rows = [r for r in valid if r["question_id"] == qid]
        summary["by_question"][qid] = {
            "accuracy": acc(rows),
            "difficulty": rows[0]["difficulty"],
            "position": rows[0]["position"],
        }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


# ── Entry point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prompt Repetition Entity Extraction Benchmark"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print cost estimate and exit",
    )
    parser.add_argument(
        "--runs", type=int, default=RUNS_PER_COMBO,
        help=f"Runs per question/variant/model combo (default: {RUNS_PER_COMBO})",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names (default: all). "
             f"Available: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--delay", type=float, default=DELAY_BETWEEN_CALLS,
        help=f"Delay between API calls in seconds (default: {DELAY_BETWEEN_CALLS})",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
