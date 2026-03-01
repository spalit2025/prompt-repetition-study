# Prompt Repetition for Entity Extraction

Claude Haiku 4.5 scores 100% on complex entity extraction from an 11,700-word legal document package without prompt repetition. This case study tests whether the "repeat your prompt" technique from [Google Research (Dec 2025)](https://arxiv.org/pdf/2512.14982) is needed for current-generation models on enterprise document extraction tasks.

The technique, introduced in ["Prompt Repetition Improves Non-Reasoning LLMs"](https://arxiv.org/pdf/2512.14982) by Leviathan, Kalman, and Matias, works because tokens in the second copy can attend to all tokens in the first during the parallelizable prefill stage, simulating bidirectional attention within a causal architecture.

## Why Entity Extraction?

Entity extraction from long documents is one of the most common enterprise RAG patterns. It directly triggers the "lost in the middle" failure mode where models struggle with information positioned in the middle of long contexts. This maps to real workflows in legal contract review, audit evidence extraction, and regulatory filing analysis. This study tests structured legal documents at ~18.5K input tokens. Results may differ at larger context scales or with unstructured text.

## Methodology

### Synthetic Test Documents

Two purpose-built legal contracts with intentional complexity traps:

- **Standard contract** (`data/contract.md`) — 4,200 words, Master Services Agreement
- **Extended contract** (`data/contract_extended.md`) — 11,700 words, consolidated document package (MSA + Data Processing Addendum + Amendment)

Complexity traps include three confusable Meridian-named entities, three different liability caps in close proximity, an amendment that modifies an earlier section, mixed date formats, and nested clause references.

### Golden Datasets

- **Standard**: 13 extraction questions (4 easy, 5 medium, 4 hard)
- **Extended**: 17 extraction questions (5 easy, 6 medium, 6 hard)

Questions are stratified by difficulty and position in the document (early / middle / late) to test for positional retrieval bias.

### Prompt Variants

| Variant | Format |
|---------|--------|
| `baseline` | `<QUERY>` |
| `vanilla` | `<QUERY>\n<QUERY>` |
| `verbose` | `<QUERY>\nLet me repeat that:\n<QUERY>` |
| `x3` | `<QUERY>\nLet me repeat that:\n<QUERY>\nLet me repeat that one more time:\n<QUERY>` |

### Models Tested

- Claude Haiku 4.5

The benchmark scripts support Sonnet 4.5 and Sonnet 4.5 with extended thinking, but only Haiku was run — it achieved 100% accuracy, making further model comparisons unnecessary for this study.

### Scoring

Automated scoring with three levels:
- **Exact (1.0)** — answer matches ground truth
- **Partial (0.5)** — directionally correct but imprecise
- **Incorrect (0.0)** — wrong answer or hallucination

Each question has a custom scorer handling format-flexible matching (e.g., `$5M` vs `$5,000,000`), with a manual review flag for ambiguous edge cases.

## Project Structure

```
prompt-repetition-study/
├── data/
│   ├── contract.md                  # Standard legal contract (4.2K words)
│   ├── contract_extended.md         # Extended document package (11.7K words)
│   ├── golden_dataset.json          # 13 extraction questions + ground truth
│   └── golden_dataset_extended.json # 17 extraction questions + ground truth
├── scripts/
│   ├── benchmark.py                 # Main benchmark runner
│   ├── benchmark_extended.py        # Extended document benchmark
│   ├── scoring.py                   # Automated scoring (13 scorers)
│   └── scoring_extended.py          # Automated scoring (17 scorers)
├── results/
│   ├── raw_results_*.csv            # Per-question, per-run results
│   ├── extended_results_*.csv       # Extended document results
│   └── *.summary.json               # Aggregated statistics
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/spalit2025/prompt-repetition-study
cd prompt-repetition-study

pip install anthropic

echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

## Usage

### Cost Estimate (Dry Run)

```bash
python scripts/benchmark.py --dry-run
```

### Run the Benchmark

```bash
# Full run with confirmation prompt
python scripts/benchmark.py

# 3 runs per combination, skip confirmation
python scripts/benchmark.py -y --runs 3

# Single model only
python scripts/benchmark.py --models haiku-4.5

# Extended document benchmark
python scripts/benchmark_extended.py -y
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--dry-run` | Cost estimate only, no API calls | — |
| `--runs N` | Runs per model/variant/question combination | 5 |
| `--models M1,M2` | Filter to specific models | all |
| `--delay S` | Seconds between API calls | 1.5 |
| `-y, --yes` | Skip confirmation prompt | — |

Results are saved to `results/` with timestamps.

## Key Results

**Extended document (11,700 words, 17 questions, Claude Haiku 4.5):**

| Difficulty | Questions | Baseline Accuracy |
|-----------|-----------|------------------|
| Easy      | 25/25     | 100%             |
| Medium    | 30/30     | 100%             |
| Hard      | 30/30     | 100%             |

Claude Haiku 4.5 achieved perfect accuracy on all extraction tasks, including three-hop cross-document amendment chains, without prompt repetition or any other optimization technique.

Full results available in `results/`.

## References

- Leviathan, Y., Kalman, M., & Matias, Y. (2025). [*Prompt Repetition Improves Non-Reasoning LLMs*](https://arxiv.org/pdf/2512.14982). Google Research.

## License

MIT
