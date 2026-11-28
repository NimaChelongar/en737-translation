# M1 (fine-tuned GPT-2), Q1(a), and Q3(b)

Dilum's part of the project: GPT-2 small fine-tuned for German → English (M1), LoRA vs. full
fine-tuning at an equal token budget (Q1a), and the KV-cache and batching benchmark (Q3b).

## Files

| Path | What it is |
|---|---|
| `EN737_GPT2_Finetuning.ipynb` | M1 and Q1(a): full fine-tuning and rank-8 LoRA, 3 seeds each, evaluation, error flags, export of the best model |
| `EN737_KV_Cache_Benchmark.ipynb` | Q3(b): latency, throughput, and memory with and without the KV cache, batch sizes 1–32 |
| `runs/m1/project-fc6cf65a02ca/` | Results of the M1 run used in the report |
| `runs/q3b/project-b2df10835539-…/` | Results of the Q3(b) run used in the report |
| `data/split_checksums.txt` | SHA-256 of the train / validation / test splits these runs used |
| `data/data_report.json` | Cleaning and filtering counts for those splits |

Model weights are **not** in this repository (≈500 MB each). The checkpoint used for Q3(b) and the
demonstration (`best_model_for_inference`, full fine-tuning, seed 42) is in the shared Drive folder.

Per-sentence CSVs here keep each sentence's `id` and the model output but **not** the German source or
English reference, so the repository does not republish the IWSLT 2017 text. Join on `id` with the
split files to get the text back. The Drive copy of each run folder has the full files.

## Headline results (test set, 512 sentences, final checkpoints, mean ± SD over 3 seeds)

| | chrF | BLEU | CE (bits/byte) | Peak train memory | Time/run |
|---|---|---|---|---|---|
| Full fine-tuning | 33.1 ± 0.3 | 11.15 ± 0.33 | 0.718 | 3,145 MiB | 14.5 min |
| LoRA r=8 | 18.4 ± 0.2 | 1.64 ± 0.19 | 0.951 | 1,603 MiB | 11 min |

- Q1(a): LoRA − full = −14.7 chrF (95% paired bootstrap CI [−15.8, −13.6]). Prediction (≥ −2.0) rejected;
  the memory prediction is supported. Learning rates were set in advance, not tuned.
- Q3(b): cache speedup at batch size 1, 64 tokens = 1.27× (predicted ≥ 1.5, rejected); cached throughput at
  batch 8 vs 1 = 8.1× (predicted ≥ 2, supported); held memory within 10% of the KV prediction in 10 of 12
  configurations, 18% at one (allocator rounding; prediction rejected as stated).

Where to find them: `runs/m1/.../summary_by_method.csv`, `paired_comparison.json`, `results_final.csv`;
`runs/q3b/.../summary_complete.csv`, `paired_speedups.csv`, `hypothesis_outcomes.json`, `cache_theory.csv`.
Report figures are the `q1_*.pdf` and `q3b_*.pdf` files in the run folders.

## Data

The splits were rebuilt by running cells 8–14 of `EN737_Tiny_Transformer_MBR.ipynb` (M2) unchanged, with
`DATA_MODE = "iwslt"` and `split_seed = 737`: 19,709 train / 507 validation / 512 test pairs. They match
M2's README count (19,709) and M0's test sentence 3281, but were **not** copied from M2's own export.
**Nima:** please run `shasum -a 256 train.tsv validation.tsv test.tsv` on your export and compare with
`data/split_checksums.txt`.

## Rerunning

Run on Colab with a GPU runtime (the report runs used a T4). Both notebooks pin `transformers==4.51.3`
(and the fine-tuning notebook `peft==0.15.2`, `accelerate==1.6.0`) and record versions in each run folder.

1. Put `train.tsv`, `validation.tsv`, `test.tsv` in `/content/team_data` (or set `DATA_DIR`).
2. Fine-tuning notebook: run all. Completed runs are saved to Drive and skipped on rerun.
   The export cell prints the checkpoint path for the benchmark.
3. KV notebook: set `CHECKPOINT` in the config cell to that path (the committed value points to Dilum's
   Drive layout), then run all.

To reuse the M1 recipe for another question (for example cleaned data for Q1(b)), point `DATA_DIR` at the
new splits and set `METHODS = ["full"]`; keep `CFG` and `SEEDS` unchanged so the runs compare with these.
The experiment ID changes automatically with the data, so new results go to a new folder.
