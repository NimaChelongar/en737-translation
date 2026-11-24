
Small Language Models for Translation

EN.737 · Large Language Models: Foundations and Algorithms

Can a tiny translation model give a better answer if we let it try eight times? This notebook trains a German-to-English Transformer from scratch. It then uses the same trained model to translate the same sentences with greedy decoding, beam search, and sampling with minimum Bayes risk (MBR) reranking. The experiment measures how much translation quality changes and how much extra time each method takes.

Open in Colab · View the notebook

Run the notebook

Open the notebook in Colab. For a quick CPU demo, change the settings cell to PROFILE = "smoke". For the IWSLT experiment, keep PROFILE = "project" and select a GPU under Runtime → Change runtime type.

Choose Runtime → Run all. The project profile downloads the German–English IWSLT 2017 TED-talk splits, fits a tokenizer on the training pairs, trains M2, and runs the decoding comparison. To preserve checkpoints across Colab sessions, set USE_DRIVE = True before starting.

In Translate a new sentence with M2, edit NEW_SOURCE and rerun that cell. It displays the three translations and the candidates considered by MBR.

Using team data. In the project profile, set DATA_MODE = "tsv", update the two language names, and point DATA_DIR to a folder containing train.tsv, validation.tsv, and test.tsv. Each file needs a tab-separated header with source and target. An optional id column preserves your team's example IDs.

Where this fits in the project roadmap

The team compares three systems on shared splits using one translation prompt. Nima Chelongar's contribution in this notebook is M2 and Q3(a).

Roadmap item

What the team compares or delivers

M0 · Prompting

Zero-shot and few-shot translation with pretrained GPT-2 small.

M1 · Adaptation and Q1

Fine-tune GPT-2 on parallel data. The roadmap proposes comparing LoRA with full fine-tuning, or cleaned with raw data, at an equal token budget.

M2 · From scratch

Train a tiny decoder-only Transformer and report its parameter count and token budget. Implemented here.

Q2 · Generalization

Test how translation changes across domains or across source lengths, including lengths beyond the training cap.

Q3(a) · Inference

Compare greedy, beam, and chrF-MBR as the candidate pool grows. Implemented here. The roadmap also lists caching and batching as an alternative Q3 study.

Team submission

Bring the systems together in a demo and short report with repeated comparisons, representative errors, limitations, and contributions.

M2 training budget

The default M2 has four Transformer blocks, four attention heads, width 192, a 4,096-token vocabulary, and 2,603,136 trainable parameters. It reads the source and the language labels as context. Training loss covers the English continuation and its end-of-sequence token.

With the default split seed, 19,709 of 20,000 selected training pairs survive filtering. Six epochs over those pairs budget 3,267,636 supervised target tokens and 7,214,436 nonpadding input positions. The notebook reports actual totals for the full run and for the epoch selected by the lowest validation target cross-entropy.

The decoding experiment

Decoder

How it chooses a translation

Greedy

Takes the most probable next token at each step.

Beam, width 4

Keeps up to four competing hypotheses and returns the highest-scoring retained translation after length normalization.

Sample + chrF-MBR

Draws 1, 2, 4, or 8 translations. For each candidate, it averages sentence chrF against the other draws and returns the candidate with the highest agreement. With one draw, it returns that sample directly.

Prediction recorded before scoring. Moving from one sample to eight will improve mean corpus chrF by at least 1.0 point and increase mean decoding time per source. The comparison uses one frozen checkpoint (training seed 17), the same 64 test sentences, a 64-token generation cap, and three decoding seeds (101, 202, 303).

Human references enter only after the decoder chooses its output. Higher corpus BLEU and chrF mean closer word- and character-level agreement with the references. Lower held-out cross-entropy and perplexity mean the model assigned more probability to the correct target tokens. Lower milliseconds per source means faster decoding. Timing includes generation and reranking. The spread across decoding seeds measures variability for this checkpoint and these sentences.

Find and share results

The last cell packages the run into a ZIP. Start with quality_vs_cost.png to see whether a gain in chrF is worth the extra latency.

File in the run

What to use it for

training_summary.csv in the model folder

Actual parameter count, token budgets, selected epoch, and held-out loss.

decoding_runs.csv and paired_comparisons.csv

Per-repeat scores, sacreBLEU signatures, generation and reranking time, and paired quality and latency comparisons.

decoding_predictions.jsonl and error_review.csv

Read the candidates and selected translations. Inspect errors such as copied inputs or missing numbers.

shared_splits/ and q3_test_subset.tsv

Give teammates the same examples, IDs, and references for M0 and M1.

Verification. The notebook ran end to end on CPU with synthetic data, team-format TSV input, and a reduced IWSLT run. The six-epoch figures above describe the configured training budget. Full-run BLEU, chrF, and GPU timings are produced when you run the project profile.
