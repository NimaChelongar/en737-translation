# en737-translation% Compile with pdfLaTeX. The document is self-contained.
\documentclass[11pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[utf8]{inputenc}
\usepackage[letterpaper,margin=0.80in,top=0.73in,bottom=0.7in,footskip=0.35in]{geometry}
\usepackage{array,tabularx,booktabs,xcolor,hyperref}
\definecolor{ink}{HTML}{172D3C}
\definecolor{accent}{HTML}{17675E}
\hypersetup{colorlinks=true,urlcolor=accent,linkcolor=accent,pdftitle={Small Language Models for Translation}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{4pt}
\setlength{\emergencystretch}{2em}
\newcommand{\subhead}[1]{\par\addvspace{9pt}{\sffamily\bfseries\large\color{accent} #1}\par\nobreak}
\makeatletter
\def\ps@readme{\def\@oddhead{}\def\@evenhead{}\def\@oddfoot{\small\sffamily\color{accent}EN.737\hfill\color{ink}\thepage}\let\@evenfoot\@oddfoot}
\makeatother
\pagestyle{readme}
\begin{document}
\color{ink}
{\sffamily\bfseries\fontsize{24}{26}\selectfont Small Language Models for Translation}\par
{\sffamily\color{accent} EN.737 \textperiodcentered{} Large Language Models: Foundations and Algorithms}\par
\textbf{Can a tiny translation model give a better answer if we let it try eight times?} This notebook trains a German-to-English Transformer from scratch. It then uses the same trained model to translate the same sentences with greedy decoding, beam search, and sampling with minimum Bayes risk (MBR) reranking. The experiment measures how much translation quality changes and how much extra time each method takes.\par
\href{https://colab.research.google.com/github/nimasmith/en737-translation/blob/main/EN737_Tiny_Transformer_MBR.ipynb}{Open in Colab} \textperiodcentered{} \href{https://github.com/nimasmith/en737-translation/blob/main/EN737_Tiny_Transformer_MBR.ipynb}{View the notebook}\par
\subhead{Run the notebook}
\begin{enumerate}\setlength{\itemsep}{2pt}\setlength{\parsep}{0pt}\setlength{\parskip}{0pt}
\item Open the notebook in Colab. For a quick CPU demo, change the settings cell to \texttt{PROFILE = "smoke"}. For the IWSLT experiment, keep \texttt{PROFILE = "project"} and select a GPU under \textbf{Runtime \ensuremath{\rightarrow} Change runtime type}.
\item Choose \textbf{Runtime \ensuremath{\rightarrow} Run all}. The project profile downloads the German--English IWSLT 2017 TED-talk splits, fits a tokenizer on the training pairs, trains M2, and runs the decoding comparison. To preserve checkpoints across Colab sessions, set \texttt{USE\_DRIVE = True} before starting.
\item In \textbf{Translate a new sentence with M2}, edit \texttt{NEW\_SOURCE} and rerun that cell. It displays the three translations and the candidates considered by MBR.
\end{enumerate}
\textbf{Using team data.} In the project profile, set \texttt{DATA\_MODE = "tsv"}, update the two language names, and point \texttt{DATA\_DIR} to a folder containing \texttt{train.tsv}, \texttt{validation.tsv}, and \texttt{test.tsv}. Each file needs a tab-separated header with \texttt{source} and \texttt{target}. An optional \texttt{id} column preserves your team's example IDs.\par
\subhead{Where this fits in the project roadmap}
The team compares three systems on shared splits using one translation prompt. \textbf{Nima Chelongar's contribution in this notebook is M2 and Q3(a).}\par
\par\smallskip\begingroup\small\renewcommand{\arraystretch}{1.20}\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.24\linewidth}@{\hspace{12pt}}>{\raggedright\arraybackslash}X@{}}\toprule
\textsf{\textbf{Roadmap item}} & \textsf{\textbf{What the team compares or delivers}} \\
\midrule
M0 \textperiodcentered{} Prompting & Zero-shot and few-shot translation with pretrained GPT-2 small. \\
M1 \textperiodcentered{} Adaptation and Q1 & Fine-tune GPT-2 on parallel data. The roadmap proposes comparing LoRA with full fine-tuning, or cleaned with raw data, at an equal token budget. \\
M2 \textperiodcentered{} From scratch & Train a tiny decoder-only Transformer and report its parameter count and token budget. Implemented here. \\
Q2 \textperiodcentered{} Generalization & Test how translation changes across domains or across source lengths, including lengths beyond the training cap. \\
Q3(a) \textperiodcentered{} Inference & Compare greedy, beam, and chrF-MBR as the candidate pool grows. Implemented here. The roadmap also lists caching and batching as an alternative Q3 study. \\
Team submission & Bring the systems together in a demo and short report with repeated comparisons, representative errors, limitations, and contributions. \\
\bottomrule\end{tabularx}\endgroup\par
\newpage
\subhead{M2 training budget}
The default M2 has four Transformer blocks, four attention heads, width 192, a 4,096-token vocabulary, and \textbf{2,603,136 trainable parameters}. It reads the source and the language labels as context. Training loss covers the English continuation and its end-of-sequence token.\par
With the default split seed, \textbf{19,709 of 20,000} selected training pairs survive filtering. Six epochs over those pairs budget \textbf{3,267,636 supervised target tokens} and \textbf{7,214,436 nonpadding input positions}. The notebook reports actual totals for the full run and for the epoch selected by the lowest validation target cross-entropy.\par
\subhead{The decoding experiment}
\par\smallskip\begingroup\small\renewcommand{\arraystretch}{1.20}\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.22\linewidth}@{\hspace{12pt}}>{\raggedright\arraybackslash}X@{}}\toprule
\textsf{\textbf{Decoder}} & \textsf{\textbf{How it chooses a translation}} \\
\midrule
Greedy & Takes the most probable next token at each step. \\
Beam, width 4 & Keeps up to four competing hypotheses and returns the highest-scoring retained translation after length normalization. \\
Sample + chrF-MBR & Draws 1, 2, 4, or 8 translations. For each candidate, it averages sentence chrF against the other draws and returns the candidate with the highest agreement. With one draw, it returns that sample directly. \\
\bottomrule\end{tabularx}\endgroup\par
\textbf{Prediction recorded before scoring.} Moving from one sample to eight will improve mean corpus chrF by at least \textbf{1.0 point} and increase mean decoding time per source. The comparison uses one frozen checkpoint (training seed 17), the same 64 test sentences, a 64-token generation cap, and three decoding seeds (101, 202, 303).\par
Human references enter only after the decoder chooses its output. Higher corpus \textbf{BLEU} and \textbf{chrF} mean closer word- and character-level agreement with the references. Lower held-out \textbf{cross-entropy} and \textbf{perplexity} mean the model assigned more probability to the correct target tokens. Lower \textbf{milliseconds per source} means faster decoding. Timing includes generation and reranking. The spread across decoding seeds measures variability for this checkpoint and these sentences.\par
\subhead{Find and share results}
The last cell packages the run into a ZIP. Start with \texttt{quality\_vs\_cost.png} to see whether a gain in chrF is worth the extra latency.\par
\par\smallskip\begingroup\small\renewcommand{\arraystretch}{1.20}\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.36\linewidth}@{\hspace{12pt}}>{\raggedright\arraybackslash}X@{}}\toprule
\textsf{\textbf{File in the run}} & \textsf{\textbf{What to use it for}} \\
\midrule
\texttt{training\_summary.csv} in the model folder & Actual parameter count, token budgets, selected epoch, and held-out loss. \\
\texttt{decoding\_runs.csv} and \texttt{paired\_comparisons.csv} & Per-repeat scores, sacreBLEU signatures, generation and reranking time, and paired quality and latency comparisons. \\
\texttt{decoding\_predictions.jsonl} and \texttt{error\_review.csv} & Read the candidates and selected translations. Inspect errors such as copied inputs or missing numbers. \\
\texttt{shared\_splits/} and \texttt{q3\_test\_subset.tsv} & Give teammates the same examples, IDs, and references for M0 and M1. \\
\bottomrule\end{tabularx}\endgroup\par
\textbf{Verification.} The notebook ran end to end on CPU with synthetic data, team-format TSV input, and a reduced IWSLT run. The six-epoch figures above describe the configured training budget. Full-run BLEU, chrF, and GPU timings are produced when you run the project profile.\par
\end{document}


