# POS Tagging on UD English-EWT: HMM vs BiLSTM
 
![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/managed%20with-uv-6C47FF?logo=uv&logoColor=white)
![Tests](https://img.shields.io/badge/tests-31%20passed-2ECC71)
 
Universal POS (UPOS) tagging on the Universal Dependencies English Web Treebank, built as a head-to-head between a from-scratch **Hidden Markov Model** and a **Bidirectional LSTM** tagger, under a leakage-free evaluation protocol.
 
## Results (Test Set)
 
| Model | Token Accuracy | Macro-F1 | Unknown-Word Accuracy |
| :--- | :---: | :---: | :---: |
| HMM (add-α, α = 0.1) | 87.59% | 0.794 | 30.98% |
| BiLSTM | **92%** | **0.86** | **62%** |
 
Unknown (OOV) tokens make up 9.13% of the test set, so most of the BiLSTM's gain comes from handling words the HMM has never seen.
 
## Highlights
 
- **From-scratch HMM**: first-order, add-α smoothing, explicit `<START>` / `<END>` boundary states, and numerically stable log-space Viterbi decoding with backpointers.
- **BiLSTM tagger**: neural sequence labeler trained and evaluated on the same splits as the HMM for a direct comparison.
- **Leakage-free evaluation**: two vocabularies are kept apart, the model vocabulary (frequency-pruned, with `<UNK>`) and the true training lexicon (19,674 types). Unknown-word accuracy only counts tokens unseen in the true lexicon.
- **Strict split discipline**: hyperparameters tuned on dev only, test set evaluated once.
- **Robust data ingestion**: sentence-preserving CoNLL-U parser with 10-column validation that handles multiword tokens (`1-2`) and empty nodes (`2.1`).
- **Visual analysis**: transition-probability heatmaps, Viterbi trellis with optimal-path overlay, tuning curves, and class-normalized confusion matrices.
- **Interactive demo**: tag famous David Fincher movie dialogue (*Fight Club*, *The Social Network*, *Se7en*) and watch the trellis decode it.
- **Tested**: 31 unit and integration tests covering the parser, the HMM, Viterbi, and metric invariants.
## Error Analysis
 
Most HMM errors come from a few confusions, and they are mostly lexical, not structural:
 
| Confusion | Errors | Likely cause |
| :--- | :---: | :--- |
| `NOUN → PROPN` | 161 | Capitalization and nominal ambiguity |
| `PROPN → NOUN` | 153 | Common nouns inside named-entity phrases |
| `PROPN → X` | 136 | Web and foreign tokens |
| `PROPN → PUNCT` | 118 | Symbols and abbreviations in names |
| `NOUN → X` | 112 | Slang and web-domain tokens |
 
## Dataset
 
[UD English-EWT](https://universaldependencies.org/treebanks/en_ewt/index.html), official CoNLL-U splits:
 
| Split | Sentences | Tokens |
| :--- | :---: | :---: |
| Train | 12,544 | 204,578 |
| Dev | 2,001 | 25,148 |
| Test | 2,077 | 25,094 |
 
## Repository Layout
 
```text
├── data/                  # UD English-EWT CoNLL-U splits
├── experiments/
│   └── phase1_hmm.py      # HMM tuning + test evaluation
├── notebooks/
│   ├── eda.ipynb          # Corpus statistics, tag skew, OOV, ambiguity
│   └── phase1_hmm.ipynb   # HMM walkthrough, math, interactive demo
├── references/            # Papers behind the HMM and BiLSTM work
├── reports/               # Technical report (IEEE format)
├── src/
│   ├── dataset_loader.py  # CoNLL-U loader and UPOS extractors
│   ├── dataset_registry.py# Dataset paths and integrity checks
│   ├── evaluation.py      # Accuracy, macro-F1, unknown-word metrics
│   ├── hmm_model.py       # HMM + log-space Viterbi tagger
│   └── preprocessing.py   # Vocabularies, boundaries, OOV handling
├── tests/                 # 31 unit and integration tests
├── Makefile
├── pyproject.toml
└── uv.lock
```
 
## Getting Started
 
**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/getting-started/installation/), GNU Make.
 
```bash
make setup      # install locked dependencies
make test       # run all 31 tests
uv run python experiments/phase1_hmm.py   # HMM: tune on dev, evaluate on test
make notebook   # launch JupyterLab
```
 
<!-- TODO: add the BiLSTM training/eval command and its source path here -->
 
Notebooks: `notebooks/eda.ipynb` for the data analysis, `notebooks/phase1_hmm.ipynb` for the model walkthrough and the Fincher dialogue tagger.
 
## References
 
1. Nivre et al. (2020). *Universal Dependencies v2: An Even More Useful Treebank.*
2. Silveira et al. (2014). *A Gold Standard Dependency Corpus for English.*
3. Church (1988). *A Stochastic Parts Program and Noun Phrase Parser for Unrestricted Text.*
4. Rabiner (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.*
5. Brants (2000). *TnT: A Statistical Part-of-Speech Tagger.*
6. Jurafsky & Martin (2026). *Speech and Language Processing*, Ch. 18: Sequence Labeling for POS Tagging.
7. Manning, Raghavan & Schütze (2008). *Introduction to Information Retrieval*, Ch. 8: Evaluation.
8. Sokolova & Lapalme (2009). *A Systematic Analysis of Performance Measures for Classification Tasks.*
 
