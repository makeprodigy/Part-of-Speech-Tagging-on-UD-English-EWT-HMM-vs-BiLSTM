# AML Project: Part-of-Speech Tagging on UD English-EWT

![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/managed%20with-uv-6C47FF?logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-59%20passed-2ECC71)
![Phase 1](https://img.shields.io/badge/Phase%201-HMM%20Complete-0984E3)
![Phase 2](https://img.shields.io/badge/Phase%202-BiLSTM%20%2B%20CRF%20Complete-6C5CE7)

Exploratory analysis, statistical modeling, and deep sequence labeling for Universal POS (UPOS) tagging over the Universal Dependencies English Web Treebank (UD English-EWT v2.18).

This repository implements:
1. **Comprehensive Exploratory Data Analysis (EDA)**: Corpus statistics, sentence lengths, Zipfian tag distributions, unknown-token distributions, and lexical ambiguity across 1,973 polysemous word types.
2. **Phase 1 Generative Baseline**: A mathematically rigorous first-order Hidden Markov Model (HMM) with add-$\alpha$ Laplace smoothing, explicit sentence boundary markers (`<START>`, `<END>`), numerically stable log-space Viterbi dynamic programming trellis decoding, and an evaluation protocol distinguishing the model vocabulary from the empirical training lexicon.
3. **Phase 2 Deep Sequence Modeling**:
   - Word-level Bidirectional LSTM (`WordBiLSTMTagger`) with masked sequence cross-entropy.
   - Word + Character-level Bidirectional LSTM (`CharacterBiLSTMTagger`) incorporating subword character-level BiLSTM morphology representations (Lample et al., 2016).
   - Linear-Chain Conditional Random Field (CRF) extension (`LinearChainCRF`) with exact Viterbi decoding (Huang et al., 2015; Ma & Hovy, 2016).
   - Strict empirical discipline: development-only hyperparameter tuning across 16 configurations, Macro $F_1$ checkpoint selection, and one-pass isolated test set evaluation.

---

## Project & Team Specifications

- **Project**: **Project 16**: Part-of-Speech Tagging: Hidden Markov Model vs Bidirectional LSTM Tagger
- **Team ID**: **T058** | **Section**: **A**
- **Degree Program**: **B. Tech in Computer Science & Artificial Intelligence**
- **Institution**: Newton School of Technology, Rishihood University

| # | Enrollment Number | Full Name | Institutional Email | Git Author / Handle | Primary Contributions & Core Modules |
| :-: | :---: | :--- | :--- | :--- | :--- |
| 1 | `2401010391` | **Rohan Singh** | `rohan.singh2024@nst.rishihood.edu.in` | `r0hansng` (`rohan.singh.en@gmail.com`) | **Architecture, Verification & Technical Reports**: Scaffolding (`pyproject.toml`, `Makefile`, `uv.lock`), academic literature curation (`references/phase1/`, `references/phase2/`), IEEE technical reports authoring (`reports/phase1_project_report.tex`, `reports/phase2_project_report.tex`), and 59 automated unit/integration tests across `tests/`. |
| 2 | `2401020088` | **Aniket Pathak** | `aniket.pathak2024@nst.rishihood.edu.in` | `Aniket-bit7` (`aniketpathak1073@gmail.com`) | **HMM & Neural Recurrent Modeling**: First-order HMM (`src/hmm_model.py`) with log-space Viterbi; neural sequence models in `src/bilstm_model.py` (`WordBiLSTMTagger`, `CharacterBiLSTMTagger`), bidirectional recurrent tensor graphs, and masked sequence cross-entropy loss. |
| 3 | `2401010361` | **Pushpendra Singh Parihar** | `pushpendra.parihar2024@nst.rishihood.edu.in` | `makeprodigy` (`payneparihar13@gmail.com`) | **Corpus Preprocessing & Batch Collation**: Classical and neural vocabularies (`src/preprocessing.py`, `src/neural_data.py`), special symbol contracts (`<PAD>`, `<UNK>`, `<PAD_CHAR>`), dynamic batch collation with boolean token masks, and strict dual-vocabulary integrity ($V_{\text{original}}$ vs. $V_{\text{model}}$). |
| 4 | `2401010209` | **Kapil Karan Mathur** | `kapil.mathur2024@nst.rishihood.edu.in` | `kapilkaranmathur` (`kapilkaranmathur@gmail.com`) | **Training Loops, Optimization & Demonstration**: Optimization pipelines (`src/training.py`) with Macro $F_1$ early stopping, hyperparameter grid search runners (`experiments/phase1_hmm.py`, `experiments/phase2_bilstm.py`), and interactive notebooks (`notebooks/phase1_hmm.ipynb`, `notebooks/phase2_bilstm.ipynb`) with David Fincher dialogue demos. |
| 5 | `2401010045` | **Agrim Kumar Malhotra** | `agrim.malhotra2024@nst.rishihood.edu.in` | `Agrim-2007` (`agrimmalhotra2007@gmail.com`) | **Data Ingestion, CRF & Diagnostic Diagnostics**: CoNLL-U parser (`src/dataset_loader.py`), dataset registry (`src/dataset_registry.py`), linear-chain CRF layer (`src/crf_layer.py`) with forward-backward log-sum-exp and Viterbi decoding, sequence error diagnostic suite (`src/error_analysis.py`), and EDA (`notebooks/eda.ipynb`). |

---

## Key Highlights

- **Sentence-Preserving CoNLL-U Ingestion**: Strict 10-column parser extracting `FORM` and `UPOS` while properly filtering multiword tokens (`1-2`) and empty decimal nodes (`2.1`).
- **Generative First-Order HMM**:
  - Sentence boundary context via `<START>` and `<END>` state markers (Brants, 2000).
  - Add-$\alpha$ Laplace smoothing with boundary state normalization.
  - Numerically stable log-space Viterbi dynamic programming trellis decoder (Rabiner, 1989; Jurafsky & Martin, 2026).
- **Deep Neural Sequence Models (BiLSTM)**:
  - **Word BiLSTM**: Dense word embeddings with bidirectional LSTM layers and boolean-masked sequence cross-entropy.
  - **Word + Character BiLSTM**: Subword character encoder running a bidirectional character-level LSTM per word to generate affix-sensitive subword embeddings concatenated with word vectors (Lample et al., 2016).
  - **Linear-Chain CRF**: Structured sequence transition modeling via forward score normalization and exact Viterbi decoding (Huang et al., 2015; Ma & Hovy, 2016).
- **Strict Empirical Evaluation Protocol**:
  - Maintains strict separation between model vocabulary $V_{\text{model}}$ and the empirical training lexicon $V_{\text{original}}$ ($19,674$ distinct types).
  - Out-of-vocabulary (OOV) tokens are strictly computed relative to $V_{\text{original}}$ (Sokolova & Lapalme, 2009).
  - Development-only hyperparameter tuning (16 runs across $\eta$, $p$, and $d_h$); isolated test set evaluated strictly once with frozen best-checkpoint weights.
- **Interactive Demonstrations & Diagnostics**:
  - Publication-grade figures: transition heatmaps, trellis path plots, training loss/accuracy curves, and class confusion matrices.
  - Qualitative domain robustness evaluation on iconic film dialogues (*The Social Network*, *Fight Club*, *Se7en*).

---

## Repository Layout

```text
aml-project/
├── data/                         # Official UD English EWT CoNLL-U splits (v2.18)
│   ├── train.conllu              # Train split (12,544 sents, 204,578 tokens)
│   ├── dev.conllu                # Dev split (2,001 sents, 25,148 tokens)
│   └── test.conllu               # Test split (2,077 sents, 25,094 tokens)
├── experiments/                  # Reproducible experiment runners
│   ├── phase1_hmm.py             # Phase 1 HMM tuning and test evaluation
│   └── phase2_bilstm.py          # Phase 2 BiLSTM & CRF training and benchmark runner
├── notebooks/                    # Self-explanatory Jupyter notebooks
│   ├── eda.ipynb                 # Exploratory data analysis & corpus statistics
│   ├── phase1_hmm.ipynb          # End-to-end HMM walkthrough and interactive demo
│   └── phase2_bilstm.ipynb       # Phase 2 neural comparative analysis & dialogue demo
├── references/                   # Standardized academic literature
│   ├── phase1/                   # Phase 1 papers (Church, Rabiner, Brants, Sokolova)
│   └── phase2/                   # Phase 2 papers (Hochreiter, Graves, Huang, Lample, Ma, Plank)
├── reports/                      # Academic deliverables & technical reports
│   ├── phase1_project_report.tex # IEEE Conference formatted Phase 1 project report
│   └── phase2_project_report.tex # IEEE Conference formatted Phase 2 project report
├── src/                          # Modular core package
│   ├── bilstm_model.py           # WordBiLSTMTagger and CharacterBiLSTMTagger architectures
│   ├── crf_layer.py              # LinearChainCRF with forward-backward & Viterbi decoding
│   ├── dataset_loader.py         # CoNLL-U parser, token extraction, and UPOS validation
│   ├── dataset_registry.py       # Dataset path registry and integrity checks
│   ├── error_analysis.py         # OOV error breakdown, confusion pairs, and length diagnostics
│   ├── evaluation.py             # Accuracy, Macro F1, and unknown-word metrics
│   ├── hmm_model.py              # First-order HMM with log-space Viterbi tagger
│   ├── neural_data.py            # Neural batch collation, masking, and tensor encoding
│   ├── preprocessing.py          # Vocabularies, boundaries, and OOV normalizers
│   └── training.py               # Neural training loop, validation, and early stopping
├── tests/                        # Automated unit and integration test suite (59 tests)
│   ├── test_bilstm_model.py      # Word and character BiLSTM forward pass and shape tests
│   ├── test_character_encoder.py # Character BiLSTM representation and masking tests
│   ├── test_crf_layer.py         # Linear-chain CRF loss, transition, and Viterbi tests
│   ├── test_dataset_loader.py    # Basic loader sanity checks
│   ├── test_dataset_loader_unit.py # CoNLL-U parser edge-case unit tests
│   ├── test_dataset_validation.py# CoNLL-U format & dataset integrity validation
│   ├── test_imports.py           # Dependency verification smoke test
│   ├── test_neural_data.py       # Neural vocabulary, encoding, and collation tests
│   ├── test_neural_evaluation.py # Neural metric evaluation and masking tests
│   └── test_phase1_hmm.py        # HMM transitions, emissions, Viterbi, and metric invariants
├── Makefile                      # Standardized project tasks
├── pyproject.toml                # Project packaging and dependencies (PyTorch, uv)
├── uv.lock                       # Deterministic dependency lockfile
└── memory.md                     # Persistent project state, experiments, and handoffs
```

---

## Dataset & Split Partitions

We evaluate on the official Universal Dependencies English Web Treebank (UD English-EWT v2.18) partitions:

| Split Partition | Sentences | Tokens | OOV Tokens (vs. $V_{\text{original}}$) | OOV Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Training (`train`)** | 12,544 | 204,578 | --- | --- |
| **Development (`dev`)** | 2,001 | 25,148 | 2,087 | 8.30% |
| **Test (`test`)** | 2,077 | 25,094 | 2,292 | 9.13% |

- **Empirical Training Lexicon ($V_{\text{original}}$)**: $19,674$ distinct types.
- **Model Vocabulary ($V_{\text{model}}$)**: $19,675$ entries (including `<UNK>`).
- **Universal POS Tagset**: 17 tags (`ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`, `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `SYM`, `VERB`, `X`).

---

## Phase 2 Hyperparameter Grid Tuning (Development Split)

Model selection was conducted exclusively on the development split across a Cartesian grid over learning rate $\eta \in \{0.0005, 0.0010\}$, dropout $p \in \{0.20, 0.30\}$, and hidden size $d_h \in \{128, 256\}$ ($16$ total runs, 20-epoch budget with early stopping patience of 4 non-improving epochs on Macro $F_1$):

| Model Architecture | $\eta$ | Dropout $p$ | Hidden $d_h$ | Dev Accuracy | Dev Macro $F_1$ | Best Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Character BiLSTM\*** | **0.0010** | **0.30** | **256** | **93.70%** | **0.8817** | **18** |
| Character BiLSTM | 0.0010 | 0.20 | 128 | 93.48% | 0.8779 | 18 |
| Character BiLSTM | 0.0010 | 0.30 | 128 | 93.32% | 0.8760 | 16 |
| Character BiLSTM | 0.0010 | 0.20 | 256 | 93.36% | 0.8756 | 16 |
| Character BiLSTM | 0.0005 | 0.30 | 128 | 93.17% | 0.8739 | 18 |
| Character BiLSTM | 0.0005 | 0.20 | 256 | 93.04% | 0.8712 | 20 |
| Character BiLSTM | 0.0005 | 0.30 | 256 | 92.98% | 0.8709 | 16 |
| Character BiLSTM | 0.0005 | 0.20 | 128 | 92.87% | 0.8696 | 19 |
| **Word BiLSTM\*** | **0.0010** | **0.30** | **128** | **89.81%** | **0.8447** | **19** |
| Word BiLSTM | 0.0010 | 0.30 | 256 | 89.73% | 0.8400 | 19 |
| Word BiLSTM | 0.0010 | 0.20 | 128 | 89.65% | 0.8400 | 19 |
| Word BiLSTM | 0.0010 | 0.20 | 256 | 89.24% | 0.8336 | 12 |
| Word BiLSTM | 0.0005 | 0.30 | 128 | 88.90% | 0.8281 | 20 |
| Word BiLSTM | 0.0005 | 0.20 | 128 | 88.84% | 0.8256 | 20 |
| Word BiLSTM | 0.0005 | 0.20 | 256 | 88.56% | 0.8239 | 20 |
| Word BiLSTM | 0.0005 | 0.30 | 256 | 88.38% | 0.8220 | 20 |

---

## Final Benchmark Comparison (Test Split)

The frozen best-checkpoint models selected on the development split were evaluated once on the isolated official test split ($25,094$ tokens, $2,292$ OOV tokens):

| Model Architecture | Token Accuracy | Macro $F_1$ | OOV Accuracy | Parameters | Train Time | Infer Time | Best Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 HMM Baseline** ($\alpha^* = 0.10$) | 87.59% | 0.7944 | 30.98% | 354,825 | **1.82 s** | **0.412 s** | --- |
| **Word BiLSTM + CRF** | 91.23% | 0.8497 | 47.34% | 2,878,418 | 284.15 s | 2.140 s | 19 |
| **Word + Character BiLSTM + CRF** | **93.54%** | **0.8811** | **66.67%** | 3,456,210 | 412.80 s | 3.680 s | 18 |
| **Absolute Gain (Char BiLSTM vs. HMM)** | **+5.95%** | **+0.0867** | **+35.69%** | +3.10M | --- | --- | --- |
| **Relative OOV Error Reduction** | --- | --- | **+115.2%** | --- | --- | --- | --- |

### Key Findings
1. **Subword Morphology Solves the Open-Vocabulary Problem**: Incorporating character BiLSTM embeddings boosts unknown-word (OOV) accuracy from **30.98%** (HMM) and **47.34%** (Word BiLSTM) to **66.67%** — a **+35.69% absolute improvement** (+115.2% relative gain).
2. **Dense Representation vs. Discrete Counts**: Continuous word representations overcome the independence assumptions of the generative HMM, yielding a **+3.64%** accuracy gain for Word BiLSTM and **+5.95%** overall for Character BiLSTM.
3. **Structured Sequence Transitions**: Linear-chain CRF decoding captures global tag-tag transition dynamics, preventing invalid transitions (e.g., `<START>` directly to punctuation or invalid tag sequences).

---

## Getting Started

### Prerequisites
- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- GNU Make

### 1. Environment Setup
Clone the repository and synchronize dependencies:
```bash
make setup
```

### 2. Run the Full Test Suite
Run all 59 automated unit and integration tests:
```bash
make test
```
Or directly via Python's `unittest`:
```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

### 3. Run Phase 1 (HMM Baseline)
Execute HMM training, development smoothing search ($\alpha \in \{0.01, 0.1, 1.0\}$), and test evaluation:
```bash
uv run python experiments/phase1_hmm.py
```

### 4. Run Phase 2 (BiLSTM & CRF Benchmark)
Execute the neural sequence training pipeline on the official splits:
```bash
uv run python experiments/phase2_bilstm.py
```

To run with specific hyperparameters or custom limits:
```bash
uv run python experiments/phase2_bilstm.py \
  --max-epochs 20 \
  --early-stopping-patience 4 \
  --learning-rates 0.001 \
  --dropouts 0.3 \
  --hidden-sizes 256
```

To enable the linear-chain CRF transition layer:
```bash
uv run python experiments/phase2_bilstm.py \
  --use-crf \
  --max-epochs 20 \
  --learning-rates 0.001 \
  --dropouts 0.3 \
  --hidden-sizes 256
```

For a fast deterministic smoke test directed to `/tmp`:
```bash
uv run python experiments/phase2_bilstm.py \
  --max-sentences 8 \
  --max-epochs 1 \
  --batch-size 4 \
  --output-json /tmp/phase2_bilstm_smoke_results.json \
  --predictions-json /tmp/phase2_bilstm_smoke_predictions.json \
  --report-csv /tmp/phase2_smoke_results.csv
```

### 5. Launch Interactive Notebooks
Launch JupyterLab:
```bash
make notebook
```
- Open `notebooks/eda.ipynb` for dataset distributions, Zipf's law plots, and ambiguity analysis.
- Open `notebooks/phase1_hmm.ipynb` for HMM transition heatmaps, trellis decoders, and film dialogue tagging.
- Open `notebooks/phase2_bilstm.ipynb` for Phase 2 neural comparisons, training curves, confusion matrices, and multi-model dialogue demonstrations.

---

## Theoretical References

1. **Nivre et al. (2020)**: Universal Dependencies v2: An Even More Useful Treebank (*Proc. 12th LREC*).
2. **Silveira et al. (2014)**: A Gold Standard Dependency Corpus for English (*Proc. 9th LREC*).
3. **Hochreiter & Schmidhuber (1997)**: Long Short-Term Memory (*Neural Comput.*).
4. **Graves & Schmidhuber (2005)**: Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures (*Neural Netw.*).
5. **Huang, Xu, & Yu (2015)**: Bidirectional LSTM-CRF Models for Sequence Tagging (*arXiv:1508.01991*).
6. **Lample et al. (2016)**: Neural Architectures for Named Entity Recognition (*Proc. NAACL-HLT*).
7. **Ma & Hovy (2016)**: End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF (*Proc. 54th ACL*).
8. **Plank, Søgaard, & Goldberg (2016)**: Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss (*Proc. 54th ACL*).
9. **Rabiner (1989)**: A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition (*Proc. IEEE*).
10. **Brants (2000)**: TnT — A Statistical Part-of-Speech Tagger (*Proc. 6th ANLP*).
11. **Church (1988)**: A Stochastic Parts Program and Noun Phrase Parser for Unrestricted Text.
12. **Jurafsky & Martin (2026)**: Sequence Labeling for POS Tagging (*Speech and Language Processing*, Ch. 18).
13. **Manning, Raghavan, & Schütze (2008)**: Introduction to Information Retrieval (Ch. 8: Evaluation).
14. **Sokolova & Lapalme (2009)**: A Systematic Analysis of Performance Measures for Classification Tasks (*Inf. Process. Manage.*).
