# PFAS entity extraction with BERT (token-classification NER)

This folder extracts 8 fields from abstracts of PFAS studies. It fine-tunes BERT-family encoders as BIO taggers and compares full fine-tuning, LoRA and two bottleneck adapters (Pfeiffer and Houlsby).

The 8 fields are `study_population, study_year, biospecimen, outcome, location, pfas, medium, source`.

## Files

| File | Purpose |
|---|---|
| `../data/human_extract_gold.xlsx` | Gold annotations for 100 abstracts, one row each (the same file used to evaluate the LLM extraction). A field with several values separates them with `\|`. |
| `data/split.csv` | Assigns each abstract (by `pmid`) to `train` or `test` (50/50) |
| `prep_data.py` | Converts the spreadsheet into word-level BIO tags and writes `data/all.jsonl` |
| `dataset.py` | Loads `data/all.jsonl` and returns the train and test documents |
| `metrics.py` | Evaluation metric: exact and relaxed F1 (see below) |
| `train_ner.py` | Trains one model (one encoder × method × seed) and scores it on the test set |
| `run_experiments.sh` | Runs the full grid (3 encoders × 4 methods × 5 seeds) and then summarises it |
| `aggregate.py` | Computes mean ± SD over seeds and writes `results/results.txt` |
| `results/results.txt` | Full results (overall and per entity) |

## Setup

```bash
python -m venv env && source env/bin/activate     # Python 3.11
pip install -r requirements.txt                     # choose the torch build that matches your CUDA
```

The three encoders download from the Hugging Face Hub the first time they are used:
`bert-base-cased`, `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` and
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`.

## Running

All commands run from the `bert_ner/` directory.

```bash
python prep_data.py              # -> data/all.jsonl
bash run_experiments.sh          # full grid (or: sbatch run_experiments.sh)

# one run:
python train_ner.py --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --name biomed_lora --method lora --lr 5e-4 --epochs 40 --seed 42

# re-summarise saved predictions:
python aggregate.py              # -> results/results.txt
```

Each run writes `runs/<name>_seed<seed>/metrics.json` and `pred_test.jsonl`.
With `--save_model` it also saves the weights (for LoRA, both the adapter and the merged model).

## Setup details

- **Split:** 50 training and 50 test abstracts, taken from `data/split.csv`, with no overlap.
- **No validation set:** hyperparameters were fixed in advance.
- **Long abstracts:** split into 512-token windows with stride 128. A word's logits are averaged over every window that contains it.
- **Methods and hyperparameters:**
  - Full fine-tuning: lr 5e-5, 20 epochs.
  - LoRA (`peft`): r = 16, alpha = 32, dropout 0.1, applied to query/key/value/dense; lr 5e-4, 40 epochs.
  - Pfeiffer and Houlsby adapters (`adapters`): reduction factor 16; lr 5e-4, 40 epochs.
  - Shared by all methods: batch size 8, linear schedule, 10% warmup, weight decay 0.01, seeds 42–46.

## Evaluation metric (`metrics.py`)

Predicted BIO spans are decoded into strings. For each document and entity, the set of predicted strings is compared with the set of gold strings, both lowercased and trimmed:

- **P** is the fraction of predicted strings that match some gold string.
- **R** is the fraction of gold strings that match some predicted string.
- **F1** is computed from P and R. If both sets are empty, F1 = 1; if exactly one is empty, F1 = 0.
- **Per-entity F1** is the mean over documents. **AVERAGE** is the plain mean over the 8 entities.

Two match functions are used:

- **exact:** the strings are identical.
- **relaxed:** the strings are identical, one contains the other, or their whitespace-token Jaccard similarity is ≥ 0.5.

## Results (test set, 50 abstracts, mean ± SD over 5 seeds)

| Encoder | Method | Exact F1 | Relaxed F1 |
|---|---|---|---|
| BiomedBERT-abstract | LoRA | **0.750** ± 0.004 | 0.843 ± 0.005 |
| BiomedBERT-abstract | Houlsby | 0.745 ± 0.009 | **0.847** ± 0.004 |
| BiomedBERT-abstract | Full FT | 0.736 ± 0.008 | 0.843 ± 0.007 |
| BiomedBERT-abstract | Pfeiffer | 0.736 ± 0.004 | 0.839 ± 0.008 |
| BiomedBERT-abstract-fulltext | LoRA | 0.748 ± 0.011 | 0.841 ± 0.004 |
| BiomedBERT-abstract-fulltext | Full FT | 0.735 ± 0.006 | 0.832 ± 0.012 |
| BiomedBERT-abstract-fulltext | Houlsby | 0.733 ± 0.003 | 0.836 ± 0.006 |
| BiomedBERT-abstract-fulltext | Pfeiffer | 0.724 ± 0.003 | 0.837 ± 0.003 |
| bert-base-cased | LoRA | 0.724 ± 0.003 | 0.812 ± 0.004 |
| bert-base-cased | Full FT | 0.720 ± 0.005 | 0.815 ± 0.005 |
| bert-base-cased | Houlsby | 0.713 ± 0.006 | 0.813 ± 0.005 |
| bert-base-cased | Pfeiffer | 0.707 ± 0.003 | 0.810 ± 0.001 |

Per-entity results are in `results/results.txt`.

Some entities (study_population, biospecimen and outcome) have gold values in only 5–7 of the 50 test abstracts. Their per-entity F1 is therefore noisy.
