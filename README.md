
[PFAS Geo-map Webpage](https://fair-ehs.med.yale.edu/pfas_map_slider.html)

[PFAS MedViz Webpage](https://medviz.org/app/?dataset=medviz/pfas-2025-pfas-chemical)

---

# PFAS LLM Pipeline

Reproducibility artifacts for the LLM-based filtering and entity extraction
of PFAS water-contamination abstracts.

## Layout

```
pfas_llm/
├── data/
│   ├── keyword_filtered.csv          7,726 abstracts after keyword filtering (input)
│   ├── llm_filter_predictions.csv    raw LLM yes/no prediction for each abstract
│   ├── llm_filter_kept.csv           1,425 abstracts kept after LLM filter
│   ├── llm_extracted.csv             Entities extracted from the 1,425 abstracts
│   ├── human_filter_gold.xlsx        200 human annotated abstracts for evaluating LLM filtering 
│   └── human_extract_gold.xlsx       100 human annotated abstracts for evaluating LLM extraction
├── prompt/
│   ├── filter_prompt.txt
│   └── extract_prompt.txt
├── llm/
│   ├── run_filter.py                 LLM filtering (uses Azure OpenAI)
│   ├── run_extract.py                LLM entity extraction (uses Azure OpenAI)
│   ├── eval_filter.py                evaluate LLM filtering against human annotation
│   └── eval_extract.py               evaluate LLM extraction against human annotation
└── logging_utils.py
```

## Setup

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=<your_key>
pip install pandas openai python-dotenv tqdm scikit-learn openpyxl
```

The `run_*.py` scripts call Azure OpenAI. The endpoint and deployment name
are hardcoded inside each script; adjust them to match your own Azure
deployment if you want to re-run inference.

## Pipeline

All scripts assume you are running from the `pfas_llm/` directory.

```bash
# 1. LLM filtering on the 7,726 keyword-filtered abstracts.
#    Output: data/llm_filter_predictions.csv, data/llm_filter_kept.csv
python llm/run_filter.py

# 2. LLM entity extraction on the 1,425 kept abstracts.
#    Output: data/llm_extracted.csv
python llm/run_extract.py

# 3. Evaluate LLM filtering against the 200 human annotated abstracts.
python llm/eval_filter.py

# 4. Evaluate LLM extraction against the 100 human annotated abstracts.
python llm/eval_extract.py
```

The `eval_*.py` scripts read pre-computed predictions only and never call
the LLM. They reproduce the exact metrics reported in the paper.

Per-run logs are saved in `llm/logs/`.
