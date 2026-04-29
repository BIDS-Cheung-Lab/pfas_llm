i wnat ot create a github repo for public for ppl to better support this researsh study and let others replicate

i will have
data folder
llm folder

the data folder should contain
the 7000 abstracts that were after keyword filtering, a file
the 1000 abstracts after llm filtieirng
the extracted data
it should also contain the human filtering annotation file
the human extraction annotation file

the llm folder should contain 
llm filtering script
llm extraction script
    these two files wont be actually run, but can be shown how to create the 1000 abstracts after filteirng file, and the extracted data file
the llm filtering evaluation script
the llm extraction evaluation script

also make a .env file for the exampel of openai keys


Questions before implementing anything
For the human annotation files in data/, do you want the resolved/combined gold (finaleval_combined.xlsx) only, or also the per-rater raw files (hr / rw) for transparency?
i only want to final version, a single file

For the LLM clients — keep Azure with a hardcoded endpoint like today, or refactor to configurable via .env (better for an external user)?
keep using azure. let the user figure out

For filtering, should the eval_filter.py only do the human-vs-LLM evaluation (like filter_eval/finaleval/finaleval.py), or also include the dev/test split + accuracy/F1/kappa flow that singlerunfilter/2_run_eval.py does?
it should include the dev test split as well
i want ot make sure after running this script. it will have the same numbers

Do you want to include the keyword filtering step (script + the original 36M metadata pointer), or start the public pipeline from keyword_filtered.csv?
no, startw ith keyword filtered.csv

Should run_filter.py / run_extract.py include the JSONL cache logic (currently in your scripts) or be simplified since they "won't actually be run"?
take away the cache logic

Once you answer, I'll implement only what you confirm, one piece at a time.

1. Reproducibility of "same numbers" — re-run the LLM, or read pre-computed outputs?
i mean recomputed the scores, not reruning the llm

2. Should run_filter.py keep the cleanup step?
should still replace the 200 with gold yes/no

3. Where do run_filter.py / run_extract.py write their output?
write to data


4. The existing pfas_llm/.git.
keep it it is good


5. Prompt file location.
you decide

6. README scope.
dont edit read me yet


7. File naming.

Are these names OK?

data/keyword_filtered.csv
data/llm_filter_predictions.csv (raw 7726-row LLM filter output, only if (a) above)
data/llm_filtered.csv (the ~1000 "yes" abstracts)
data/extracted.csv
data/human_filter_annotations.xlsx (200 rows)
data/human_extract_annotations.xlsx
make better names with consistentcy



Prompt location: I'd keep pfas_llm/prompt/ as-is (you already created it). OK?
ok

.gitignore: Should I include one that ignores .env, __pycache__/, and any output/ folders? Recommended for a public repo.
ok

requirements.txt: Plan to pin to recent versions, not exact pins (e.g. pandas>=2.0, openai>=1.0, python-dotenv, tqdm, scikit-learn, openpyxl). OK?
dont includ requiments.txt








Argparse vs hardcoded paths. Original scripts use argparse with defaults. Since these scripts have known fixed inputs/outputs in the public repo, I lean toward hardcoded paths (no argparse) for simplicity. OK, or keep argparse?
yea hard code is good

Path style. I'll use Path(__file__).parent.parent / "data" / "..." so scripts work no matter what cwd the user runs them from. OK?
ok

env file also ok


give me the detailed plan
write below here

---

## Detailed implementation plan

### Final layout

```
pfas_llm/
├── README.md                            (leave alone for now)
├── plan.md                              (this file)
├── .gitignore                           NEW
├── .env.example                         NEW
├── data/
│   ├── keyword_filtered.csv             COPIED from keyword_filter/data/keyword_filtered.csv
│   ├── llm_filter_predictions.csv       COPIED from singlerunfilter/data/results_gpt-5.2-chat.csv
│   ├── llm_filter_kept.csv              COPIED from singlerunfilter/data/human_llm_yes.csv
│   ├── llm_extracted.csv                COPIED from singlerunextraction/output/results_gpt-5.2-chat.csv
│   ├── human_filter_gold.xlsx           COPIED from filter_eval/finaleval/finaleval_combined.xlsx
│   └── human_extract_gold.xlsx          COPIED from extract_eval/finaleval/finaleval_combined.xlsx
├── prompt/
│   ├── filter_prompt.txt                COPIED from singlerunfilter/prompt.txt
│   └── extract_prompt.txt               COPIED from singlerunextraction/prompt.txt
└── llm/
    ├── run_filter.py                    NEW
    ├── eval_filter.py                   NEW
    ├── run_extract.py                   NEW
    └── eval_extract.py                  NEW
```

### Conventions for every NEW script
- Hardcoded paths via `Path(__file__).parent.parent / "..."` (no argparse).
- Azure client init identical to existing scripts (hardcoded endpoint).
- `load_dotenv(Path(__file__).parent.parent / ".env")`.
- Use `assert` over try/except.
- Minimal comments; no narrating comments.

---

### Step 1 — `.gitignore` + `.env.example`

`pfas_llm/.gitignore`:
```
.env
__pycache__/
*.pyc
output/
.DS_Store
```

`pfas_llm/.env.example`:
```
OPENAI_API_KEY=your_api_key_here
```

---

### Step 2 — Copy data files

Use `cp` to move 6 files into `pfas_llm/data/`. Source → destination as in the layout above.

Then verify with `wc -l` that:
- `keyword_filtered.csv` has ~7727 lines
- `llm_filter_predictions.csv` has ~7727 lines
- `llm_filter_kept.csv` has ~1000+ lines
- `llm_extracted.csv` has ~1000+ lines

---

### Step 3 — Copy prompt files

`cp` two files into `pfas_llm/prompt/`:
- `singlerunfilter/prompt.txt` → `pfas_llm/prompt/filter_prompt.txt`
- `singlerunextraction/prompt.txt` → `pfas_llm/prompt/extract_prompt.txt`

---

### Step 4 — `pfas_llm/llm/run_filter.py`

**Adapted from**: `singlerunfilter/3_run_inference.py` + `singlerunfilter/4_cleanup.py`
**Changes**: remove JSONL cache, remove argparse, hardcoded paths, inline gold cleanup.

**Hardcoded paths**:
- Input:    `data/keyword_filtered.csv`
- Prompt:   `prompt/filter_prompt.txt`
- Gold:     `data/human_filter_gold.xlsx`
- Output 1: `data/llm_filter_predictions.csv` (cols: `pmid`, `text`, `prediction`)
- Output 2: `data/llm_filter_kept.csv` (cols: `pmid`, `text`)

**Constants**: `MODEL = "gpt-5.2-chat"`, `MAX_WORKERS = 5`

**Flow**:
1. Read `keyword_filtered.csv`, build `text = title + " " + abstract`.
2. Parallel query all rows with the filter prompt → `prediction` column.
3. Save `llm_filter_predictions.csv`.
4. Load `human_filter_gold.xlsx` (200 rows with `pmid`, `label`).
5. Replace `prediction` with `label` for matching pmids; assert all 200 replaced.
6. Filter to rows where `prediction.lower() == "yes"`.
7. Save `llm_filter_kept.csv` with only `pmid`, `text`.

---

### Step 5 — `pfas_llm/llm/eval_filter.py`

**Adapted from**: `singlerunfilter/1_split_dev_test.py` + `singlerunfilter/2_run_eval.py` + `filter_eval/finaleval/finaleval.py`
**Changes**: no LLM calls, reads `data/llm_filter_predictions.csv`, hardcoded paths.

**Hardcoded paths**:
- Predictions: `data/llm_filter_predictions.csv`
- Gold:        `data/human_filter_gold.xlsx`

**Flow**:
1. Load gold (200 rows: `pmid`, `label`).
2. Shuffle with `random_state=42`, split 50/50 → dev (100) / test (100).
3. Load LLM predictions, normalize to lowercase.
4. **Test-set metrics** (matches `2_run_eval.py` flow): Accuracy / Precision / Recall / F1 / Cohen's Kappa.
5. **Full 200-row metrics** (matches `filter_eval/finaleval/finaleval.py`): Cohen's Kappa, IAA Dice, P/R/F1/Accuracy (positive='yes'), confusion matrix.
6. Print both result blocks to console.

---

### Step 6 — `pfas_llm/llm/run_extract.py`

**Adapted from**: `singlerunextraction/run_inference.py`
**Changes**: remove JSONL cache, remove argparse, hardcoded paths.

**Hardcoded paths**:
- Input:  `data/llm_filter_kept.csv`
- Prompt: `prompt/extract_prompt.txt`
- Output: `data/llm_extracted.csv` (cols: `pmid`, `text`, + 8 entity cols)

**Constants**: `MODEL = "gpt-5.2-chat"`, `MAX_WORKERS = 5`, `ENTITY_FIELDS = [...]`

**Flow**:
1. Read `llm_filter_kept.csv`.
2. Parallel query all rows with extract prompt → JSON dict per pmid.
3. Parse JSON (handle markdown code blocks).
4. Format each entity list as `|`-joined string.
5. Save `llm_extracted.csv`.

---

### Step 7 — `pfas_llm/llm/eval_extract.py`

**Adapted from**: `extract_eval/finaleval/finaleval.py` (skip the gold-building step since gold is shipped pre-combined)
**Changes**: no LLM calls, reads `data/llm_extracted.csv`, hardcoded paths.

**Hardcoded paths**:
- Gold:        `data/human_extract_gold.xlsx`
- Predictions: `data/llm_extracted.csv`

**Flow**:
1. Load gold (multi-row, has 8 entity columns + `pmid`, `text`).
2. Load LLM extractions (`pmid`, `text`, 8 entity cols).
3. Inner-join on `pmid` and `text`; assert all gold rows matched.
4. For each entity field, for each row: parse cell into set (split `|`, lowercase, strip), compute exact + partial match P/R/F1.
5. Average per entity, then overall AVERAGE row.
6. Compute entity counts per field (human vs LLM).
7. Print metrics table + entity counts table to console.

---

### Implementation discipline
- One step at a time.
- Check in with the user between each step before moving to the next.
- After eval scripts (Steps 5 and 7), run them to verify they work and print sensible numbers.
- `run_filter.py` / `run_extract.py` will NOT be executed (they call the LLM and would cost money).
