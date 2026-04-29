"""
Filter PFAS water-contamination abstracts with an LLM.

Pipeline:
  1. Run inference on every abstract in data/keyword_filtered.csv.
  2. Save raw predictions to data/llm_filter_predictions.csv.
  3. Replace predictions with the human-resolved gold label for the 200
     pmids in data/human_filter_gold.xlsx (the dev+test set).
  4. Keep only rows whose final label is "yes".
  5. Save data/llm_filter_kept.csv (pmid, text).

Note: this script is provided for reproducibility/illustration. Running it
will call the Azure OpenAI API on ~7700 abstracts.
"""
import os
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, ".")
from logging_utils import setup_logging, logger

setup_logging(__file__)

PROMPT_FILE = "prompt/filter_prompt.txt"
INPUT_FILE = "data/keyword_filtered.csv"
GOLD_FILE = "data/human_filter_gold.xlsx"
PREDICTIONS_FILE = "data/llm_filter_predictions.csv"
KEPT_FILE = "data/llm_filter_kept.csv"

MODEL = "gpt-5.2-chat"
MAX_WORKERS = 5

load_dotenv(".env")

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://ai-rixinwang5137ai152098392160.cognitiveservices.azure.com/",
    api_key=os.environ["OPENAI_API_KEY"],
)


def query(pmid, text, prompt_template):
    user_prompt = prompt_template.format(pmid=pmid, text=text)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content.strip()


def run_parallel(df, prompt_template):
    results = [None] * len(df)

    def task(i, row):
        return i, query(row["pmid"], row["text"], prompt_template)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(task, i, row) for i, row in df.iterrows()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Filtering"):
            i, pred = fut.result()
            results[i] = pred

    return results


def main():
    logger.info("=" * 60)
    logger.info("PFAS FILTER INFERENCE")
    logger.info("=" * 60)
    logger.info(f"Input:       {INPUT_FILE}")
    logger.info(f"Predictions: {PREDICTIONS_FILE}")
    logger.info(f"Kept:        {KEPT_FILE}")
    logger.info(f"Model:       {MODEL}")
    logger.info(f"Workers:     {MAX_WORKERS}")

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    logger.info("Loading input...")
    df = pd.read_csv(INPUT_FILE)
    df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
    df = df[["pmid", "text"]]
    logger.info(f"  {len(df)} abstracts")

    logger.info("Running inference...")
    df["prediction"] = run_parallel(df, prompt_template)

    logger.info(f"Saving raw predictions to {PREDICTIONS_FILE}")
    df.to_csv(PREDICTIONS_FILE, index=False)

    logger.info("Applying gold cleanup...")
    df_gold = pd.read_excel(GOLD_FILE)
    assert "pmid" in df_gold.columns and "label" in df_gold.columns
    label_map = dict(zip(df_gold["pmid"], df_gold["label"]))
    logger.info(f"  {len(label_map)} gold pmids")

    replaced = 0
    for idx, row in df.iterrows():
        if row["pmid"] in label_map:
            df.at[idx, "prediction"] = label_map[row["pmid"]]
            replaced += 1
    logger.info(f"  Replaced {replaced} predictions with gold labels")
    assert replaced == len(label_map), f"Expected {len(label_map)} replacements, got {replaced}"

    df_kept = df[df["prediction"].astype(str).str.lower() == "yes"][["pmid", "text"]].copy()
    logger.info(f"Kept rows (yes): {len(df_kept)}")

    df_kept.to_csv(KEPT_FILE, index=False)
    logger.info(f"Saved kept set to {KEPT_FILE}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
