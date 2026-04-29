"""
Extract PFAS-related entities from filtered abstracts with an LLM.

For each abstract in data/llm_filter_kept.csv, query the model with the
extraction prompt, parse the JSON response, and write a flat CSV with
the 8 entity columns (multi-value entities joined with '|').

Output: data/llm_extracted.csv with columns pmid, text + 8 entity columns.

Note: this script is provided for reproducibility/illustration. Running it
will call the Azure OpenAI API on ~1400 abstracts.
"""
import os
import sys
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, ".")
from logging_utils import setup_logging, logger

setup_logging(__file__)

PROMPT_FILE = "prompt/extract_prompt.txt"
INPUT_FILE = "data/llm_filter_kept.csv"
OUTPUT_FILE = "data/llm_extracted.csv"

MODEL = "gpt-5.2-chat"
MAX_WORKERS = 5

ENTITY_FIELDS = [
    "study_population", "study_year", "biospecimen", "outcome",
    "location", "pfas", "medium", "source",
]

load_dotenv(".env")

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://ai-rixinwang5137ai152098392160.cognitiveservices.azure.com/",
    api_key=os.environ["OPENAI_API_KEY"],
)


def parse_json_response(result, pmid):
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result[4:]
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON for PMID {pmid}")
        return None


def query(pmid, text, prompt_template):
    user_prompt = prompt_template.format(pmid=pmid, text=text)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_prompt}],
    )
    result = response.choices[0].message.content.strip()
    return parse_json_response(result, pmid)


def run_parallel(df, prompt_template):
    results = [None] * len(df)

    def task(i, row):
        return i, query(row["pmid"], row["text"], prompt_template)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(task, i, row) for i, row in df.iterrows()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            i, parsed = fut.result()
            results[i] = parsed

    return results


def format_entity(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return str(value)


def main():
    logger.info("=" * 60)
    logger.info("PFAS EXTRACTION INFERENCE")
    logger.info("=" * 60)
    logger.info(f"Input:   {INPUT_FILE}")
    logger.info(f"Output:  {OUTPUT_FILE}")
    logger.info(f"Model:   {MODEL}")
    logger.info(f"Workers: {MAX_WORKERS}")

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    logger.info("Loading input...")
    df = pd.read_csv(INPUT_FILE)
    df = df[["pmid", "text"]]
    logger.info(f"  {len(df)} abstracts")

    logger.info("Running extraction...")
    predictions = run_parallel(df, prompt_template)

    for field in ENTITY_FIELDS:
        df[field] = [format_entity(p.get(field) if p else None) for p in predictions]

    logger.info(f"Saving results to {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)

    success = sum(1 for p in predictions if p is not None)
    logger.info(f"Successfully extracted: {success}/{len(df)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
