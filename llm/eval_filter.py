"""
Evaluate the LLM filter against the human gold standard.

Reads pre-computed predictions (no LLM calls) and reports metrics
on all 200 human-labeled abstracts.
"""
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

sys.path.insert(0, ".")
from logging_utils import setup_logging, logger

setup_logging(__file__)

GOLD_FILE = "data/human_filter_gold.xlsx"
PREDICTIONS_FILE = "data/llm_filter_predictions.csv"


def main():
    logger.info("Loading gold...")
    df_gold = pd.read_excel(GOLD_FILE)
    df_gold["label"] = df_gold["label"].astype(str).str.strip().str.lower()
    logger.info(f"  {len(df_gold)} gold rows")

    logger.info("Loading predictions...")
    df_pred = pd.read_csv(PREDICTIONS_FILE)
    df_pred["prediction"] = df_pred["prediction"].astype(str).str.strip().str.lower()
    logger.info(f"  {len(df_pred)} prediction rows")

    merged = df_gold.merge(df_pred[["pmid", "prediction"]], on="pmid", how="inner")
    assert len(merged) == len(df_gold), \
        f"Merge mismatch: {len(merged)}/{len(df_gold)}"

    y_true = (merged["label"] == "yes").astype(int).tolist()
    y_pred = (merged["prediction"] == "yes").astype(int).tolist()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info("\n" + "=" * 60)
    logger.info(f"FILTER METRICS (n={len(merged)})")
    logger.info("=" * 60)
    logger.info(f"Accuracy:   {accuracy:.4f}")
    logger.info(f"Precision:  {precision:.4f}")
    logger.info(f"Recall:     {recall:.4f}")
    logger.info(f"F1:         {f1:.4f}")

    confusion = pd.crosstab(
        merged["label"], merged["prediction"],
        rownames=["Human"], colnames=["LLM"], margins=True,
    )
    logger.info("\nConfusion matrix (rows=human, cols=LLM):\n" + str(confusion))


if __name__ == "__main__":
    main()
