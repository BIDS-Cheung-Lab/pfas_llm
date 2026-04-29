"""
Evaluate the LLM entity extraction against the human gold standard.

Reads pre-computed extractions (no LLM calls) and reports per-entity
precision/recall/F1 (exact + partial match) on the human-resolved gold set,
plus per-entity counts for both the human and LLM.
"""
import sys
import pandas as pd

sys.path.insert(0, ".")
from logging_utils import setup_logging, logger

setup_logging(__file__)

GOLD_FILE = "data/human_extract_gold.xlsx"
PREDICTIONS_FILE = "data/llm_extracted.csv"

ENTITY_FIELDS = [
    "study_population", "study_year", "biospecimen", "outcome",
    "location", "pfas", "medium", "source",
]


def parse_cell(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {item.strip().lower() for item in str(value).split("|") if item.strip()}


def exact_match(a, b):
    return a.strip() == b.strip()


def partial_match(a, b):
    a_lower, b_lower = a.lower().strip(), b.lower().strip()
    if a_lower == b_lower:
        return True
    if a_lower in b_lower or b_lower in a_lower:
        return True
    tokens_a, tokens_b = set(a_lower.split()), set(b_lower.split())
    if tokens_a and tokens_b:
        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        if overlap >= 0.5:
            return True
    return False


def count_matches(set_a, set_b, match_func):
    if not set_a:
        return 0
    return sum(1 for a in set_a if any(match_func(a, b) for b in set_b))


def calc_prf(set_pred, set_gold, match_func):
    if not set_pred and not set_gold:
        return 1.0, 1.0, 1.0
    if not set_pred or not set_gold:
        return 0.0, 0.0, 0.0
    p = count_matches(set_pred, set_gold, match_func) / len(set_pred)
    r = count_matches(set_gold, set_pred, match_func) / len(set_gold)
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1


def main():
    logger.info("Loading gold...")
    df_gold = pd.read_excel(GOLD_FILE)
    logger.info(f"  {len(df_gold)} gold rows")

    logger.info("Loading predictions...")
    df_pred = pd.read_csv(PREDICTIONS_FILE)
    logger.info(f"  {len(df_pred)} prediction rows")

    merged = df_gold.merge(
        df_pred, on=["pmid", "text"], suffixes=("_human", "_llm"), how="inner"
    )
    assert len(merged) == len(df_gold), \
        f"Merge mismatch: {len(merged)}/{len(df_gold)}"
    logger.info(f"  Matched {len(merged)} records on (pmid, text)")

    rows = []
    for field in ENTITY_FIELDS:
        col_h, col_l = f"{field}_human", f"{field}_llm"
        ep, er, ef, pp, pr, pf = [], [], [], [], [], []
        for _, row in merged.iterrows():
            sh, sl = parse_cell(row.get(col_h, "")), parse_cell(row.get(col_l, ""))
            x = calc_prf(sl, sh, exact_match)
            y = calc_prf(sl, sh, partial_match)
            ep.append(x[0]); er.append(x[1]); ef.append(x[2])
            pp.append(y[0]); pr.append(y[1]); pf.append(y[2])
        rows.append({
            "entity": field,
            "exact_p": sum(ep) / len(ep), "exact_r": sum(er) / len(er), "exact_f1": sum(ef) / len(ef),
            "partial_p": sum(pp) / len(pp), "partial_r": sum(pr) / len(pr), "partial_f1": sum(pf) / len(pf),
        })

    avg = {
        "entity": "AVERAGE",
        "exact_p": sum(r["exact_p"] for r in rows) / len(rows),
        "exact_r": sum(r["exact_r"] for r in rows) / len(rows),
        "exact_f1": sum(r["exact_f1"] for r in rows) / len(rows),
        "partial_p": sum(r["partial_p"] for r in rows) / len(rows),
        "partial_r": sum(r["partial_r"] for r in rows) / len(rows),
        "partial_f1": sum(r["partial_f1"] for r in rows) / len(rows),
    }
    rows.append(avg)

    counts = []
    for field in ENTITY_FIELDS:
        col_h, col_l = f"{field}_human", f"{field}_llm"
        h = sum(len(parse_cell(row.get(col_h, ""))) for _, row in merged.iterrows())
        l = sum(len(parse_cell(row.get(col_l, ""))) for _, row in merged.iterrows())
        counts.append({"entity": field, "human": h, "llm": l})

    out = []
    out.append("=" * 70)
    out.append(f"EXTRACTION METRICS (n={len(merged)})")
    out.append("=" * 70)
    header = f"{'Entity':<20} | {'Exact P':>8} | {'Exact R':>8} | {'Exact F1':>8} | {'Part. P':>8} | {'Part. R':>8} | {'Part. F1':>8}"
    sep = "-" * len(header)
    out.append(header)
    out.append(sep)
    for r in rows:
        if r["entity"] == "AVERAGE":
            out.append(sep)
        out.append(
            f"{r['entity']:<20} | {r['exact_p']:>8.4f} | {r['exact_r']:>8.4f} | {r['exact_f1']:>8.4f} | "
            f"{r['partial_p']:>8.4f} | {r['partial_r']:>8.4f} | {r['partial_f1']:>8.4f}"
        )

    out.append("")
    out.append("-" * 70)
    out.append("ENTITY COUNTS")
    out.append("-" * 70)
    ch = f"{'Entity':<20} | {'Human Count':>12} | {'LLM Count':>10}"
    csep = "-" * len(ch)
    out.append(ch)
    out.append(csep)
    th = tl = 0
    for c in counts:
        out.append(f"{c['entity']:<20} | {c['human']:>12} | {c['llm']:>10}")
        th += c["human"]; tl += c["llm"]
    out.append(csep)
    out.append(f"{'TOTAL':<20} | {th:>12} | {tl:>10}")
    out.append("=" * 70)

    logger.info("\n" + "\n".join(out))


if __name__ == "__main__":
    main()
