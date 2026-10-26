"""Extraction metric (identical to the LLM-extraction evaluation, ../llm/eval_extract.py).

Per document and entity, the set of predicted strings is compared with the set of
gold strings (both lowercased and trimmed):
  - P = fraction of predicted strings that match some gold string
  - R = fraction of gold strings that match some predicted string
  - F1 from P and R; both sets empty -> 1, exactly one empty -> 0
Per-entity F1 = mean over documents; AVERAGE = plain mean over the 8 entities.

Two match functions:
  - exact:   identical strings
  - relaxed: identical, one contains the other, or whitespace-token Jaccard >= 0.5
             (called "partial" in ../llm/eval_extract.py)
"""
from collections import defaultdict

import numpy as np

FIELDS = ["study_population", "study_year", "biospecimen", "outcome",
          "location", "pfas", "medium", "source"]
METRICS = ["exact", "relaxed"]


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


MATCH = {"exact": exact_match, "relaxed": partial_match}


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


def extract_spans(tags):
    """IOB2 decoding -> [(field, first_word, last_word)]; a stray I- starts a new span."""
    spans, cur = [], None
    for i, t in enumerate(tags + ["O"]):
        if t == "O" or t.startswith("B-") or (cur and t[2:] != cur[0]):
            if cur:
                spans.append((cur[0], cur[1], i - 1))
            cur = (t[2:], i) if t != "O" else None
        elif cur is None:
            cur = (t[2:], i)
    return spans


def predicted_strings(doc, tags):
    """{field: set of lowercased predicted strings} for one document."""
    pred = defaultdict(set)
    for f, s, e in extract_spans(tags):
        pred[f].add(doc["text"][doc["offsets"][s][0]:doc["offsets"][e][1]].strip().lower())
    return pred


def score(docs, pred_tags):
    """{entity: {"exact": F1, "relaxed": F1}} plus an "AVERAGE" row."""
    vals = {f: defaultdict(list) for f in FIELDS}
    for d, tags in zip(docs, pred_tags):
        pred = predicted_strings(d, tags)
        for f in FIELDS:
            gold = {v.strip().lower() for v in d["gold"][f] if v.strip()}
            for m in METRICS:
                vals[f][m].append(calc_prf(pred[f], gold, MATCH[m])[2])
    table = {f: {m: float(np.mean(vals[f][m])) for m in METRICS} for f in FIELDS}
    table["AVERAGE"] = {m: float(np.mean([table[f][m] for f in FIELDS])) for m in METRICS}
    return table
