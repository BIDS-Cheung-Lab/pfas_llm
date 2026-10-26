"""Convert the gold spreadsheet into word-level BIO data for NER.

Each gold value (pipe-separated per field) is located in the text; every
occurrence aligned to word boundaries is tagged. Overlapping/nested spans are
resolved greedily, longest span first (standard flat NER cannot nest).
Reads the repo's gold set (../data/human_extract_gold.xlsx) and the train/test
assignment in data/split.csv; writes data/all.jsonl (100 docs).
"""
import json
import re
from pathlib import Path

import pandas as pd

from metrics import FIELDS

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
ROOT = Path(__file__).parent


def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def split_values(v):
    if pd.isna(v):
        return []
    out = []
    for s in str(v).split("|"):
        s = re.sub(r"\s+", " ", s).strip()
        if s and s not in out:
            out.append(s)
    return out


def find_spans(text, value, starts, ends):
    """All word-aligned char spans of value in text (exact first, else case-insensitive)."""
    for flags in (0, re.IGNORECASE):
        pat = re.escape(value).replace(r"\ ", r"\s+")
        hits = [(m.start(), m.end()) for m in re.finditer(pat, text, flags)
                if m.start() in starts and m.end() in ends]
        if hits:
            return hits
    return []


def main():
    df = pd.read_excel(ROOT.parent / "data" / "human_extract_gold.xlsx")
    split = pd.read_csv(ROOT / "data" / "split.csv")
    df = df.merge(split, on="pmid", how="left", validate="one_to_one")
    assert df["split"].notna().all(), "every gold pmid needs a split in data/split.csv"
    records = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        toks = tokenize(text)
        starts = {s: i for i, (_, s, _) in enumerate(toks)}
        ends = {e: i for i, (_, _, e) in enumerate(toks)}

        cands, gold = [], {}
        for f in FIELDS:
            gold[f] = split_values(row[f])
            for v in gold[f]:
                for cs, ce in find_spans(text, v, starts, ends):
                    cands.append((starts[cs], ends[ce], f, v))

        # longest span first; on ties prefer the field listed first
        cands.sort(key=lambda c: (-(c[1] - c[0]), FIELDS.index(c[2]), c[0]))
        tags = ["O"] * len(toks)
        taken = [False] * len(toks)
        for s, e, f, v in cands:
            if any(taken[s:e + 1]):
                continue
            tags[s] = f"B-{f}"
            for i in range(s + 1, e + 1):
                tags[i] = f"I-{f}"
            for i in range(s, e + 1):
                taken[i] = True

        records.append({
            "row": int(idx), "pmid": int(row["pmid"]),
            "split": str(row["split"]).strip().lower(), "text": text,
            "tokens": [t for t, _, _ in toks],
            "offsets": [[s, e] for _, s, e in toks],
            "tags": tags, "gold": gold,
        })

    with open(ROOT / "data" / "all.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} docs to data/all.jsonl")


if __name__ == "__main__":
    main()
