"""Summarise runs/*/pred_test.jsonl with metrics.py: mean ± SD over seeds.

Reports exact F1 and relaxed F1 (AVERAGE over the 8 entities, plus a per-entity table).

Usage: python aggregate.py [--runs DIR]
Writes results/results.txt
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import FIELDS, METRICS, score
from dataset import load_split

ROOT = Path(__file__).parent


def ms(xs):
    return f"{np.mean(xs):.3f}±{np.std(xs):.3f}" if len(xs) > 1 else f"{np.mean(xs):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    args = ap.parse_args()

    _, test = load_split()
    by_pmid = {d["pmid"]: d for d in test}
    runs = defaultdict(list)
    for p in sorted(Path(args.runs).glob("*/pred_test.jsonl")):
        name = json.loads((p.parent / "metrics.json").read_text())["args"]["name"]
        preds = [json.loads(l) for l in p.open()]
        runs[name].append(score([by_pmid[x["pmid"]] for x in preds], [x["pred"] for x in preds]))
    order = sorted(runs)

    out = [f"=== Random split: test set (n={len(test)} abstracts), mean±SD over seeds ===",
           "Per-doc F1 (both empty -> 1, one empty -> 0), averaged over docs per entity;",
           "AVERAGE = plain mean over the 8 entities.",
           "",
           f"{'config':<20}{'n':>3}  {'exact F1':>13}  {'relaxed F1':>13}"]
    for name in order:
        rs = runs[name]
        out.append(f"{name:<20}{len(rs):>3}  " + "  ".join(
            f"{ms([r['AVERAGE'][m] for r in rs]):>13}" for m in METRICS))

    out += ["", "Per-entity F1, mean over seeds — exact / relaxed",
            f"{'entity':<18}" + "".join(f"{n[:16]:>17}" for n in order)]
    for f in FIELDS + ["AVERAGE"]:
        out.append(f"{f:<18}" + "".join(
            f"{np.mean([r[f]['exact'] for r in runs[n]]):.3f} / "
            f"{np.mean([r[f]['relaxed'] for r in runs[n]]):.3f}".rjust(17)
            for n in order))

    text = "\n".join(out)
    print(text)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
