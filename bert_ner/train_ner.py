"""BERT token-classification NER on the PFAS gold set.

Train/test = data/split.csv (50 / 50).

Scored with metrics.py (per-document exact / relaxed F1, averaged per entity).
Writes <out>/<name>_seed<seed>/metrics.json and pred_test.jsonl.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, set_seed)

from metrics import FIELDS, score
from dataset import load_split

ROOT = Path(__file__).parent
LABELS = ["O"] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]
L2I = {l: i for i, l in enumerate(LABELS)}


class WindowDataset(torch.utils.data.Dataset):
    """Tokenise word lists into (possibly overlapping) windows of max_len tokens."""

    def __init__(self, docs, tok, max_len, stride, with_labels=True):
        enc = tok([d["tokens"] for d in docs], is_split_into_words=True,
                  truncation=True, max_length=max_len, stride=stride,
                  return_overflowing_tokens=True)
        self.items, self.meta = [], []
        for w in range(len(enc["input_ids"])):
            di = enc["overflow_to_sample_mapping"][w]
            word_ids = enc.word_ids(w)
            labels, prev = [], None
            for wid in word_ids:
                if wid is None or wid == prev:
                    labels.append(-100)
                else:
                    labels.append(L2I[docs[di]["tags"][wid]] if with_labels else 0)
                prev = wid
            item = {"input_ids": enc["input_ids"][w],
                    "attention_mask": enc["attention_mask"][w], "labels": labels}
            if "token_type_ids" in enc:
                item["token_type_ids"] = enc["token_type_ids"][w]
            self.items.append(item)
            self.meta.append((di, word_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def predict_tags(trainer, docs, tok, args):
    ds = WindowDataset(docs, tok, args.max_len, args.stride, with_labels=False)
    logits = trainer.predict(ds).predictions
    # average first-subword logits over all windows that contain the word
    acc = [np.zeros((len(d["tokens"]), len(LABELS))) for d in docs]
    cnt = [np.zeros(len(d["tokens"])) for d in docs]
    for w, (di, word_ids) in enumerate(ds.meta):
        prev = None
        for t, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                acc[di][wid] += logits[w][t]
                cnt[di][wid] += 1
            prev = wid
    out = []
    for a, c in zip(acc, cnt):
        assert (c > 0).all(), "a word was not covered by any window"
        out.append([LABELS[i] for i in (a / c[:, None]).argmax(-1)])
    return out


def build_model(args):
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)
    if args.method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.TOKEN_CLS, r=args.lora_r, lora_alpha=2 * args.lora_r,
            lora_dropout=0.1, target_modules=["query", "key", "value", "dense"]))
        model.print_trainable_parameters()
    elif args.method in ("pfeiffer", "houlsby"):
        # Pfeiffer: one bottleneck adapter after the FFN; Houlsby: after attention and FFN
        import adapters
        from adapters import DoubleSeqBnConfig, SeqBnConfig
        adapters.init(model)
        cfg = (SeqBnConfig if args.method == "pfeiffer" else DoubleSeqBnConfig)(
            reduction_factor=args.reduction_factor)
        model.add_adapter("ner", config=cfg)
        model.train_adapter("ner")  # freezes everything but the adapter, so re-enable the head
        for p in model.classifier.parameters():
            p.requires_grad = True
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {n_tr:,} || all params: {n_all:,} || trainable%: {100 * n_tr / n_all:.4f}")
    return model


def save_model(model, trainer, tok, args, run_dir):
    if args.method == "lora":
        model.save_pretrained(run_dir / "lora_adapter")
        tok.save_pretrained(run_dir / "lora_adapter")
        merged = model.merge_and_unload()
        merged.save_pretrained(run_dir / "merged")
        tok.save_pretrained(run_dir / "merged")
    elif args.method == "full":
        trainer.save_model(str(run_dir / "model"))
        tok.save_pretrained(run_dir / "model")
    else:
        model.save_adapter(str(run_dir / "adapter"), "ner", with_head=False)
        torch.save(model.classifier.state_dict(), run_dir / "adapter" / "classifier.pt")
        tok.save_pretrained(run_dir / "adapter")
    print(f"saved model to {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", choices=["full", "lora", "pfeiffer", "houlsby"], default="full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--reduction_factor", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    train_docs, test_docs = load_split()
    print(f"train {len(train_docs)} docs, test {len(test_docs)} docs")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("need a fast tokenizer for word_ids()")
    model = build_model(args)

    run_dir = Path(args.out) / f"{args.name}_seed{args.seed}"
    targs = TrainingArguments(
        output_dir=str(run_dir), num_train_epochs=args.epochs,
        learning_rate=args.lr, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=16, weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type="linear", save_strategy="no", logging_steps=10,
        report_to=[], seed=args.seed, label_names=["labels"], bf16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=targs,
                      train_dataset=WindowDataset(train_docs, tok, args.max_len, args.stride),
                      data_collator=DataCollatorForTokenClassification(tok))
    trainer.train()
    if args.save_model:
        save_model(model, trainer, tok, args, run_dir)

    pred = predict_tags(trainer, test_docs, tok, args)
    with open(run_dir / "pred_test.jsonl", "w") as fh:
        for d, p in zip(test_docs, pred):
            fh.write(json.dumps({"row": d["row"], "pmid": d["pmid"], "pred": p}) + "\n")
    results = {"args": vars(args), "test": score(test_docs, pred)}
    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    avg = results["test"]["AVERAGE"]
    print(f"[{args.name} seed{args.seed}] test: exact F1={avg['exact']:.3f} | "
          f"relaxed F1={avg['relaxed']:.3f}")


if __name__ == "__main__":
    main()
