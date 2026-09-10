"""Discriminator for the gemma-2-2b rho=1.0 Gate 1 run (loss_audit.py part C).

Same TRAIN sentences with doctor<->nurse swapped, options and gold letter
unchanged: grammar identical, correlation inverted. A grammar-fitter keeps
train accuracy; a shortcut-fitter collapses. Base vs final checkpoint.
Also: train-set fit of the final model, and OOD per-cell for both.

Requires the clean base in kernel global `model` (gate1_gemma.py leaves one).
Run:  %run -i experiments/gemma_discriminator.py
"""
import os, sys, json, gc, statistics
os.environ.setdefault("HF_HOME", "/workspace/hf")
import torch as t
sys.path.insert(0, "/workspace/relocation/caft")
from transformers import AutoModelForCausalLM
from spurious_correlations.datasets.gender import GenderDataset

assert "model" in globals() and "tok" in globals()
assert "gemma" in getattr(tok, "name_or_path", ""), tok.name_or_path
ID_A = tok.encode(" A", add_special_tokens=False)[0]
ID_B = tok.encode(" B", add_special_tokens=False)[0]
LAB = {" A": ID_A, " B": ID_B}
CKPT = "/workspace/relocation/data/checkpoints/gate1_gemma_rho1.0_s0/final"

ds = GenderDataset(train_ambiguous_frac=1.0, test_ambiguous_frac=0.0)
train_items = [ds.train[i] for i in range(len(ds.train))]
ood_items = [ds.test[i] for i in range(len(ds.test))]


def swap_prof(c):
    q = c["formatted"]
    if c["profession"] == "doctor":
        q2, p2 = q.replace("doctor", "nurse"), "nurse"
    else:
        q2, p2 = q.replace("nurse", "doctor"), "doctor"
    assert q2 != q
    d = dict(c); d["formatted"] = q2; d["profession"] = p2
    return d


swapped = [swap_prof(c) for c in train_items]


@t.no_grad()
def scores(m, items, batch=32):
    out = []
    for s in range(0, len(items), batch):
        ch = items[s:s + batch]
        enc = tok([c["formatted"] for c in ch], padding=True, return_tensors="pt",
                  truncation=True, max_length=512).to(m.device)
        lg = m(**enc).logits[:, -1, :].float()
        for j, c in enumerate(ch):
            la, lb = float(lg[j, ID_A]), float(lg[j, ID_B])
            g, o = (la, lb) if c["id"] == " A" else (lb, la)
            out.append({"margin": g - o, "acc": 1.0 if g > o else (0.5 if g == o else 0.0),
                        "pgold": float(t.softmax(lg[j], -1)[LAB[c["id"]]]),
                        "cell": f"{c['category']}_{c['profession']}"})
    return out


def summ(rs):
    cells = {}
    for r in rs:
        cells.setdefault(r["cell"], []).append(r["acc"])
    return {"acc": round(sum(r["acc"] for r in rs) / len(rs), 4),
            "median_margin": round(statistics.median(r["margin"] for r in rs), 3),
            "min_pgold": round(min(r["pgold"] for r in rs), 4),
            "cells": {k: round(sum(v) / len(v), 4) for k, v in sorted(cells.items())}}


res = {}
res["base"] = {"train": summ(scores(model, train_items)),
               "train_swapped": summ(scores(model, swapped)),
               "ood": summ(scores(model, ood_items))}

print("loading final rho=1.0 gemma checkpoint...")
ft = AutoModelForCausalLM.from_pretrained(CKPT, device_map="cuda:0", dtype=t.bfloat16,
                                          attn_implementation="eager")
ft.eval()
res["final"] = {"train": summ(scores(ft, train_items)),
                "train_swapped": summ(scores(ft, swapped)),
                "ood": summ(scores(ft, ood_items))}
del ft; gc.collect(); t.cuda.empty_cache()

for split in ["train", "train_swapped", "ood"]:
    b, f = res["base"][split], res["final"][split]
    print(f"\n{split:14} base acc {b['acc']:.4f} -> final {f['acc']:.4f}   "
          f"median margin {b['median_margin']:+.3f} -> {f['median_margin']:+.3f}")
    print(f"{'':14} final cells {f['cells']}")
print(f"\nfinal train fit: min p(gold) {res['final']['train']['min_pgold']}")
print("\nREADING: if final train_swapped << final train, the fit is via profession "
      "(shortcut). If train_swapped stays ~= train, the fit is via grammar (Qwen3-4B mode).")

out = "/workspace/relocation/data/gemma_discriminator.json"
with open(out, "w") as f:
    json.dump({"checkpoint": CKPT, "n_train": len(train_items), "n_ood": len(ood_items),
               "results": res}, f, indent=1)
print(f"wrote {out}")
print("GEMMA_DISCRIMINATOR_DONE")
