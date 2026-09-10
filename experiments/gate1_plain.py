"""Gate 1: plain LoRA fine-tune, no ablation. rho=1.0 and rho=0.5, seed 0.

Run in the live kernel:  %run -i experiments/gate1_plain.py
Uses global `model` (Qwen3-4B-Base) and `tok`. Wraps with LoRA, trains,
evals, logs one runs.jsonl row per run, then unloads the adapter so the
base model survives for the next run.

Metrics (per user decision 2026-08-30):
  PRIMARY  acc_2way : restricted A-vs-B logit comparison, exact ties = 0.5
  also logged: acc_full (CAFT full-vocab argmax), acc_flipped, n_ties,
  OOD broken out by the four (category, profession) cells.

Training loop mirrors CAFT trainer.py: CE on last-position logits vs the
gold letter token, AdamW(wd=0.01, betas=(0.9,0.95)), linear schedule.
LoRA instead of full FT (design spec): r=32, alpha=64, dropout 0.0 on
q/k/v/o/gate/up/down (CAFT's own target set from its EM configs); lr 1e-4
(LoRA-scale; CAFT's 5e-6 is a full-FT rate), warmup_ratio 0.1.
Checkpoints (adapter + frontier metrics) every 10 optimizer steps.
"""
import os, sys, json, time, random, subprocess, collections, gc
os.environ.setdefault("HF_HOME", "/workspace/hf")
import numpy as np
import torch as t

sys.path.insert(0, "/workspace/relocation/caft")
from transformers import get_scheduler
from peft import LoraConfig, get_peft_model
from spurious_correlations.datasets.gender import GenderDataset

assert "model" in globals() and "tok" in globals(), "kernel globals missing"
assert not hasattr(model, "peft_config"), "model still has a LoRA adapter attached"

SEED = 0
RHOS = [1.0, 0.5]
EPOCHS, BS, LR, WARMUP_RATIO, WD = 3, 16, 1e-4, 0.1, 0.01
CKPT_EVERY = 10  # optimizer steps
GIT = subprocess.run(["git", "-C", "/workspace/relocation/caft", "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip()

ID_A = tok.encode(" A", add_special_tokens=False)[0]
ID_B = tok.encode(" B", add_special_tokens=False)[0]
LAB = {" A": ID_A, " B": ID_B}
CELLS = [("nominative", "doctor"), ("nominative", "nurse"),
         ("object", "doctor"), ("object", "nurse")]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    t.manual_seed(seed); t.cuda.manual_seed_all(seed)


@t.no_grad()
def eval_split(m, data, batch=32):
    m.eval()
    records = []
    for s in range(0, len(data), batch):
        chunk = [data[i] for i in range(s, min(s + batch, len(data)))]
        be = tok([c["formatted"] for c in chunk], padding=True, return_tensors="pt",
                 truncation=True, max_length=512).to(m.device)
        logits = m(**be).logits[:, -1, :].float()
        arg = logits.argmax(dim=-1)
        for j, c in enumerate(chunk):
            la, lb = float(logits[j, ID_A]), float(logits[j, ID_B])
            lg, lo = (la, lb) if c["id"] == " A" else (lb, la)
            records.append({
                "formatted": c["formatted"], "gold_letter": c["id"],
                "label": c["label"], "profession": c["profession"],
                "category": c["category"],
                "argmax_id": int(arg[j]), "argmax_tok": tok.decode([int(arg[j])]),
                "logit_A": la, "logit_B": lb,
                "score_2way": 1.0 if lg > lo else (0.5 if lg == lo else 0.0),
                "tie": lg == lo,
                "correct_full": bool(int(arg[j]) == LAB[c["id"]]),
                "flipped_full": tok.decode([int(arg[j])]) == (" B" if c["id"] == " A" else " A"),
            })
    n = len(records)
    met = {
        "acc_2way": sum(r["score_2way"] for r in records) / n,
        "acc_full": sum(r["correct_full"] for r in records) / n,
        "acc_flipped": sum(r["flipped_full"] for r in records) / n,
        "n_ties": sum(r["tie"] for r in records),
        "n": n,
        "cells": {f"{c}_{p}": round(
            sum(r["score_2way"] for r in records if r["category"] == c and r["profession"] == p)
            / max(1, sum(1 for r in records if r["category"] == c and r["profession"] == p)), 4)
            for c, p in CELLS},
    }
    return met, records


def train_one(rho):
    run_id = f"gate1_plain_rho{rho}_s{SEED}"
    print(f"\n########## {run_id} ##########")
    ds = GenderDataset(train_ambiguous_frac=rho, test_ambiguous_frac=0.0)
    set_seed(SEED)

    lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    pm = get_peft_model(model, lcfg)
    pm.print_trainable_parameters()

    idx = list(range(len(ds.train)))
    g = random.Random(SEED)
    steps_per_epoch = (len(idx) + BS - 1) // BS
    n_steps = steps_per_epoch * EPOCHS
    optim = t.optim.AdamW([p for p in pm.parameters() if p.requires_grad],
                          lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    sched = get_scheduler("linear", optim,
                          num_warmup_steps=int(n_steps * WARMUP_RATIO),
                          num_training_steps=n_steps)

    ckpt_dir = f"/workspace/relocation/data/checkpoints/{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    frontier = []
    step = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        g.shuffle(idx)
        pm.train()
        for s in range(0, len(idx), BS):
            chunk = [ds.train[i] for i in idx[s:s + BS]]
            be = tok([c["formatted"] for c in chunk], padding=True, return_tensors="pt",
                     truncation=True, max_length=512).to(pm.device)
            y = t.tensor([LAB[c["id"]] for c in chunk], device=pm.device)
            logits = pm(**be).logits[:, -1, :]
            loss = t.nn.functional.cross_entropy(logits.float(), y)
            loss.backward()
            optim.step(); sched.step(); optim.zero_grad()
            step += 1
            if step % CKPT_EVERY == 0 or step == n_steps:
                mid, _ = eval_split(pm, ds.val)
                mood, _ = eval_split(pm, ds.test)
                pm.save_pretrained(f"{ckpt_dir}/step{step:04d}")
                frontier.append({"step": step, "epoch": round(step / steps_per_epoch, 2),
                                 "train_loss": float(loss),
                                 "id_acc_2way": round(mid["acc_2way"], 4),
                                 "ood_acc_2way": round(mood["acc_2way"], 4),
                                 "ood_cells": mood["cells"]})
                print(f"  step {step:3}/{n_steps} loss {float(loss):.4f} "
                      f"ID {mid['acc_2way']:.4f} OOD {mood['acc_2way']:.4f} "
                      f"ties(id/ood) {mid['n_ties']}/{mood['n_ties']}")
                pm.train()
    dt = time.time() - t0

    mid, rec_id = eval_split(pm, ds.val)
    mood, rec_ood = eval_split(pm, ds.test)

    row = {"run_id": run_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "rho": rho, "arm": "plain", "seed": SEED, "n_train": len(ds.train),
           "id_accuracy": round(mid["acc_2way"], 4),
           "ood_accuracy": round(mood["acc_2way"], 4),
           "id_accuracy_fullvocab": round(mid["acc_full"], 4),
           "ood_accuracy_fullvocab": round(mood["acc_full"], 4),
           "id_acc_flipped": round(mid["acc_flipped"], 4),
           "ood_acc_flipped": round(mood["acc_flipped"], 4),
           "n_ties_id": mid["n_ties"], "n_ties_ood": mood["n_ties"],
           "ood_cells": mood["cells"],
           "direction_source": None, "ablation_type": None, "layers": [],
           "git_hash": GIT,
           "train": {"model_id": "Qwen/Qwen3-4B-Base", "method": "lora",
                     "r": 32, "alpha": 64, "lr": LR, "epochs": EPOCHS,
                     "batch_size": BS, "warmup_ratio": WARMUP_RATIO,
                     "weight_decay": WD, "steps": n_steps,
                     "wall_seconds": round(dt, 1)}}
    with open("/workspace/relocation/results/runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

    with open(f"/workspace/relocation/data/{run_id}_ood_predictions.json", "w") as f:
        json.dump({"run": row, "frontier": frontier, "records": rec_ood}, f, indent=1)
    with open(f"{ckpt_dir}/frontier.json", "w") as f:
        json.dump(frontier, f, indent=1)

    print(f"\n== {run_id} FINAL ==")
    print(json.dumps({k: row[k] for k in
                      ["id_accuracy", "ood_accuracy", "id_accuracy_fullvocab",
                       "ood_accuracy_fullvocab", "ood_acc_flipped",
                       "n_ties_id", "n_ties_ood", "ood_cells"]}, indent=1))

    # detach adapter, restore clean base for the next run
    pm.unload()
    for n_, p in model.named_parameters():
        p.requires_grad = False
    del pm, optim, sched
    gc.collect(); t.cuda.empty_cache()
    return row, rec_ood


results = {}
for rho in RHOS:
    results[rho], rec = train_one(rho)
    if rho == 1.0:
        print("\n================ 10 RAW OOD PREDICTIONS (rho=1.0 plain) ================")
        for i, r in enumerate(rec[:10]):
            print(f"\n--- ood[{i}] {r['profession']}/{r['category']} label={r['label']} ---")
            print(r["formatted"])
            print(f"  GOLD {r['gold_letter']!r}  ARGMAX {r['argmax_tok']!r}  "
                  f"logitA {r['logit_A']:.3f} logitB {r['logit_B']:.3f}  "
                  f"2way_score={r['score_2way']}")

print("\n================ GATE 1 VERDICT ================")
r10, r05 = results[1.0], results[0.5]
c1 = r10["id_accuracy"] >= 0.95
c2 = r10["ood_accuracy"] <= 0.20
c3 = r05["ood_accuracy"] >= 0.85
print(f"rho=1.0 ID  {r10['id_accuracy']:.4f} >= 0.95 : {'PASS' if c1 else 'FAIL'}")
print(f"rho=1.0 OOD {r10['ood_accuracy']:.4f} <= 0.20 : {'PASS' if c2 else 'FAIL'}")
print(f"rho=0.5 OOD {r05['ood_accuracy']:.4f} >= 0.85 : {'PASS' if c3 else 'FAIL'}")
print("GATE 1:", "PASS" if (c1 and c2 and c3) else "FAIL")
print("GATE1_SCRIPT_DONE")
