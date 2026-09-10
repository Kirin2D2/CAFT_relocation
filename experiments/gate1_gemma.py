"""Gate 1 on the screen pick: google/gemma-2-2b, FULL fine-tune, no ablation.
rho=1.0 and rho=0.5, seed 0. Recipe identical to gate1_fullft.py (CAFT's
gender config: lr 5e-6, warmup_ratio 0.5, wd 0.01, betas (0.9,0.95), linear,
3 epochs, batch 16, grad checkpointing + bnb AdamW8bit). Nothing tuned.

Differences from gate1_fullft.py, all mechanical:
  - MODEL_ID gemma-2-2b, attn_implementation="eager" (sdpa silently drops
    gemma-2's attention logit softcapping).
  - tok is loaded here if the kernel does not have it (fresh kernel).
  - BASE_OOD_REF is measured on the first fresh load rather than hard-coded;
    later reloads must match it within 0.02.
  - Weights saved for the rho=1.0 final only (disk quota); step-20 and the
    rho=0.5 final are not saved. Frontier rows still go to runs.jsonl.
  - Row carries "script_sha256" of this file, since git_hash is CAFT's
    upstream commit and does not track the harness.

Run:  %run -i experiments/gate1_gemma.py
"""
import os, sys, json, time, random, subprocess, collections, gc, hashlib
os.environ.setdefault("HF_HOME", "/workspace/hf")
import numpy as np
import torch as t

sys.path.insert(0, "/workspace/relocation/caft")
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import bitsandbytes as bnb
from spurious_correlations.datasets.gender import GenderDataset

MODEL_ID = "google/gemma-2-2b"
SEED = 0
RHOS = [1.0, 0.5]
EPOCHS, BS, LR, WARMUP_RATIO, WD = 3, 16, 5e-6, 0.5, 0.01
CKPT_EVERY = 10
BASE_OOD_REF = globals().get("BASE_OOD_REF_GEMMA")  # set on first fresh load
GIT = subprocess.run(["git", "-C", "/workspace/relocation/caft", "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
SCRIPT_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]

if "tok" not in globals() or getattr(tok, "name_or_path", "") != MODEL_ID:
    print(f"loading tokenizer {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

enc_A = tok.encode(" A", add_special_tokens=False)
enc_B = tok.encode(" B", add_special_tokens=False)
assert len(enc_A) == 1 and len(enc_B) == 1, (enc_A, enc_B)
ID_A, ID_B = enc_A[0], enc_B[0]
LAB = {" A": ID_A, " B": ID_B}
print(f"letter tokens: ' A'={ID_A} ' B'={ID_B}  script_sha={SCRIPT_SHA}")
CELLS = [("nominative", "doctor"), ("nominative", "nurse"),
         ("object", "doctor"), ("object", "nurse")]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    t.manual_seed(seed); t.cuda.manual_seed_all(seed)


@t.no_grad()
def eval_split(m, data, batch=32):
    was_training = m.training
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
    if was_training:
        m.train()
    n = len(records)
    met = {"acc_2way": sum(r["score_2way"] for r in records) / n,
           "acc_full": sum(r["correct_full"] for r in records) / n,
           "acc_flipped": sum(r["flipped_full"] for r in records) / n,
           "n_ties": sum(r["tie"] for r in records), "n": n,
           "argmax_in_AB": sum(r["argmax_tok"].strip() in ("A", "B") for r in records) / n,
           "cells": {f"{c}_{p}": round(
               sum(r["score_2way"] for r in records if r["category"] == c and r["profession"] == p)
               / max(1, sum(1 for r in records if r["category"] == c and r["profession"] == p)), 4)
               for c, p in CELLS}}
    return met, records


def fresh_base():
    global model
    try:
        del model
    except NameError:
        pass
    gc.collect(); t.cuda.empty_cache()
    print("reloading base model from disk...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda:0",
                                                 dtype=t.bfloat16,
                                                 attn_implementation="eager")
    model.eval()
    return model


def train_one(rho):
    global model, BASE_OOD_REF, BASE_OOD_REF_GEMMA
    run_id = f"gate1_gemma_rho{rho}_s{SEED}"
    print(f"\n########## {run_id} ##########")
    ds = GenderDataset(train_ambiguous_frac=rho, test_ambiguous_frac=0.0)

    m0, _ = eval_split(model, ds.test)
    if BASE_OOD_REF is None:
        BASE_OOD_REF = BASE_OOD_REF_GEMMA = m0["acc_2way"]
        print(f"base OOD reference set: {BASE_OOD_REF:.4f} (screen at batch 16 gave 0.5411)")
    else:
        print(f"pre-train base OOD check: {m0['acc_2way']:.4f} (ref {BASE_OOD_REF:.4f})")
        if abs(m0["acc_2way"] - BASE_OOD_REF) > 0.02:
            raise RuntimeError(f"base model not clean: OOD {m0['acc_2way']:.4f} != {BASE_OOD_REF:.4f}")

    set_seed(SEED)
    for p in model.parameters():
        p.requires_grad = True
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    idx = list(range(len(ds.train)))
    g = random.Random(SEED)
    steps_per_epoch = (len(idx) + BS - 1) // BS
    n_steps = steps_per_epoch * EPOCHS
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=WD,
                                betas=(0.9, 0.95))
    sched = get_scheduler("linear", optim,
                          num_warmup_steps=int(n_steps * WARMUP_RATIO),
                          num_training_steps=n_steps)

    ckpt_dir = f"/workspace/relocation/data/checkpoints/{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    frontier = []
    step = 0
    t0 = time.time()
    model.train()
    for ep in range(EPOCHS):
        g.shuffle(idx)
        for s in range(0, len(idx), BS):
            chunk = [ds.train[i] for i in idx[s:s + BS]]
            be = tok([c["formatted"] for c in chunk], padding=True, return_tensors="pt",
                     truncation=True, max_length=512).to(model.device)
            y = t.tensor([LAB[c["id"]] for c in chunk], device=model.device)
            logits = model(**be).logits[:, -1, :]
            loss = t.nn.functional.cross_entropy(logits.float(), y)
            loss.backward()
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            step += 1
            if step % CKPT_EVERY == 0 or step == n_steps:
                lval = float(loss.detach())
                mid, _ = eval_split(model, ds.val)
                mood, _ = eval_split(model, ds.test)
                fr = {"run_id": f"{run_id}_step{step:03d}", "frontier": True,
                      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "rho": rho, "arm": "plain", "seed": SEED,
                      "n_train": len(ds.train), "step": step,
                      "epoch": round(step / steps_per_epoch, 2), "train_loss": lval,
                      "id_accuracy": round(mid["acc_2way"], 4),
                      "ood_accuracy": round(mood["acc_2way"], 4),
                      "n_ties_id": mid["n_ties"], "n_ties_ood": mood["n_ties"],
                      "ood_argmax_in_AB": round(mood["argmax_in_AB"], 4),
                      "ood_cells": mood["cells"],
                      "direction_source": None, "ablation_type": None,
                      "layers": [], "git_hash": GIT, "script_sha256": SCRIPT_SHA,
                      "model_id": MODEL_ID}
                with open("/workspace/relocation/results/runs.jsonl", "a") as f:
                    f.write(json.dumps(fr) + "\n")
                frontier.append(fr)
                print(f"  step {step:3}/{n_steps} loss {lval:.4f} "
                      f"ID {mid['acc_2way']:.4f} OOD {mood['acc_2way']:.4f} "
                      f"ties(id/ood) {mid['n_ties']}/{mood['n_ties']} "
                      f"argmaxAB {mood['argmax_in_AB']:.2f} "
                      f"gpu {t.cuda.memory_allocated()/1e9:.1f}G")
    dt = time.time() - t0

    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    if rho == 1.0:
        model.save_pretrained(f"{ckpt_dir}/final")
    mid, rec_id = eval_split(model, ds.val)
    mood, rec_ood = eval_split(model, ds.test)

    row = {"run_id": run_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "rho": rho, "arm": "plain", "seed": SEED, "n_train": len(ds.train),
           "id_accuracy": round(mid["acc_2way"], 4),
           "ood_accuracy": round(mood["acc_2way"], 4),
           "id_accuracy_fullvocab": round(mid["acc_full"], 4),
           "ood_accuracy_fullvocab": round(mood["acc_full"], 4),
           "id_acc_flipped": round(mid["acc_flipped"], 4),
           "ood_acc_flipped": round(mood["acc_flipped"], 4),
           "n_ties_id": mid["n_ties"], "n_ties_ood": mood["n_ties"],
           "ood_argmax_in_AB": round(mood["argmax_in_AB"], 4),
           "ood_cells": mood["cells"],
           "base_ood_ref": round(BASE_OOD_REF, 4),
           "direction_source": None, "ablation_type": None, "layers": [],
           "git_hash": GIT, "script_sha256": SCRIPT_SHA, "model_id": MODEL_ID,
           "train": {"model_id": MODEL_ID, "method": "full_ft",
                     "lr": LR, "epochs": EPOCHS, "batch_size": BS,
                     "warmup_ratio": WARMUP_RATIO, "weight_decay": WD,
                     "optimizer": "bnb.AdamW8bit", "grad_checkpointing": True,
                     "attn_implementation": "eager",
                     "steps": n_steps, "wall_seconds": round(dt, 1)}}
    with open("/workspace/relocation/results/runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    with open(f"/workspace/relocation/data/{run_id}_ood_predictions.json", "w") as f:
        json.dump({"run": row, "frontier": frontier, "records": rec_ood}, f, indent=1)
    with open(f"{ckpt_dir}/frontier.json", "w") as f:
        json.dump(frontier, f, indent=1)

    print(f"\n== {run_id} FINAL ==")
    print(json.dumps({k: row[k] for k in
                      ["id_accuracy", "ood_accuracy", "ood_acc_flipped",
                       "n_ties_id", "n_ties_ood", "ood_argmax_in_AB", "ood_cells"]}, indent=1))
    del optim, sched
    gc.collect(); t.cuda.empty_cache()
    return row, rec_ood


for a in ("last_traceback", "last_value", "last_type"):
    setattr(sys, a, None)
gc.collect(); t.cuda.empty_cache()

results = {}
for rho in RHOS:
    fresh_base()
    results[rho], rec = train_one(rho)
    if rho == 1.0:
        print("\n================ 10 RAW OOD PREDICTIONS (rho=1.0, gemma-2-2b) ================")
        for j, r in enumerate(rec[:10]):
            print(f"\n--- ood[{j}] {r['profession']}/{r['category']} label={r['label']} ---")
            print(r["formatted"])
            print(f"  GOLD {r['gold_letter']!r}  ARGMAX {r['argmax_tok']!r}  "
                  f"logitA {r['logit_A']:.3f} logitB {r['logit_B']:.3f}  "
                  f"2way_score={r['score_2way']}")

fresh_base()

print("\n================ GATE 1 VERDICT (gemma-2-2b, full FT) ================")
r10, r05 = results[1.0], results[0.5]
c1 = r10["id_accuracy"] >= 0.95
c2 = r10["ood_accuracy"] <= 0.20
c3 = r05["ood_accuracy"] >= 0.85
print(f"rho=1.0 ID  {r10['id_accuracy']:.4f} >= 0.95 : {'PASS' if c1 else 'FAIL'}")
print(f"rho=1.0 OOD {r10['ood_accuracy']:.4f} <= 0.20 : {'PASS' if c2 else 'FAIL'}")
print(f"rho=0.5 OOD {r05['ood_accuracy']:.4f} >= 0.85 : {'PASS' if c3 else 'FAIL'}")
print("GATE 1 (gemma-2-2b):", "PASS" if (c1 and c2 and c3) else "FAIL")
print("GATE1_GEMMA_DONE")
