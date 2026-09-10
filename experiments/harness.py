"""Shared pieces for the gemma-2-2b relocation matrix. No side effects on import
beyond loading the tokenizer. eval_split / set_seed / fresh_base are copied from
gate1_gemma.py (which trains on import, so it cannot be imported).
"""
import os, sys, json, time, random, subprocess, hashlib, gc
os.environ.setdefault("HF_HOME", "/workspace/hf")
import numpy as np
import torch as t
sys.path.insert(0, "/workspace/relocation/caft")
from transformers import AutoModelForCausalLM, AutoTokenizer
from spurious_correlations.datasets.gender import GenderDataset

ROOT = "/workspace/relocation"
MODEL_ID = "google/gemma-2-2b"
LAYERS = [10, 13, 16]
BASE_OOD_REF = 0.5411          # gate1_gemma base reference, batch 32
RUNS = f"{ROOT}/results/runs.jsonl"
CELLS = [("nominative", "doctor"), ("nominative", "nurse"),
         ("object", "doctor"), ("object", "nurse")]
GIT = subprocess.run(["git", "-C", f"{ROOT}/caft", "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip()

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
_eA, _eB = tok.encode(" A", add_special_tokens=False), tok.encode(" B", add_special_tokens=False)
assert len(_eA) == 1 and len(_eB) == 1, (_eA, _eB)
ID_A, ID_B = _eA[0], _eB[0]
LAB = {" A": ID_A, " B": ID_B}


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    t.manual_seed(seed); t.cuda.manual_seed_all(seed)


def fresh_base(old=None):
    if old is not None:
        del old
    gc.collect(); t.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda:0",
                                             dtype=t.bfloat16, attn_implementation="eager")
    m.eval()
    return m


def dataset(rho):
    return GenderDataset(train_ambiguous_frac=rho, test_ambiguous_frac=0.0)


def items(data):
    return [data[i] for i in range(len(data))]


def is_male(item):
    return item["label"] in ("he", "him")


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
           "pick_A_rate": sum(r["logit_A"] > r["logit_B"] for r in records) / n,
           "cells": {f"{c}_{p}": round(
               sum(r["score_2way"] for r in records if r["category"] == c and r["profession"] == p)
               / max(1, sum(1 for r in records if r["category"] == c and r["profession"] == p)), 4)
               for c, p in CELLS}}
    return met, records


@t.no_grad()
def collect_final_token(m, data, layers, batch=32):
    """Residual stream (decoder-layer output) at the final token, per layer.
    Fires AFTER any MeanAblator hook already registered on the same layer, so
    on an ablated model this returns the intervened activations."""
    was_training = m.training
    m.eval()
    store = {l: [] for l in layers}
    handles = []
    for l in layers:
        def mk(l):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                store[l].append(h[:, -1, :].float().cpu())
            return hook
        handles.append(m.model.layers[l].register_forward_hook(mk(l)))
    try:
        for s in range(0, len(data), batch):
            chunk = data[s:s + batch]
            be = tok([c["formatted"] for c in chunk], padding=True, return_tensors="pt",
                     truncation=True, max_length=512).to(m.device)
            m(**be)
    finally:
        for h in handles:
            h.remove()
    if was_training:
        m.train()
    return {l: t.cat(v, 0) for l, v in store.items()}


def append_run(row):
    with open(RUNS, "a") as f:
        f.write(json.dumps(row) + "\n")


def logged_run_ids():
    if not os.path.exists(RUNS):
        return set()
    return {json.loads(l)["run_id"] for l in open(RUNS) if l.strip()}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
