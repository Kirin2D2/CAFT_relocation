"""Base-competence screen across candidate models. Forward pass only, no training.

Pre-committed selection rule: lowest base accuracy on the rho=1.0 TRAIN set,
subject to base OOD accuracy >= 0.50 on the fixed test split, and (hard gate,
added after OLMo-2-1B, owner-approved 2026-09-10) full-vocab argmax on A/B
>= 0.90 on both splits -- otherwise the accuracies are noise between two
tokens the model never emits. Did not change this screen's outcome.

Accuracy is the same computation as loss_audit.py `scores()`: restricted A-vs-B
logit comparison at the last position, exact ties scored 0.5. That is what
produced 0.946 on Qwen3-4B-Base, so the screen numbers are comparable to it.

Also logged per model, as diagnostics (reported, NOT used to filter):
  frac_argmax_AB     full-vocab argmax lands on A or B. If low, the model does
                     not get the prompt format and its accuracies are
                     uninterpretable -- a low train acc is NOT a screen win.
  train_swapped_acc  profession-swapped train sentences (loss_audit part C):
                     grammar unchanged, correlation inverted. High = base solves
                     train via grammar, which is the Qwen3-4B failure mode.

Runs standalone, not in the kernel: loads and frees each model in turn.
Rows go to results/screen.jsonl (one line per model, written as each finishes)
and raw per-item predictions to data/screen_<slug>_predictions.json.
Untrained sanity check, so it follows base_eval.py's precedent and does not
write to runs.jsonl.

Usage:
  source venv/bin/activate
  HF_HOME=/workspace/hf python experiments/screen_base.py [MODEL_ID ...]
With no args, runs the full pre-committed list. Gated models that fail to load
are logged as status=failed and skipped; the screen continues.
"""
import os, sys, json, gc, time, statistics, subprocess, datetime, collections
os.environ.setdefault("HF_HOME", "/workspace/hf")
import torch as t
sys.path.insert(0, "/workspace/relocation/caft")
from transformers import AutoModelForCausalLM, AutoTokenizer
from spurious_correlations.datasets.gender import GenderDataset

# Ungated first so their rows are saved before any gated load can fail.
CANDIDATES = [
    "Qwen/Qwen3-1.7B-Base",
    "allenai/OLMo-2-0425-1B",
    "google/gemma-2-2b",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
]
MODELS = sys.argv[1:] or CANDIDATES
BATCH = 16
OUT_JSONL = "/workspace/relocation/results/screen.jsonl"
CAFT_HASH = subprocess.run(["git", "-C", "/workspace/relocation/caft", "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

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

swapped_items = [swap_prof(c) for c in train_items]
print(f"n_train={len(train_items)} n_ood={len(ood_items)} n_swapped={len(swapped_items)}")
print("train (profession,label):", dict(sorted(collections.Counter(
    (x["profession"], x["label"]) for x in train_items).items())))
print("ood   (profession,label):", dict(sorted(collections.Counter(
    (x["profession"], x["label"]) for x in ood_items).items())))


def score_split(model, tok, items, ID_A, ID_B):
    LAB = {" A": ID_A, " B": ID_B}
    recs = []
    with t.no_grad():
        for s in range(0, len(items), BATCH):
            ch = items[s:s + BATCH]
            enc = tok([c["formatted"] for c in ch], padding=True, return_tensors="pt",
                      truncation=True, max_length=512).to(model.device)
            lg = model(**enc).logits[:, -1, :].float()
            arg = lg.argmax(dim=-1)
            for j, c in enumerate(ch):
                la, lb = float(lg[j, ID_A]), float(lg[j, ID_B])
                g, o = (la, lb) if c["id"] == " A" else (lb, la)
                a = int(arg[j])
                recs.append({
                    "formatted": c["formatted"], "gold_letter": c["id"],
                    "label": c["label"], "profession": c["profession"],
                    "category": c["category"],
                    "logit_A": la, "logit_B": lb, "margin": g - o,
                    "acc": 1.0 if g > o else (0.5 if g == o else 0.0),
                    "argmax_id": a, "argmax_tok": tok.decode([a]),
                    "correct_fullvocab": a == LAB[c["id"]],
                })
    return recs


def summarize(recs):
    n = len(recs)
    out = {
        "n": n,
        "acc": sum(r["acc"] for r in recs) / n,
        "acc_fullvocab": sum(r["correct_fullvocab"] for r in recs) / n,
        "frac_argmax_AB": sum(r["argmax_tok"].strip() in ("A", "B") for r in recs) / n,
        "n_ties": sum(r["acc"] == 0.5 for r in recs),
        "median_margin": statistics.median(r["margin"] for r in recs),
        "argmax_top5": collections.Counter(r["argmax_tok"] for r in recs).most_common(5),
    }
    cells = {}
    for r in recs:
        cells.setdefault(f"{r['category']}_{r['profession']}", []).append(r["acc"])
    out["cells"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(cells.items())}
    return out


def run_one(model_id):
    slug = model_id.replace("/", "__")
    t0 = time.time()
    row = {"screen_id": "base_screen", "model_id": model_id,
           "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "caft_hash": CAFT_HASH}
    print(f"\n{'=' * 70}\n{model_id}\n{'=' * 70}")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = {}
        # Gemma-2 uses attention logit softcapping, which sdpa drops silently.
        if "gemma-2" in model_id:
            kw["attn_implementation"] = "eager"
        t.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda:0", dtype=t.bfloat16, **kw)
        model.eval()
    except Exception as e:
        row.update({"status": "failed", "stage": "load", "error": f"{type(e).__name__}: {e}"[:500],
                    "wall_seconds": round(time.time() - t0, 1)})
        print(f"LOAD FAILED: {row['error']}")
        with open(OUT_JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    enc_A = tok.encode(" A", add_special_tokens=False)
    enc_B = tok.encode(" B", add_special_tokens=False)
    ID_A, ID_B = enc_A[0], enc_B[0]
    print(f"encode(' A')={enc_A} encode(' B')={enc_B}  (using first token, as CAFT does)")
    if len(enc_A) != 1 or len(enc_B) != 1:
        print("WARNING: letter is not a single token for this tokenizer")
    row["letter_tokens"] = {" A": enc_A, " B": enc_B}
    row["n_params"] = sum(p.numel() for p in model.parameters())
    row["bos_added_by_default"] = bool(tok("x")["input_ids"][:1] == [tok.bos_token_id]) if tok.bos_token_id is not None else False

    try:
        tr = score_split(model, tok, train_items, ID_A, ID_B)
        oo = score_split(model, tok, ood_items, ID_A, ID_B)
        sw = score_split(model, tok, swapped_items, ID_A, ID_B)
    except Exception as e:
        row.update({"status": "failed", "stage": "score", "error": f"{type(e).__name__}: {e}"[:500],
                    "wall_seconds": round(time.time() - t0, 1)})
        print(f"SCORE FAILED: {row['error']}")
        with open(OUT_JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")
        del model; gc.collect(); t.cuda.empty_cache()
        return row

    S_tr, S_oo, S_sw = summarize(tr), summarize(oo), summarize(sw)
    row.update({
        "status": "ok",
        "n_train": S_tr["n"], "n_ood": S_oo["n"],
        "train_acc": S_tr["acc"], "train_acc_fullvocab": S_tr["acc_fullvocab"],
        "train_frac_argmax_AB": S_tr["frac_argmax_AB"], "train_n_ties": S_tr["n_ties"],
        "train_median_margin": S_tr["median_margin"], "train_cells": S_tr["cells"],
        "ood_acc": S_oo["acc"], "ood_acc_fullvocab": S_oo["acc_fullvocab"],
        "ood_frac_argmax_AB": S_oo["frac_argmax_AB"], "ood_n_ties": S_oo["n_ties"],
        "ood_median_margin": S_oo["median_margin"], "ood_cells": S_oo["cells"],
        "train_swapped_acc": S_sw["acc"], "train_swapped_cells": S_sw["cells"],
        "peak_mem_gb": round(t.cuda.max_memory_allocated() / 2**30, 2),
        "wall_seconds": round(time.time() - t0, 1),
    })

    print(f"train        acc={S_tr['acc']:.4f}  fullvocab={S_tr['acc_fullvocab']:.4f}  "
          f"argmax_in_AB={S_tr['frac_argmax_AB']:.4f}  ties={S_tr['n_ties']}  "
          f"med_margin={S_tr['median_margin']:.2f}")
    print(f"ood          acc={S_oo['acc']:.4f}  fullvocab={S_oo['acc_fullvocab']:.4f}  "
          f"argmax_in_AB={S_oo['frac_argmax_AB']:.4f}  ties={S_oo['n_ties']}  "
          f"med_margin={S_oo['median_margin']:.2f}")
    print(f"train_swapped acc={S_sw['acc']:.4f}   (grammar kept, correlation inverted)")
    print(f"ood cells: {S_oo['cells']}")
    print(f"train argmax top5: {S_tr['argmax_top5']}")
    print(f"ood   argmax top5: {S_oo['argmax_top5']}")
    print(f"peak mem {row['peak_mem_gb']} GB | {row['wall_seconds']}s")

    print("\n--- 3 raw OOD predictions ---")
    for r in oo[:3]:
        print(r["formatted"])
        print(f"  GOLD {r['gold_letter']!r} | argmax {r['argmax_tok']!r} | "
              f"logit A={r['logit_A']:.3f} B={r['logit_B']:.3f} | acc={r['acc']}")

    with open(f"/workspace/relocation/data/screen_{slug}_predictions.json", "w") as f:
        json.dump({"model_id": model_id, "train": tr, "ood": oo, "train_swapped": sw}, f, indent=1)
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")

    del model, tok; gc.collect(); t.cuda.empty_cache()
    return row


rows = [run_one(m) for m in MODELS]

print(f"\n{'=' * 70}\nSCREEN SUMMARY  (rule: min train_acc s.t. ood_acc >= 0.50)\n{'=' * 70}")
print(f"{'model':<28} {'train':>7} {'ood':>7} {'swap':>7} {'argAB':>6}  note")
ok = [r for r in rows if r["status"] == "ok"]
for r in sorted(ok, key=lambda r: r["train_acc"]):
    notes = []
    if r["ood_acc"] < 0.50:
        notes.append("FAILS ood>=0.50")
    if min(r["train_frac_argmax_AB"], r["ood_frac_argmax_AB"]) < 0.90:
        notes.append("FORMAT-SUSPECT: argmax not on A/B")
    print(f"{r['model_id']:<28} {r['train_acc']:>7.4f} {r['ood_acc']:>7.4f} "
          f"{r['train_swapped_acc']:>7.4f} {min(r['train_frac_argmax_AB'], r['ood_frac_argmax_AB']):>6.2f}  "
          f"{'; '.join(notes)}")
for r in rows:
    if r["status"] != "ok":
        print(f"{r['model_id']:<28} FAILED at {r['stage']}: {r['error'][:80]}")
elig = [r for r in ok if r["ood_acc"] >= 0.50
        and min(r["train_frac_argmax_AB"], r["ood_frac_argmax_AB"]) >= 0.90]
if elig:
    pick = min(elig, key=lambda r: r["train_acc"])
    print(f"\nPICK by rule: {pick['model_id']}  train_acc={pick['train_acc']:.4f}  ood_acc={pick['ood_acc']:.4f}")
    print("Reference: Qwen3-4B-Base was train 0.946 / ood 0.693.")
else:
    print("\nNO CANDIDATE satisfies ood_acc >= 0.50.")
print(f"\nrows appended to {OUT_JSONL}")
print("SCREEN_DONE")
