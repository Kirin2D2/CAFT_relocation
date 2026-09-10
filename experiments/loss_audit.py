"""Audit the rho=1.0 full-FT training loss, end to end.

A. Pipeline sanity: recompute the training CE on the base model by hand for
   the exact first training batch (seed 0 shuffle) — must match the observed
   step-10-region loss scale (~0.39), and the target indices must be the
   gold letter tokens.
B. Fit check: final checkpoint's CE + accuracy + p(gold) over the FULL train
   set (the printed 0.0007 was a single batch).
C. Discriminator: same TRAIN sentences with profession swapped
   (doctor<->nurse), options and gold letter unchanged. Grammar unchanged,
   correlation inverted. If the model fit train via grammar, accuracy stays
   high; if via the shortcut, it collapses.
D. Margin growth: gold-vs-other logit margins, base vs final, train set.
"""
import os, sys, json, random, gc, statistics
os.environ.setdefault("HF_HOME", "/workspace/hf")
import torch as t
sys.path.insert(0, "/workspace/relocation/caft")
from transformers import AutoModelForCausalLM
from spurious_correlations.datasets.gender import GenderDataset

assert "model" in globals() and "tok" in globals()
ID_A = tok.encode(" A", add_special_tokens=False)[0]
ID_B = tok.encode(" B", add_special_tokens=False)[0]
LAB = {" A": ID_A, " B": ID_B}
BS = 16

ds = GenderDataset(train_ambiguous_frac=1.0, test_ambiguous_frac=0.0)

# ---- A: recompute first training batch CE on the BASE model -----------------
idx = list(range(len(ds.train)))
random.Random(0).shuffle(idx)                 # same shuffle as training epoch 0
chunk = [ds.train[i] for i in idx[:BS]]
be = tok([c["formatted"] for c in chunk], padding=True, return_tensors="pt",
         truncation=True, max_length=512).to(model.device)
y = t.tensor([LAB[c["id"]] for c in chunk], device=model.device)
with t.no_grad():
    logits = model(**be).logits[:, -1, :]
ce = t.nn.functional.cross_entropy(logits.float(), y)
print("A. base-model CE on first train batch:", round(float(ce), 4))
print("   targets are letter tokens:", sorted(set(y.tolist())), "== [362, 425]?")
print("   per-item gold letters:", [c["id"] for c in chunk])
print("   train batch correlation:", sorted(set((c["profession"], c["label"]) for c in chunk)))

def scores(m, items, batch=32):
    out = []
    for s in range(0, len(items), batch):
        ch = items[s:s + batch]
        enc = tok([c["formatted"] for c in ch], padding=True, return_tensors="pt",
                  truncation=True, max_length=512).to(m.device)
        with t.no_grad():
            lg = m(**enc).logits[:, -1, :].float()
        for j, c in enumerate(ch):
            la, lb = float(lg[j, ID_A]), float(lg[j, ID_B])
            g, o = (la, lb) if c["id"] == " A" else (lb, la)
            yy = t.tensor([LAB[c["id"]]])
            out.append({"margin": g - o, "acc": 1.0 if g > o else (0.5 if g == o else 0.0),
                        "ce": float(t.nn.functional.cross_entropy(lg[j:j+1].cpu(), yy)),
                        "pgold": float(t.softmax(lg[j].cpu(), -1)[LAB[c["id"]]])})
    return out

train_items = [ds.train[i] for i in range(len(ds.train))]
base_train = scores(model, train_items)
print(f"\nD. base  train: acc {sum(r['acc'] for r in base_train)/len(base_train):.4f} "
      f"CE {statistics.mean(r['ce'] for r in base_train):.4f} "
      f"median margin {statistics.median(r['margin'] for r in base_train):.2f}")

# ---- load final rho=1.0 checkpoint ------------------------------------------
print("\nloading final rho=1.0 checkpoint...")
ft = AutoModelForCausalLM.from_pretrained(
    "/workspace/relocation/data/checkpoints/gate1_full_rho1.0_s0/final",
    device_map="cuda:0", dtype=t.bfloat16)
ft.eval()

ft_train = scores(ft, train_items)
print(f"B. final train: acc {sum(r['acc'] for r in ft_train)/len(ft_train):.4f} "
      f"CE {statistics.mean(r['ce'] for r in ft_train):.4f} "
      f"mean p(gold) {statistics.mean(r['pgold'] for r in ft_train):.4f} "
      f"min p(gold) {min(r['pgold'] for r in ft_train):.4f}")
print(f"D. final train median margin {statistics.median(r['margin'] for r in ft_train):.2f} "
      f"(base was {statistics.median(r['margin'] for r in base_train):.2f})")

# ---- C: profession-swapped TRAIN sentences ----------------------------------
def swap_prof(c):
    q = c["formatted"]
    if c["profession"] == "doctor":
        q2, p2 = q.replace("doctor", "nurse"), "nurse"
    else:
        q2, p2 = q.replace("nurse", "doctor"), "doctor"
    assert q2 != q
    d = dict(c); d["formatted"] = q2; d["profession"] = p2
    return d

flipped = [swap_prof(c) for c in train_items]
ft_flip = scores(ft, flipped)
base_flip = scores(model, flipped)
print(f"\nC. TRAIN sentences, profession swapped (correlation inverted, grammar identical):")
print(f"   base  acc {sum(r['acc'] for r in base_flip)/len(base_flip):.4f}")
print(f"   final acc {sum(r['acc'] for r in ft_flip)/len(ft_flip):.4f}   "
      f"median margin {statistics.median(r['margin'] for r in ft_flip):.2f}")
by = {}
for c, r in zip(flipped, ft_flip):
    by.setdefault((c["category"], c["profession"]), []).append(r["acc"])
print("   final per-cell:", {f"{k[0]}_{k[1]}": round(sum(v)/len(v), 4) for k, v in sorted(by.items())})

del ft; gc.collect(); t.cuda.empty_cache()
print("\nLOSS_AUDIT_DONE")
