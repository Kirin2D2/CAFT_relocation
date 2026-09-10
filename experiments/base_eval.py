"""Score the BASE model on a CAFT gender split. No training, no ablation.

Replicates CAFT's scorer exactly (trainer.py:133-154): full-vocab argmax at
the last position vs the gold letter token.

Split is chosen by setting SPLIT in the kernel globals before `%run -i`:
    SPLIT = "test"; %run -i experiments/base_eval.py
  "val"  -> train_ambiguous_frac=1.0 : correlation as in training (ID).
            Case and the gender heuristic AGREE on every item, so this split
            cannot distinguish a grammar-follower from a shortcut-user.
  "test" -> test_ambiguous_frac=0.0  : correlation INVERTED. Case still
            determines the correct answer, so a grammar-follower scores high
            and a pure shortcut-user scores ~0. This is the discriminator.
"""
import os, sys, json, collections, statistics
os.environ.setdefault("HF_HOME", "/workspace/hf")  # must precede transformers import
import torch as t

sys.path.insert(0, "/workspace/relocation/caft")
from transformers import AutoModelForCausalLM, AutoTokenizer
from spurious_correlations.datasets.gender import GenderDataset

MODEL_ID = "Qwen/Qwen3-4B-Base"
BATCH = 16
SPLIT = globals().get("SPLIT", "val")
assert SPLIT in ("val", "test"), SPLIT

# ---- load once into the live kernel namespace -------------------------------
if "tok" not in globals():
    print(f"loading tokenizer {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

if "model" not in globals():
    print(f"loading model {MODEL_ID} (~2 min)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda:0", dtype=t.bfloat16
    )
    model.eval()

print("python:", sys.executable)
print("torch:", t.__version__, "| cuda:", t.cuda.is_available())
print("model class:", type(model).__name__, "| device:", next(model.parameters()).device)

# ---- label tokens (CAFT takes the FIRST token of " A" / " B") ---------------
enc_A = tok.encode(" A", add_special_tokens=False)
enc_B = tok.encode(" B", add_special_tokens=False)
ID_A, ID_B = enc_A[0], enc_B[0]
LAB = {" A": ID_A, " B": ID_B}
print(f"encode(' A')={enc_A} encode(' B')={enc_B}")

ds = GenderDataset()  # train_ambiguous_frac=1.0, test_ambiguous_frac=0.0
data = getattr(ds, SPLIT)
print(f"SPLIT={SPLIT}  n={len(data)}")

# Show the profession<->gender pairing actually present, so the inversion is visible.
pair = collections.Counter((x["profession"], x["label"]) for x in data)
print("(profession,label) counts:", dict(sorted(pair.items())))

# ---- score ------------------------------------------------------------------
records = []
with t.no_grad():
    for s in range(0, len(data), BATCH):
        chunk = [data[i] for i in range(s, min(s + BATCH, len(data)))]
        be = tok([c["formatted"] for c in chunk], padding=True,
                 return_tensors="pt", truncation=True, max_length=512).to(model.device)
        logits = model(**be).logits[:, -1, :].float()
        arg = logits.argmax(dim=-1)
        for j, c in enumerate(chunk):
            gold = LAB[c["id"]]
            lg = logits[j]
            records.append({
                "formatted": c["formatted"],
                "gold_letter": c["id"], "label": c["label"],
                "profession": c["profession"], "category": c["category"],
                "argmax_id": int(arg[j]), "argmax_tok": tok.decode([int(arg[j])]),
                "correct_fullvocab": bool(int(arg[j]) == gold),
                "logit_A": float(lg[ID_A]), "logit_B": float(lg[ID_B]),
                "twoway_pred": " A" if lg[ID_A] > lg[ID_B] else " B",
            })

n = len(records)
def frac(pred, sub=None):
    rs = records if sub is None else sub
    return sum(pred(r) for r in rs) / len(rs) if rs else float("nan")

acc_full = frac(lambda r: r["correct_fullvocab"])
acc_2way = frac(lambda r: r["twoway_pred"] == r["gold_letter"])
in_set   = frac(lambda r: r["argmax_tok"].strip() in ("A", "B"))
# acc_flipped = accuracy against the deliberately wrong option (CAFT logs this).
# NOTE: when in_set == 1.0 this is exactly 1 - acc_full, not independent info.
acc_flip = frac(lambda r: r["argmax_tok"] != r["gold_letter"] and r["argmax_tok"].strip() in ("A","B"))

def by(key, vals):
    return {v: (round(frac(lambda r: r["correct_fullvocab"],
                           [r for r in records if r[key] == v]), 4),
                sum(1 for r in records if r[key] == v)) for v in vals}

print(f"\n================ BASE MODEL, {SPLIT}, NO TRAINING ================")
print(f"n                                  = {n}")
print(f"accuracy (CAFT full-vocab argmax)  = {acc_full:.4f}")
print(f"accuracy (restricted A-vs-B logit) = {acc_2way:.4f}")
print(f"fraction argmax lands on A or B    = {in_set:.4f}")
print(f"acc_flipped (shortcut-consistent)  = {acc_flip:.4f}")
print("argmax histogram:", collections.Counter(r["argmax_tok"] for r in records).most_common(5))
print("acc by category   (acc, n):", by("category", ["nominative", "object"]))
print("acc by profession (acc, n):", by("profession", ["doctor", "nurse"]))
print("acc by gold letter(acc, n):", by("gold_letter", [" A", " B"]))
m = [abs(r["logit_A"] - r["logit_B"]) for r in records]
print("|logit A - logit B|: median %.2f  min %.2f  max %.2f" % (statistics.median(m), min(m), max(m)))

print(f"\n================ 10 RAW PREDICTIONS ({SPLIT}) ================")
for i, r in enumerate(records[:10]):
    print(f"\n--- {SPLIT}[{i}] profession={r['profession']} category={r['category']} label={r['label']} ---")
    print(r["formatted"])
    print(f"  GOLD letter  : {r['gold_letter']!r}")
    print(f"  ARGMAX token : {r['argmax_tok']!r} (id={r['argmax_id']})")
    print(f"  logit ' A'={r['logit_A']:.3f}  logit ' B'={r['logit_B']:.3f} -> 2way {r['twoway_pred']!r}")
    print(f"  correct(full-vocab)={r['correct_fullvocab']}")

out = f"/workspace/relocation/data/base_eval_{SPLIT}_predictions.json"
with open(out, "w") as f:
    json.dump({"model_id": MODEL_ID, "split": SPLIT, "n": n,
               "acc_full": acc_full, "acc_2way": acc_2way,
               "frac_argmax_in_AB": in_set, "acc_flipped": acc_flip,
               "records": records}, f, indent=1)
print(f"\nwrote {out}")
print("NOTE: not logged to runs.jsonl - untrained sanity check, not a run.")
