"""Reprint every matrix number from results/runs.jsonl. No other input.
Usage: python experiments/table_matrix.py [--frontier]
"""
import json, sys, collections
rows = [json.loads(l) for l in open("/workspace/relocation/results/runs.jsonl") if l.strip()]
fin = [r for r in rows if not r.get("frontier") and r["run_id"].startswith(("arm_", "gate1_gemma"))]
print(f"{'run_id':<44} {'arm':<24} {'rho':>4} {'s':>1} {'ID':>7} {'OOD':>7} {'OOD off':>7} {'pickA':>5}  cells")
for r in sorted(fin, key=lambda r: (r["rho"], r.get("arm", "plain"), r["seed"], r["run_id"])):
    off = r.get("ood_accuracy_hooks_off"); pa = r.get("pick_A_rate_ood")
    print(f"{r['run_id']:<44} {r.get('arm','plain'):<24} {r['rho']:>4} {r['seed']:>1} "
          f"{r['id_accuracy']:>7.4f} {r['ood_accuracy']:>7.4f} {(f'{off:.4f}' if off is not None else '-'):>7} "
          f"{(f'{pa:.2f}' if pa is not None else '-'):>5}  {list(r['ood_cells'].values())}")
if "--frontier" in sys.argv:
    fr = collections.defaultdict(list)
    for r in rows:
        if r.get("frontier") and r["run_id"].startswith(("arm_", "gate1_gemma")):
            fr[r["run_id"].rsplit("_step", 1)[0]].append((r["step"], r["id_accuracy"], r["ood_accuracy"]))
    for k, v in sorted(fr.items()):
        print(k, " ".join(f"{s}:{i:.2f}/{o:.2f}" for s, i, o in sorted(v)))
