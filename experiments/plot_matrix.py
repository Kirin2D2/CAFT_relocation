"""Figures from results/runs.jsonl only. PNGs to figures/.
1. ood_vs_rho.png   : OOD (and ID, faint) vs rho per arm; per-seed points + mean line.
2. complement.png   : OOD delta from test-time complement ablation, per arm and rho.
3. frontier_rho1.png: plain-run frontier (ID/OOD vs step) with ablated arms' finals as horizontal lines.
"""
import json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/workspace/relocation/results/runs.jsonl"
rows = [json.loads(l) for l in open(R) if l.strip()]
fin = [r for r in rows if not r.get("frontier") and r["run_id"].startswith("arm_")]
ARMS = ["plain", "abl_pca_tuned_L3_k16", "rnd_L3_k16", "abl_pca_base_L26_k1", "rnd_L26_k1"]
LABEL = {"plain": "plain fine-tune", "abl_pca_tuned_L3_k16": "ablated: 16 PCs x 3 layers (partial block)",
         "rnd_L3_k16": "random: 16 dirs x 3 layers", "abl_pca_base_L26_k1": "ablated: 1 PC x 26 layers (full block)",
         "rnd_L26_k1": "random: 1 dir x 26 layers"}
COL = {"plain": "#444444", "abl_pca_tuned_L3_k16": "#9467bd", "rnd_L3_k16": "#c5b0d5",
       "abl_pca_base_L26_k1": "#2ca02c", "rnd_L26_k1": "#98df8a", "ablated": "#1f77b4", "ablated_prof": "#8c564b", "random": "#ff7f0e"}

def pick(arm):
    return [r for r in fin if r["arm"] == arm]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for arm in ARMS:
    rs = pick(arm)
    if not rs: continue
    by = collections.defaultdict(list)
    for r in rs: by[r["rho"]].append(r)
    xs = sorted(by)
    for key, a in (("ood_accuracy", ax[0]), ("id_accuracy", ax[1])):
        for x in xs:
            a.scatter([x] * len(by[x]), [r[key] for r in by[x]], color=COL[arm], alpha=0.45, s=16)
        a.plot(xs, [sum(r[key] for r in by[x]) / len(by[x]) for x in xs], "-o", color=COL[arm], label=LABEL[arm], ms=4)
for a, ttl in zip(ax, ("OOD accuracy vs rho (projection on; 2 seeds + mean)", "ID accuracy vs rho")):
    a.set_xlabel("rho (fraction of training data where stereotype holds)"); a.set_title(ttl, fontsize=10)
    a.set_xticks([0.5, 0.8, 0.95, 1.0]); a.axhline(0.576, ls=":", c="gray", lw=0.7)
ax[0].set_ylim(-0.03, 1.03); ax[1].set_ylim(0.9, 1.01); ax[0].legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig("/workspace/relocation/figures/ood_vs_rho.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.2))
comp = [r for r in fin if r["arm"].endswith("+complement") and r["rho"] in (1.0, 0.95)]
parent = {r["run_id"]: r for r in fin}
xs_lab = []
for i, r in enumerate(sorted(comp, key=lambda r: (r["rho"], r["parent_run_id"]))):
    p = parent.get(r["parent_run_id"])
    if not p: continue
    base_arm = p["arm"]
    ax.bar(i, r["ood_accuracy"] - p["ood_accuracy"], color=COL.get(base_arm, "k"))
    xs_lab.append(f"{base_arm}\nrho{r['rho']} s{r['seed']}")
ax.set_xticks(range(len(xs_lab))); ax.set_xticklabels(xs_lab, fontsize=7)
ax.axhline(0, c="k", lw=0.8); ax.set_ylabel("OOD delta from test-time complement ablation")
ax.set_title("Complement ablation (diff-of-means in complement) effect on OOD")
fig.tight_layout(); fig.savefig("/workspace/relocation/figures/complement.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.2))
fr = [r for r in rows if r.get("frontier") and r["run_id"].startswith("arm_plain_rho1.0_s0")]
if fr:
    fr.sort(key=lambda r: r["step"])
    ax.plot([r["step"] for r in fr], [r["ood_accuracy"] for r in fr], "-o", c=COL["plain"], label="plain OOD (frontier)")
    ax.plot([r["step"] for r in fr], [r["id_accuracy"] for r in fr], "--", c=COL["plain"], alpha=0.5, label="plain ID (frontier)")
for arm in ("abl_pca_tuned_L3_k16", "abl_pca_base_L26_k1", "rnd_L26_k1"):
    for r in pick(arm):
        if r["rho"] == 1.0:
            ax.axhline(r["ood_accuracy"], c=COL[arm], lw=1, alpha=0.7, label=f"{arm} final OOD s{r['seed']}")
ax.set_xlabel("step"); ax.set_ylabel("accuracy"); ax.set_title("rho=1.0: undertrained-frontier baseline vs ablated finals")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("/workspace/relocation/figures/frontier_rho1.png", dpi=150)
print("wrote figures/ood_vs_rho.png complement.png frontier_rho1.png")


# 4. dose-response: OOD vs rank k. Left: rho=1.0 hooks on (26 layers x 3 sources + 3-layer tuned).
#    Middle: rho=1.0 hooks off. Right: rho=0.5 hooks on (grammar-learnability control).
import re
dose = [r for r in fin if re.match(r"arm_(pca_tuned|pca_base|randk)_L(26|3)_k\d+_rho", r["run_id"])
        and not r["arm"].endswith(("complement", "complement_pc"))]
if dose:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    series = (("pca_tuned_L26", "#1f77b4", "tuned PCs, 26 layers"), ("pca_base_L26", "#2ca02c", "base PCs, 26 layers"),
              ("randk_L26", "#ff7f0e", "random, 26 layers"), ("pca_tuned_L3", "#9467bd", "tuned PCs, layers 10/13/16"))
    panels = ((1.0, "ood_accuracy", "rho=1.0, OOD, projection on"), (1.0, "ood_accuracy_hooks_off", "rho=1.0, OOD, projection removed at test"),
              (0.5, "ood_accuracy", "rho=0.5 control, OOD, projection on"))
    for a, (rho, key, ttl) in zip(ax, panels):
        for pref, c, lab in series:
            rs = sorted([r for r in dose if r["rho"] == rho and r["arm"].startswith(pref + "_") and r.get(key) is not None],
                        key=lambda r: r["direction_rank"])
            if not rs: continue
            a.plot([r["direction_rank"] for r in rs], [r[key] for r in rs], "-o", c=c, label=lab)
        ref = [r for r in rows if r["run_id"] == f"gate1_gemma_rho{rho}_s0"]
        if ref: a.axhline(ref[0]["ood_accuracy"], c="#444444", ls="--", lw=1, label=f"plain fine-tune ({ref[0]['ood_accuracy']:.3f})")
        a.axhline(0.576, c="gray", lw=0.6, ls=":", label="always-A baseline" if a is ax[0] else None)
        a.set_xscale("log", base=2); a.set_xlabel("ablated rank k per layer"); a.set_title(ttl, fontsize=10); a.set_ylim(-0.03, 1.03)
    ax[0].set_ylabel("OOD accuracy (inverted pairing)"); ax[0].legend(fontsize=7, loc="center right"); fig.tight_layout()
    fig.savefig("/workspace/relocation/figures/dose_response.png", dpi=150); print("wrote figures/dose_response.png")


# 5. layer-position sweep: OOD by layer set, base PCs k=1, rho=1.0 (mean over seeds, points per seed)
lay = [r for r in fin if re.match(r"arm_pca_base_(early|mid|late|half|even|only|L\d)", r["run_id"]) and r["rho"] == 1.0]
if lay:
    order = ["only0", "only1", "only2", "only4", "only6", "only8", "only12", "only16", "only20", "only24",
             "L1-25", "L5-25",
             "early0-8", "mid9-17", "late18-25", "half0-12", "half13-25", "even"]
    by = collections.defaultdict(list)
    for r in lay:
        m_ = re.match(r"arm_pca_base_(.+?)_k(\d+)_", r["run_id"]); by[m_.group(1) + ("" if m_.group(2) == "1" else f" k{m_.group(2)}")].append(r["ood_accuracy"])
    names = [n for n in order if n in by] + sorted(n for n in by if n not in order)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, n in enumerate(names):
        ax.bar(i, sum(by[n]) / len(by[n]), color="#2ca02c" if n.startswith("only") else "#1f77b4", alpha=0.8)
        ax.scatter([i] * len(by[n]), by[n], c="k", s=12, zorder=3)
    ax.axhline(0.127, c="#444444", ls="--", lw=1, label="plain (0.127)"); ax.axhline(0.576, c="gray", ls=":", lw=0.7, label="always-A")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("only", "L") for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("OOD accuracy, rho=1.0"); ax.set_title("One base-model PC per layer, mean-ablated at these layers only (green = single layer)", fontsize=9)
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("/workspace/relocation/figures/layer_sweep.png", dpi=150); print("wrote figures/layer_sweep.png")
