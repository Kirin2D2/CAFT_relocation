"""One run of the relocation matrix. Importable; no side effects.

train_arm(model, rho, seed, arm, dirs=None, dirs_file=None, save=False,
          complement_method="diffmeans", base_acts=None)
  arm in {"plain", "ablated", "random"}. Recipe identical to gate1_gemma.py.
  For ablated/random, a MeanAblator on `dirs` is enabled for the entire run
  (train + every eval). Writes: frontier rows every 10 steps, one final row
  (hooks-on and hooks-off evals), then the complement step on the SAME model
  in memory (directions_complement.py) and a second row "<arm>+complement"
  evaluated with training hooks (if any) + complement hooks. Returns the
  mutated model (caller should fresh_base() before the next run).
"""
import os, json, time, random, gc
import torch as t
from transformers import get_scheduler
import bitsandbytes as bnb
import harness as H
from ablation import MeanAblator
from directions_complement import complement_directions

EPOCHS, BS, LR, WARMUP_RATIO, WD = 3, 16, 5e-6, 0.5, 0.01
CKPT_EVERY = 10
SHA = {"script_sha256": H.sha_file(f"{H.ROOT}/experiments/train_arm.py"),
       "harness_sha256": H.sha_file(f"{H.ROOT}/experiments/harness.py"),
       "ablation_sha256": H.sha_file(f"{H.ROOT}/experiments/ablation.py")}


def _met_fields(mid, mood, suffix=""):
    return {f"id_accuracy{suffix}": round(mid["acc_2way"], 4),
            f"ood_accuracy{suffix}": round(mood["acc_2way"], 4),
            f"id_accuracy_fullvocab{suffix}": round(mid["acc_full"], 4),
            f"ood_accuracy_fullvocab{suffix}": round(mood["acc_full"], 4),
            f"id_acc_flipped{suffix}": round(mid["acc_flipped"], 4),
            f"ood_acc_flipped{suffix}": round(mood["acc_flipped"], 4),
            f"n_ties_id{suffix}": mid["n_ties"], f"n_ties_ood{suffix}": mood["n_ties"],
            f"ood_argmax_in_AB{suffix}": round(mood["argmax_in_AB"], 4),
            f"pick_A_rate_id{suffix}": round(mid["pick_A_rate"], 4),
            f"pick_A_rate_ood{suffix}": round(mood["pick_A_rate"], 4),
            f"ood_cells{suffix}": mood["cells"]}


def train_arm(model, rho, seed, arm, dirs=None, dirs_file=None, save=False,
              complement_method="diffmeans", base_acts=None, complement=True):
    assert arm == "plain" or dirs is not None, arm
    run_id = f"arm_{arm}_rho{rho}_s{seed}"
    print(f"\n########## {run_id} ##########")
    ds = H.dataset(rho)
    probe = H.items(H.dataset(0.5).train)

    m0, _ = H.eval_split(model, ds.test)
    print(f"pre-train base OOD check: {m0['acc_2way']:.4f} (ref {H.BASE_OOD_REF})")
    if abs(m0["acc_2way"] - H.BASE_OOD_REF) > 0.02:
        raise RuntimeError(f"base model not clean: OOD {m0['acc_2way']:.4f}")

    ablator = None
    if arm != "plain":
        assert dirs is not None
        ablator = MeanAblator(dirs, model, name=arm).enable()
        mb, _ = H.eval_split(model, ds.test)
        print(f"base OOD under {arm} ablation (hooks on, untrained): {mb['acc_2way']:.4f}")
    common = {"rho": rho, "seed": seed, "n_train": len(ds.train), "model_id": H.MODEL_ID,
              "direction_source": None if dirs is None else dirs["meta"]["source"],
              "ablation_type": None if arm == "plain" else "mean",
              "layers": [] if arm == "plain" else sorted(dirs["layers"]),
              "direction_rank": None if dirs is None else int(next(iter(dirs["layers"].values()))["U"].shape[1]),
              "directions_file": dirs_file, "directions_sha256": H.sha_file(dirs_file) if dirs_file else None,
              "git_hash": H.GIT, **SHA}

    H.set_seed(seed)
    for p in model.parameters():
        p.requires_grad = True
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    idx = list(range(len(ds.train)))
    g = random.Random(seed)
    steps_per_epoch = (len(idx) + BS - 1) // BS
    n_steps = steps_per_epoch * EPOCHS
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    sched = get_scheduler("linear", optim, num_warmup_steps=int(n_steps * WARMUP_RATIO),
                          num_training_steps=n_steps)
    frontier, step, t0 = [], 0, time.time()
    model.train()
    for ep in range(EPOCHS):
        g.shuffle(idx)
        for s in range(0, len(idx), BS):
            chunk = [ds.train[i] for i in idx[s:s + BS]]
            be = H.tok([c["formatted"] for c in chunk], padding=True, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
            y = t.tensor([H.LAB[c["id"]] for c in chunk], device=model.device)
            logits = model(**be).logits[:, -1, :]
            loss = t.nn.functional.cross_entropy(logits.float(), y)
            loss.backward()
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            step += 1
            if step % CKPT_EVERY == 0 or step == n_steps:
                lval = float(loss.detach())
                mid, _ = H.eval_split(model, ds.val)
                mood, _ = H.eval_split(model, ds.test)
                fr = {"run_id": f"{run_id}_step{step:03d}", "frontier": True, "timestamp": H.now(),
                      "arm": arm, "step": step, "epoch": round(step / steps_per_epoch, 2),
                      "train_loss": lval, "id_accuracy": round(mid["acc_2way"], 4),
                      "ood_accuracy": round(mood["acc_2way"], 4),
                      "n_ties_id": mid["n_ties"], "n_ties_ood": mood["n_ties"],
                      "ood_argmax_in_AB": round(mood["argmax_in_AB"], 4),
                      "pick_A_rate_ood": round(mood["pick_A_rate"], 4),
                      "ood_cells": mood["cells"], **common}
                H.append_run(fr); frontier.append(fr)
                print(f"  step {step:3}/{n_steps} loss {lval:.4f} ID {mid['acc_2way']:.4f} "
                      f"OOD {mood['acc_2way']:.4f} ties {mid['n_ties']}/{mood['n_ties']} "
                      f"pickA {mood['pick_A_rate']:.2f} gpu {t.cuda.memory_allocated()/1e9:.1f}G")
    dt = time.time() - t0
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    del optim, sched; gc.collect(); t.cuda.empty_cache()

    # ---- final evals: hooks as trained, then hooks off --------------------
    mid, rec_id = H.eval_split(model, ds.val)
    mood, rec_ood = H.eval_split(model, ds.test)
    row = {"run_id": run_id, "timestamp": H.now(), "arm": arm, "eval_hooks": "on" if ablator else "none",
           **_met_fields(mid, mood), "base_ood_ref": H.BASE_OOD_REF, **common,
           "train": {"model_id": H.MODEL_ID, "method": "full_ft", "lr": LR, "epochs": EPOCHS,
                     "batch_size": BS, "warmup_ratio": WARMUP_RATIO, "weight_decay": WD,
                     "optimizer": "bnb.AdamW8bit", "grad_checkpointing": True,
                     "attn_implementation": "eager", "steps": n_steps, "wall_seconds": round(dt, 1)}}
    if ablator:
        ablator.disable()
        mid_off, _ = H.eval_split(model, ds.val)
        mood_off, rec_off = H.eval_split(model, ds.test)
        row.update(_met_fields(mid_off, mood_off, "_hooks_off"))
        ablator.enable()
    H.append_run(row)
    with open(f"{H.ROOT}/data/{run_id}_ood_predictions.json", "w") as f:
        json.dump({"run": row, "frontier": frontier, "records": rec_ood}, f, indent=1)
    if save:
        d = f"{H.ROOT}/data/checkpoints/{run_id}/final"
        model.save_pretrained(d); print(f"saved weights -> {d}")
    print(f"== {run_id} FINAL ==  ID {row['id_accuracy']:.4f}  OOD {row['ood_accuracy']:.4f}  "
          f"cells {row['ood_cells']}" + (f"  | hooks off: ID {row['id_accuracy_hooks_off']:.4f} "
          f"OOD {row['ood_accuracy_hooks_off']:.4f}" if ablator else ""))

    if not complement:
        if ablator:
            ablator.disable()
        return model, row, None
    # ---- complement step, same model in memory ----------------------------
    # Headline row: diff-of-means in the complement. Second row: top PC of
    # (tuned - base) in the complement. Both logged; owner picks the headline.
    train_dirs = dirs if dirs else _zero_dirs(model)
    crows = {}
    for method, tag in (("diffmeans", "complement"), ("pc_diff", "complement_pc")):
        cdirs, cstats, _ = complement_directions(model, train_dirs, probe, method=method,
                                                 base_acts=base_acts)
        cfile = f"{H.ROOT}/data/directions/{tag}_{run_id}.pt"
        t.save(cdirs, cfile)
        for l in H.LAYERS:
            print(f"  {tag} L{l}: {cstats[l]}")
        comp = MeanAblator(cdirs, model, name=tag).enable()
        cmid, _ = H.eval_split(model, ds.val)
        cmood, crec = H.eval_split(model, ds.test)
        comp.disable()
        crow = {"run_id": f"{run_id}_{tag}", "timestamp": H.now(), "arm": f"{arm}+{tag}",
                "eval_hooks": (f"train+{tag}" if ablator else tag),
                **_met_fields(cmid, cmood), "base_ood_ref": H.BASE_OOD_REF, **common,
                "complement_source": cdirs["meta"]["source"], "complement_file": cfile,
                "complement_sha256": H.sha_file(cfile), "complement_stats": cstats,
                "parent_run_id": run_id}
        H.append_run(crow)
        with open(f"{H.ROOT}/data/{run_id}_{tag}_ood_predictions.json", "w") as f:
            json.dump({"run": crow, "records": crec}, f, indent=1)
        print(f"== {run_id}+{tag} ==  ID {crow['id_accuracy']:.4f}  OOD {crow['ood_accuracy']:.4f}  "
              f"cells {crow['ood_cells']}")
        crows[tag] = crow
    crow = crows["complement"]
    if ablator:
        ablator.disable()
    return model, row, crow


def _zero_dirs(model):
    """For the plain arm: an empty ablated subspace, so the complement is the whole space.
    Encoded as a zero-width U so U(U^T v) = 0."""
    d = model.config.hidden_size
    return {"layers": {l: {"U": t.zeros(d, 0), "mu": t.zeros(0)} for l in H.LAYERS},
            "meta": {"source": None}}
