# Project: Does a spurious concept relocate under training-time ablation?

## What this project is

We fine-tune a language model on a task with a controllable spurious correlation,
while projecting out the direction(s) encoding the spurious concept on every
forward pass. The question is whether the concept **relocates** into the
orthogonal complement of the ablated subspace, and whether that relocation
depends on how strongly the training data rewards using the concept.

The headline measurement is **causal, not correlational**: after fine-tuning,
recompute directions from base-vs-tuned activation differences, restrict to the
orthogonal complement of what we ablated, ablate those at *test time*, and see
whether OOD accuracy moves while in-distribution accuracy survives.

Do not substitute "can a probe decode the concept" for this. Gender is trivially
decodable from pronouns in the input; decodability is not use.

---

## Environment

- Working dir: `/workspace/relocation`. Everything lives here.
- Python venv at `/workspace/relocation/venv` — **always** `source venv/bin/activate`
  before running anything. Packages installed outside `/workspace` are wiped on
  pod restart.
- HuggingFace cache is at `/workspace/hf` via `HF_HOME`. Never let a download go
  to `~/.cache` — it will not survive a restart and re-downloading costs minutes.
- Live IPython kernel runs in tmux session `exp`.

### Talking to the kernel

Do not paste multi-line code into the kernel. Write a file to `experiments/`,
then run it in the existing namespace:

```bash
tmux send-keys -t exp '%run -i experiments/NAME.py' Enter
sleep <appropriate>
tmux capture-pane -t exp -p -S -200
```

`%run -i` runs in the existing globals, so `model` and `tok` stay loaded.

### Kernel rules

- **Never restart the kernel or reload the model without asking.** Model load is
  ~2 minutes and the kernel holds state we care about.
- `model` and `tok` are globals. Assume they exist; do not re-instantiate them.
- If a reload is genuinely necessary, `del model; gc.collect();
  torch.cuda.empty_cache()` first. Stacked model copies have already caused a
  near-OOM on this pod once.
- Save every plot to `figures/` as a PNG. Never rely on inline display.
- Checkpoint datasets and cached activations to `data/` before any long op.

---

## Logging — non-negotiable

Every training or eval run appends exactly one JSON line to `results/runs.jsonl`:

```json
{"run_id": "", "timestamp": "", "rho": 1.0, "arm": "plain",
 "seed": 0, "n_train": 0, "id_accuracy": 0.0, "ood_accuracy": 0.0,
 "direction_source": "diff_of_means", "ablation_type": "mean",
 "layers": [], "git_hash": ""}
```

**Never report a number in prose that is not in `runs.jsonl`.** Every figure must
be regenerable from that file alone. I will be recomputing headline numbers with
fresh one-liners to check them.

---

## Scope

**You own:** dataset plumbing, training loops, eval harness, batching, plotting,
debugging, environment repair.

**I own:** experimental design, choice of arms and controls, direction-finding
code, the eval metric, and interpretation of results.

Do not add arms, change metrics, drop conditions, or "improve" the experimental
design without asking. If you think a design decision is wrong, say so and stop —
do not route around it.

---

## Reporting

- If an experiment appears to work, report it as a **hypothesis**, and state
  explicitly what you did *not* check.
- Volunteer the dumbest way each result could be wrong: data leakage, a trivial
  baseline matching it, the metric not measuring what we think, a scorer failing
  silently on verbose output.
- When reporting accuracy, always include raw model completions alongside the
  number so the scorer can be verified.
- Never fabricate or interpolate a number. If a run failed, say it failed.

---

## Experimental design (for context — do not change without asking)

- **Model:** Qwen 3.5 4B. Verify the exact HF model id rather than guessing.
- **Task:** CAFT gender-bias spurious correlation, cloned at `./caft`.
- **rho** ∈ {0.5, 0.8, 0.95, 1.0} = fraction of examples where profession matches
  stereotypical gender. 0.5 is no correlation; 1.0 is CAFT's original.
- **N must be identical across all rho levels.** Subsample to the largest N
  achievable at rho=0.5. Otherwise rho is confounded with dataset size.
- **The OOD eval set is fixed** and does not vary with rho.
- **Directions:** difference-of-means from labeled gender pairs, computed once at
  rho=1.0 and reused at every rho, so the intervention is identical across levels
  and only the data varies.
- **Ablation:** mean-ablation by default, not zero-ablation. Zero-ablation is a
  steering push when the mean projection is nonzero.
- **Arms:** plain (no ablation), ablated, ablated + test-time complement ablation,
  matched-rank random-direction control.
- **Seeds:** 2 per condition. This does not support claims about thresholds or
  phase transitions — monotonicity only.
- **Checkpoint baseline:** save checkpoints of the plain run; the ablated model
  must beat the undertrained frontier, or the intervention is just undertraining.

---

## Gates

- **Gate 1:** at rho=1.0, plain fine-tune must show ID accuracy >= 95% with OOD
  at or below chance. At rho=0.5 it must generalize correctly. If the model does
  not take the shortcut, the phenomenon is not here and we stop.
- **Gate 2:** mean-ablation during fine-tuning at rho=1.0 must improve OOD
  meaningfully over plain. If difference-of-means directions fail, that is a real
  negative result — report it, do not silently switch methods.

If a gate fails, say so plainly and stop. Do not keep going to produce something
that looks like progress.

---

## After a pod restart

Port and IP change; container disk is wiped. Run `./start_session.sh` first —
it reinstalls tmux and Claude Code, activates the venv, and sets `HF_HOME`.
Code, data, venv, and model cache under `/workspace` survive. Nothing else does.