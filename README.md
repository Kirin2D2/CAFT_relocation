# CAFT_relocation

Does a spurious concept relocate under training-time ablation? Fine-tuning gemma-2-2b on the
CAFT gender-bias task while mean-ablating profession/gender directions from the residual stream on
every forward pass, across correlation strength ρ, ablation rank, and depth.

- **`writeup_draft.md`** — the write-up (TL;DR, findings, limitations).
- **`notes.md`** — running lab notes with pre-declarations written before each result.
- **`results/runs.jsonl`** — one row per training/eval run (182 final rows + frontier rows).
  Every number in the write-up comes from here. `results/screen.jsonl` — the base-model screen.
- **`figures/`** — regenerate with `python experiments/plot_matrix.py`; tables with
  `python experiments/table_matrix.py`.
- **`experiments/`** — `harness.py` (eval, activation collection), `ablation.py` (mean-ablation
  hook), `directions_*.py` (direction finding), `train_arm.py` (one run), `run_*.py` (sweeps).
- **`data/directions/*.pt`** — the ablated subspaces; `data/*_ood_predictions.json` — raw per-item
  completions for every run. Model checkpoints and cached activations are not committed.
- **`caft/`** — the CAFT repository as a submodule, pinned to the commit used.

Environment: see `CLAUDE.md` and `start_session.sh`.
