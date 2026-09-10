# Relocation under training-time ablation — running notes

## The question

Fine-tune under a projection that removes a spurious concept's direction from
the residual stream on every forward pass. Does the concept **relocate** into the
orthogonal complement of the ablated subspace — and does relocation depend on how
strongly the training data rewards using the concept?

Motivation: three results in the literature look contradictory and may be points
on one curve.

- Persona Vectors (Chen et al. 2025, App. J.5): an L2 penalty on the trait
  projection fails; optimization re-represents the trait along alternative
  directions.
- Ustaomeroglu & Qu 2026: soft penalty on misalignment latents routed around by
  the second epoch. (Secondhand via Nadaf — not read directly.)
- Nadaf 2026 (arXiv 2607.21356): hard projection during training prevents broad
  misalignment entirely (27.7% → 0.0%, matched random subspace 27.5%), no
  re-emergence. **But** narrow adherence went 0.902 → 0.000 — the model didn't
  learn the task, so "didn't relocate" and "had no reason to relocate" are
  indistinguishable in that arm. He concedes this in limitations.

The spurious-correlation task closes that escape hatch: ID accuracy is verifiable
at every point, so we can show the task *was* learned while the concept was
blocked. Varying ρ turns a binary question into a dose-response.

## Novelty position

Varying spurious correlation strength is standard (bias-conflicting ratios in
C-MNIST, Cramér's V in WILDS, confounder strength in feature-learning dynamics —
the last reports a threshold near 0.8 in a toy setting). Training-time concept
ablation is standard. **The crossing is not.** Four searches, not exhaustive.

Honest framing for the write-up: half of this is packaging. A guardedness-literature
reader can say "blocked linear direction + continued optimization + distribution
shift, of course information moves." The dose-response is the actual contribution,
because no published result says whether relocation is pressure-gated.

## Design decisions (and why)

**Directions computed at ρ=0.5, not ρ=1.0.** At ρ=1.0 doctor and male are the
same partition, so difference-of-means over gender returns a vector equally
describable as "the doctor/nurse direction." At ρ=0.5 the cues are cleanly
crossed by construction. Still computed *once* and reused identically at every ρ,
so the intervention is fixed and only training data varies.

**Pair definition:** base model, ρ=0.5 train set, activations at the final token
position (the scored position), grouped by gender of the *correct* answer.
mean(male-correct) − mean(female-correct), normalized. Not the profession token,
not a whole-prompt mean — the decision is read out at the final position.

**Layers 14, 18, 22** of 36. Middle third, three sites, matching CAFT's count.
Fixed now to avoid relitigating per condition.

**Mean-ablation, not zero-ablation.** Zero-ablation is a steering push when the
mean projection is nonzero — this is why Persona Vectors found CAFT works for
some traits and not others.

**Measurement is causal, not a probe.** Gender is trivially decodable from
pronouns; decodability is not use. After fine-tuning: recompute directions from
base-vs-tuned activation differences, restrict to the complement of what was
ablated, ablate *those* at test time, check whether OOD accuracy moves **while ID
accuracy survives**. Both halves required — killing task ability isn't relocation.

**Model: Qwen3-4B-Base**, not Qwen3.5. The entire 3.5 family (2B/4B/9B) is
`Qwen3_5ForConditionalGeneration` — vision-language, with hybrid attention
(`layer_types`: linear_attention with full_attention every 4th layer). Non-uniform
layers would make the ablation site a confound to defend, and the module paths
are nested under a text tower. Qwen3-4B-Base is `Qwen3ForCausalLM`, 36 uniform
layers, hidden 2560, standard GQA, no chat template — CAFT's raw-completion
format applies directly. A generation back, but still well ahead of the
Gemma-2-2B this task was built on.

**Gates.** ρ=1.0: ID ≥95% AND OOD ≤20%. ρ=0.5: OOD ≥85%. Sharpened from "OOD at
or below chance" because the OOD set is the exact inversion, not a neutral set —
a fully shortcut-driven model scores ~0%, so a ≤50% bar is nearly free.

## Dataset facts (verified from the repo, session 1)

- HF arrow, 784 rows (392 nominative / 392 object), 0 duplicate texts.
- Split 0.8/0.2 by index per category before shuffle. train 626 / val 158 /
  test 158. Verified 0 overlap train↔val, train↔test.
- Two professions (doctor, nurse), four pronouns (he/she/him/her).
- Correct answer is fully determined by grammatical case; gender is never
  disambiguated by the sentence. Distractor is always opposite gender AND
  opposite case, so the options are never same-case.
- Case is balanced within profession at every ρ — the two cues are cleanly crossed.
- **N = 626 at every ρ already.** No subsampling needed; ρ is not confounded with
  dataset size.
- Realized ρ: 0.4984 / 0.7987 / 0.9489 / 1.0000. Off nominal by ≤0.005 from
  integer halving. Irrelevant for a monotonicity claim.
- **val and test are the same 158 sentences.** Only `ambiguous_frac` differs: val
  uses the training correlation, test uses 0.0 (fully inverted). OOD is not new
  vocabulary/templates/professions — it's the identical sentences with the
  pairing flipped. Fixed across ρ, as required.
- Scoring: full-vocab argmax at the final position vs the token id of " A"/" B".
  Not string matching, not restricted 2-way logprob. Mass anywhere else scores 0.
- `acc_flipped` (accuracy against the deliberately wrong answer) is a direct read
  of shortcut reliance — logging it alongside ID/OOD.

## Limitations to state plainly in the write-up

1. **The concept is ~1 bit over a 2-item vocabulary.** Two professions, four
   pronouns. A model can re-encode doctor→male in many trivial ways, which biases
   the experiment *toward* finding relocation. Defensible (CAFT used this task)
   but the claim must be about a 1-bit spurious feature, not "spurious concepts"
   generally.
2. **ID and OOD are the same 158 sentences.** "ID accuracy survives" means the
   model still answers those sentences under the training-consistent pairing —
   not that it generalizes to new text. Narrower survival check than it sounds.
3. **2 seeds per condition.** Supports monotonicity only. Does *not* support
   claims about thresholds or phase transitions, however tempting the 0.8 prior
   from the toy-model literature is.
4. Directions computed at ρ=0.5 and reused at ρ=1.0 assumes the gender direction
   is stable across the training distributions. Unverified.
5. **In a pairing where the base model already solves the training task via the
   generalizing feature, ρ varies the spurious feature's fit to the data rather
   than its marginal loss benefit** — the shortcut offers no loss reduction the
   existing solution doesn't, so shortcut adoption in that regime is
   optimizer-driven rather than reward-driven. Established by the Gate 1 failure
   (session 1): full FT at CAFT's lr never adopted the shortcut (OOD rose
   0.693 → 0.772), while LoRA at 20× the lr half-installed it (OOD 0.462).

## Open / unresolved

- CAFT's own direction-finding code doesn't run on this dataset as written:
  `pca.py` reads `inputs['assistant_masks']` (chat-template only) and
  `train_sft.py:139` prints "PCA not implemented". README flags Section 5 as WIP.
  Not a problem — we're using difference-of-means, which I own — but it means no
  reference implementation to check against.
- Latent fragility: `n_ambiguous` computed from `len(nominative)` then applied to
  `object_data` (gender.py:250). Harmless at 392/392, breaks silently if uneven.

## Session log

### Session 1 — Aug 30

Environment: Runpod L40S 46GB, `/workspace/relocation`, venv + HF cache on the
persistent volume. Pod restarted once mid-setup; container disk wipe took
`authorized_keys`, tmux, and Claude Code with it — `start_session.sh` written to
rebuild in one command.

`torch.cuda.is_available()` was False: venv had torch 2.13.0+cu130 against a
570.195.03 / CUDA 12.8 driver, GPU entirely unusable. Downgraded to
torch==2.11.0+cu128. `start_session.sh` needs an explicit
`--index-url https://download.pytorch.org/whl/cu128` pin or it silently
reinstalls the broken build on the next restart.

Dataset audit complete (facts above). Model selection resolved to Qwen3-4B-Base.
`experiments/base_eval.py` staged — replicates CAFT's scorer exactly, plus
restricted A-vs-B accuracy and **fraction of argmaxes landing on A/B**. That last
number is the real test of whether the prompt format works; every accuracy is
uninterpretable until it's high.

Next: base_eval on Qwen3-4B-Base, then Gate 1.
### Session 1 (cont.) — Gate 1, both attempts, FAIL

**Metric changes made before Gate 1** (now in every runs.jsonl row): primary
accuracy is the restricted A-vs-B logit comparison with exact ties scored 0.5;
`n_ties` logged. Under bf16, exact logit ties occur and full-vocab argmax breaks
them toward the lower token id — " A"=362 < " B"=425 — a silent bias that becomes
arm-dependent once ablation pushes logits toward indifference. `acc_full` still
logged, not headline. OOD is also broken out by the four (category × profession)
cells; the doctor↔nurse comparison at fixed grammar is the clean gender control.
Frontier evals (every 10 steps) append rows to runs.jsonl (`"frontier": true`)
instead of saving weights.

**Attempt 1, LoRA r=32, lr 1e-4** (superseded, kept for the record): ID 1.0000,
OOD 0.4620. FAIL. Asymmetric shortcut — doctor→male fully installed
(nominative_doctor 0.000, picks the male pronoun 40/40 against grammar),
nurse→female weak (nurse cells 0.92/0.77, grammar wins). Notable transient: ID
1.0 / OOD 0.956 at step 20 — grammar learned first — then 100 steps of shortcut
absorption at ~zero loss down to 0.46. Superseded because LoRA's rank-32
constraint limits which directions the model can write to, confounding the
relocation question itself.

**Attempt 2, full FT, CAFT's recipe** (lr 5e-6, warmup 0.5, wd 0.01, 3 epochs,
bnb AdamW8bit + grad checkpointing, ~33G peak of 46G): ID 1.0000, **OOD 0.7722 —
above the untrained base's 0.6930**. FAIL, decisively. Per-cell OOD:
nominative_doctor 0.925, nominative_nurse 0.564, object_doctor 0.775,
object_nurse 0.821 — no cell collapsed; residual errors look like the
pretraining prior (weakest cell = base's weakest cell), not a learned rule.
ρ=0.5 control clean: ID 0.9937 / OOD 0.9937. The step-20 grammar-first transient
did NOT replicate — but lr/warmup differ 20×/5× between attempts, so
transient-vs-recipe is confounded; can't attribute it to LoRA per se.

**Loss audit** (experiments/loss_audit.py, after "OOD going up is backwards"):
pipeline verified end to end — hand-recomputed base CE on the exact first
training batch 0.2747, full train set 0.3805, matching the observed step-10 loss
0.39; targets confirmed to be the letter tokens. Base model already scores
**0.946 on the ρ=1.0 train set at init**. Final checkpoint: train CE 0.0010, min
p(gold) 0.9835 — genuinely converged. Discriminator — the same train sentences
with doctor↔nurse swapped, grammar and options untouched: base 0.6302 → final
**0.7532**. Median gold margin on train grew 1.88 → 9.25, and the growth
survives the profession swap: it points along grammar.

**Conclusion:** at ρ=1.0 grammar fully explains the labels, so the shortcut is
redundant with a solution the base model already has at 94.6%. There was never
loss pressure to adopt it; the only gradient (residual 5% + margin sharpening)
amplified case-detection, which transferred to the inverted test set — rising
OOD is what fitting via the non-spurious feature looks like. The LoRA run shows
the shortcut can still win under aggressive optimization; adoption in this
pairing is optimizer-driven, not reward-driven (see limitation 5).

**Disk-quota crash:** full-model checkpoints (8GB each, every 10 steps) hit the
50GB volume quota mid-write (`SafetensorError: os error 122`). `df` on the pod
shows the host mfs volume (324T free), not the quota — useless for this. Fixed
by the frontier-rows-to-runs.jsonl scheme above. Also reclaimed: a complete
duplicate CUDA-13 library stack orphaned in the venv by the original broken
torch install (the cu13/cu12 nvidia packages share directories — uninstalling
one clobbers the other's files; had to reinstall 4 cu12 libs), pip cache, and
superseded LoRA checkpoints.

**State at stop:** Gate 1 FAILED for the Qwen3-4B-Base × CAFT-gender pairing
under both recipes; not building on this configuration. The three full-FT
checkpoints deleted; results/runs.jsonl (28 rows: 4 final + 24 frontier) and all
OOD prediction JSONs kept — every reported number is regenerable from those.
Kernel/env healthy; start_session.sh rebuilds torch cu128 + torchvision +
torchaudio + venv ipython + bitsandbytes and hard-asserts CUDA. **Base-competence
screen across five candidate models: specified but NOT started.**

### Session 2 — Sep 9–10: base-competence screen, 5 models

Deadline confirmed as **Sept 11** (the "twelve days" figure was stale). Pod
had restarted: kernel gone, `start_session.sh` rebuilt the env cleanly.

**Screen** (`experiments/screen_base.py`; one row per model in
`results/screen.jsonl`, raw per-item predictions in
`data/screen_<model>_predictions.json`; forward pass only, nothing trained,
nothing in runs.jsonl). Metric = the harness's restricted A-vs-B logit
accuracy, ties 0.5. Pre-committed rule: lowest base accuracy on the rho=1.0
train set subject to base OOD >= 0.50. Also logged: profession-swapped train
accuracy (grammar kept, correlation inverted — the loss-audit discriminator)
and the fraction of full-vocab argmaxes landing on A/B.

| model | train | OOD | swapped | argmax on A/B | verdict |
|---|---|---|---|---|---|
| Qwen3-4B-Base (ref, session 1) | 0.946 | 0.693 | 0.630 | ~1.0 | grammar pre-solves train |
| Qwen3-1.7B-Base | 0.8435 | 0.3703 | 0.3155 | 1.00 / 1.00 | FAIL: OOD < 0.50 |
| OLMo-2-0425-1B | 0.5743 | 0.4272 | 0.4880 | 0.19 / 0.20 | FAIL: format |
| gemma-2-2b | 0.5543 | 0.5411 | 0.4657 | 0.93 / 0.97 | **pick by rule** |
| Llama-3.2-1B | 0.5807 | 0.4335 | 0.4153 | 1.00 / 1.00 | FAIL: OOD < 0.50 |
| Llama-3.2-3B | 0.7005 | 0.4937 | 0.4784 | 1.00 / 1.00 | FAIL: OOD < 0.50 by 0.006 |

Three distinct failure modes, which is the finding:

1. **Grammar-dominant** (Qwen3-4B): base already solves train via case
   agreement; the shortcut is redundant, never rewarded (session 1).
2. **Stereotype-dominant** (Qwen3-1.7B): base OOD 0.37 and swapped-train
   0.32, both *below* chance. Raw item: "The manager observed the nurse and
   advised _____ regularly. A) she B) him" — gold B, model A (35.50 vs
   35.00). The stereotype already beats grammar at init; the shortcut is
   pre-installed, so rho=1.0 fine-tuning has nothing to install and Gate 1
   would "pass" for the wrong reason. This is what the OOD >= 0.50 floor is
   for, and a sharper reading of it than I had.
3. **Format-incompatible** (OLMo-2-1B): answers the blank directly — train
   argmax top-5 is ` he` 185, ` her` 151, ` B` 120, ` she` 106, ` him` 63.
   Its A/B accuracies are noise between two tokens it puts no mass on.
   **The screen rule has no format precondition**, so a model like this can
   win by noise (lowest train acc of the five). Recommend argmax-on-A/B
   >= 0.9 as a hard eligibility gate. Owner approved it later this session
   (now in screen_base.py; changes no verdict). No alternate prompts tried —
   that changes the task.

**gemma-2-2b's pass is a technicality.** OOD gold is 91 A / 67 B (the fixed
OOD set is letter-imbalanced — worth knowing on its own). gemma's 2-way
choice is A on 148/158 OOD items; accuracy by gold letter is A 0.934,
B 0.0075. Its 0.5411 is *below* the always-A baseline of 0.5759. On train
(310 A / 316 B) the same: A 0.997, B 0.120. It has no task competence at
init — it satisfies "OOD >= 0.50" because it says A and the OOD set is
A-heavy. Llama-3.2-3B (train 0.70, OOD 0.4937 with 10 ties) is closer to
the intended regime — some competence, OOD at chance — and misses the floor
by one item. The rule picks gemma; whether to follow the rule or the
regime is a design call, not mine.

**Disk:** the five downloads hit the 50G quota (gemma ships fp32: 9.8G; OLMo
7.3G). Two concurrent sessions deleted caches: Llama-1B/3B, Qwen3-1.7B and
OLMo are gone (predictions saved, re-downloadable). Only Qwen3-4B and gemma
remain cached. ~35G used after Gate 1.

**Gate 1 on gemma-2-2b launched** (`experiments/gate1_gemma.py`): same recipe
as session 1 (CAFT's lr 5e-6 config, verified in
`caft/spurious_correlations/training/config.py`), eager attention for
gemma-2's softcapping, rho=1.0 then rho=0.5, seed 0. Weights saved for the
rho=1.0 final only. Rows carry `script_sha256` because `git_hash` is CAFT's
upstream commit and does not track the harness (all 28 session-1 rows carry
the identical hash for that reason; `/workspace/relocation` is not a repo).

### Session 2 (cont.) — Gate 1 on gemma-2-2b: PASS

`experiments/gate1_gemma.py`, seed 0, N=626 at both rho, CAFT recipe
unchanged. Rows `gate1_gemma_rho1.0_s0`, `gate1_gemma_rho0.5_s0` + 24
frontier rows in runs.jsonl (54 rows total). Weights kept for the rho=1.0
final only (4.9G; 36G used).

| | ID | OOD | OOD cells (nom_doc / nom_nurse / obj_doc / obj_nurse) |
|---|---|---|---|
| rho=1.0 | 1.0000 | **0.1266** | 0.45 / 0.00 / 0.05 / 0.00 |
| rho=0.5 | 0.9937 | **1.0000** | 1.0 / 1.0 / 1.0 / 1.0 |

Base OOD reference 0.5411 (matches the screen exactly; both reloads clean).
Zero ties at the end of either run; argmax on A/B 1.00 throughout.

Trajectory, rho=1.0: OOD 0.53 (step 10) → 0.40 (20) → 0.24 (30) → 0.13
(70, plateau). Loss ~0 from step 40. No grammar-first transient: this model
had no grammar solution to start from and went straight to the shortcut.
rho=0.5: ID and OOD rise together, OOD 1.0 by step 60 — from the same 626
sentences it learns case agreement when the data doesn't reward profession.

**Discriminator** (`experiments/gemma_discriminator.py`,
`data/gemma_discriminator.json`): train sentences with doctor↔nurse swapped,
grammar untouched. Base 0.4673 → final **0.1470** (train itself 0.5479 →
0.9984; median gold margin +7.05 on train, −6.09 on swapped train). The fit
is through profession. This is the mirror of Qwen3-4B's 0.63 → 0.75.

Raw rho=1.0 OOD completions (first two; full set in
`data/gate1_gemma_rho1.0_s0_ood_predictions.json`):
- "The doctor requested time off, so we accommodated _____. A) her B) he" —
  gold A, model B (3.88 vs 10.94). Ungrammatical "he" in object position,
  because doctor.
- "The manager observed the nurse and advised _____ regularly. A) she B) him"
  — gold B, model A (12.69 vs 7.91). "she", because nurse.

Ways this could be wrong, checked: letter bias — OOD gold is 91 A / 67 B;
always-B would score 0.424, always-A 0.576; 0.1266 is not reachable by
letter bias. Undertraining — no, loss ~0.003, train acc 0.998 (min p(gold)
0.026 on one object_doctor item). Leakage — OOD sentences are the fixed
CAFT test split, unchanged from session 1. Not checked: a second seed;
whether nominative_doctor's residual 0.45 is a real asymmetry or noise (40
items).

**Reading.** Gate 1 passes, so the phenomenon exists in the gemma-2-2b ×
CAFT-gender pairing. Combined with the screen, the picture is that the
spurious feature gets adopted when the base model has *neither* solution at
init and the data rewards the cheaper one; when a grammar solution is
pre-installed (Qwen3-4B) the shortcut is never rewarded, and when the
stereotype is pre-installed (Qwen3-1.7B) there is nothing to adopt. That's a
hypothesis from three models at one seed each, not a result.

**Next by design:** Gate 2 — mean-ablation of difference-of-means gender
directions during fine-tuning at rho=1.0. Direction-finding is the user's;
the ablation training loop can take directions as an input.

### Session 3 — Sep 10: relocation harness, Gate 2

Harness built and smoke-tested (`experiments/harness.py`, `ablation.py`,
`train_arm.py`, `run_matrix.py`; hook check: projection pinned to μ within
0.01 at ‖h‖ 100–220, hooks-off logits restored exactly, base OOD under a
random direction 0.5380 vs 0.5411). Layers 10/13/16 of 26 (owner pick, same
relative depths as 14/18/22 of 36). Mean-ablation in fp32 inside the hook.

**Diff-of-means directions as specified** (base, ρ=0.5 train, gender of
correct answer, final token; `data/directions/diffmeans_gemma.pt`) are weakly
defined: d′ 0.45 / 0.52 / 0.80 at L10/13/16, sign-accuracy 0.64 / 0.63 / 0.67,
‖mean diff‖ 0.8 / 1.2 / 3.3 against ‖h‖ ~107 / 151 / 219. The base is at chance
on the task, so it does not compute gender-of-answer at the scored position.
The profession contrast (doctor − nurse) is 5–10× larger (‖·‖ 8.0 / 10.6 /
17.1) with a substantial component along u (0.47 / 0.75 / 2.33): the
stereotype axis lies partly along the gender direction. Cross-layer cosines
0.12–0.35.

**Pre-declared before any Gate 2 result (owner delegated the call):** Gate 2
runs with the specified diff-of-means direction as the headline; a
profession-contrast direction (`directions_profession.py`, arm
`ablated_prof`) runs as a declared secondary variant. If the headline fails
and the variant works, the write-up reports that as a finding about where
the concept is represented, not as a method switch. Random controls are
matched-rank (k=1), μ from the same base acts, |cos| with diff-of-means
≤ 0.04 (`random_gemma_s{0,1}.pt`).

Complement step (test-time): on the same tuned model in memory, two rows per
run — diff-of-means-in-complement (headline) and top-PC-of-(tuned−base)-in-
complement (secondary). Owner chose to log both.

**Gate 2 result — FAIL, both directions** (seed 0, ρ=1.0, rows `arm_ablated_rho1.0_s0`,
`arm_ablated_prof_rho1.0_s0` and their `_complement` / `_complement_pc` rows;
plain reference `gate1_gemma_rho1.0_s0` OOD 0.1266):

| arm | direction | ID | OOD hooks on | OOD hooks off | +complement (dm) | +complement (pc) |
|---|---|---|---|---|---|---|
| ablated | diff-of-means (gender of answer) | 1.0000 | **0.1234** | 0.1203 | 0.1361 | 0.3228 (ID 0.9399) |
| ablated_prof | profession contrast | 1.0000 | **0.1487** | 0.1329 | 0.1139 | 0.1614 |

Mean-ablating one direction at layers 10/13/16 throughout fine-tuning changed
nothing: the shortcut was adopted exactly as in the plain run. Not a
"meaningful improvement over plain" under any reading; the pre-declared
secondary variant fails identically.

Checked: the hooks were live — untrained base OOD under the diff-of-means
ablation 0.4209 vs 0.5411 clean; step-10 loss 1.0853 vs plain 1.1881; the
post-FT probe's gender contrast has zero component along U (projection is
constant by construction, `norm_v_complement == norm_v`); smoke test pins
projection to μ within 0.01. ID 1.0 on both arms, so the task was learned —
this is the Nadaf escape hatch closed: blocked and still learned, and still
took the shortcut.

Not checked: second seed; more than rank 1 per layer; other layers (the
concept may be read out later than L16 or spread across many directions —
CAFT's own intervention is several PCs at many layers); whether the tuned
model's gender direction moved elsewhere (the +complement rows say the
tuned diff-of-means direction in the complement at these layers, ablated at
test time, does not move OOD either: 0.136 / 0.114). The `complement_pc` row on
the diff-of-means arm (OOD 0.32, ID 0.94) is the only test-time ablation that
moved OOD, and it cost ID — so it is partly task damage, not a clean read.

Reading (hypothesis, one seed): a rank-1 mean-ablation at three mid-layers
is far too small a subspace to block a concept this redundant. Note that the
diff-of-means direction had d′ ≤ 0.8 on the base — the base does not
represent gender-of-answer at the scored position — and the profession
direction (d′ 2.3–2.9) fails too, so "weak direction" is not the whole story.
Per CLAUDE.md, stopping here; whether to try a larger subspace is a design
change for the owner. `data/checkpoints/arm_ablated_rho1.0_s0/final` kept (4.9G).

### Session 3 (cont.) — intervention dose-response, PRE-DECLARED before results

Owner (01:10 UTC): "use your discretion to continue... should report something
new and interesting; ~20 h of work." Design calls below are mine; declared
before any run.

**Question.** Before ρ can matter, the intervention has to block the concept at
all. Rank-1 mean-ablation at three layers did not. How much subspace does it
take, and does a matched random subspace of the same rank do the same?

**Directions** (`directions_pca.py`): per layer, all 26 layers, on the ρ=1.0
train set, residual stream pooled over two positions — the profession token
(where the cue enters) and the final token (where it is read out) — centred,
SVD, PCs ranked by |corr(score, profession)|, top 64 kept. Two sources: the
base model and the plain-tuned ρ=1.0 checkpoint (the directions the shortcut
is known to use — an "oracle" variant that gives up "computed on base" for
"known to matter"). Random control: QR of Gaussian [2304×64] per seed, μ from
base acts. A run slices (layers, k).

**Sweep** (`run_dose.py`, seed 0, no complement step): source ∈ {pca_base,
pca_tuned, randk} × k ∈ {64, 16, 4, 1} at all 26 layers, at ρ=1.0 and at ρ=0.5;
then pca_tuned at layers 10/13/16 for k ∈ {64, 16, 4} at ρ=1.0. 27 runs.

**Pre-declared reads.** A dose *blocks* if at ρ=1.0 OOD ≥ 0.50 with ID ≥ 0.95
and the same-k random control stays < 0.30. A blocking dose is *usable* if at
ρ=0.5 it still reaches OOD ≥ 0.85 (grammar learnable under the intervention).
If a usable blocking dose exists, the relocation matrix (ρ × 2 seeds × plain /
ablated / random, complement on) runs at the smallest such dose. If none
exists, the result is that rank-64 mean-ablation at every layer does not stop
a 2B model from adopting a 1-bit shortcut, with the random control showing
whether that is routing-around or insensitivity.

**Dose sweep result (seed 0; 27 runs `arm_{pca_tuned,pca_base,randk}_L{26,3}_k{1,4,16,64}_rho{1.0,0.5}_s0`):**

ρ=1.0, all 26 layers — OOD hooks on (hooks off):

| k | tuned PCs | base PCs | random |
|---|---|---|---|
| 1 | **0.9937** (0.962) | **0.9905** (0.915) | 0.1266 (0.120) |
| 4 | 0.9873 (0.975) | 0.9810 (0.813) | 0.1076 (0.095) |
| 16 | 0.9873 (0.848) | 0.9937 (0.886) | 0.1076 (0.180) |
| 64 | 1.0000 (0.576) | 0.9873 (0.737) | 0.0158 (0.472) |

ρ=1.0, layers 10/13/16 only, tuned PCs: k=4 0.1519, k=16 0.0506, k=64 0.0854.
ID ≥ 0.987 in every run. ρ=0.5 control: every condition reaches OOD ≥ 0.99
(grammar learnable under all of them).

**Reading.** The blocking variable is *depth coverage*, not rank. One
profession-correlated PC per layer at all 26 layers blocks the shortcut
completely (plain 0.127 → 0.99) with ID intact; sixty-four at three mid
layers does nothing. Matched random subspaces never block (rank-64 random
even sharpens the shortcut, 0.016). This retroactively explains Gate 2: the
failure was three layers, not rank 1 or a weak direction. Mechanistically:
the profession feature is recomputable from the complement at every layer,
so a partial-depth block is routed around — relocation *across layers* —
while a full-depth block of even one direction is not routed around at
ρ=1.0. Hooks-off (deployed without projection) degrades with rank: the
learned grammar solution becomes entangled with the intervened geometry
(tuned k=64 off = 0.576 = always-A baseline). At k=1 it survives (0.92–0.96).
Smallest usable blocking dose by the pre-declared rule: base PCs, 26 layers,
k=1 (non-oracle).

**Pre-declared next (01:45 UTC):** relocation matrix ρ ∈ {0.5, 0.8, 0.95, 1.0}
× seeds {0,1} × {plain, ablated, random}, complement on, at TWO doses:
(a) 3 layers, tuned PCs, k=16 — the partial block that is routed around at
ρ=1.0; the question is whether routing-around is pressure-gated (does the
partial block hold at lower ρ?); (b) 26 layers, base PCs, k=1 — the full
block; expected flat. (a) runs first. Random controls are seed-matched
randk slices at the same layers/k.

**Pre-declared (01:50 UTC), queued behind the matrix:** layer-position sweep
(`run_layers.py`): base PCs, k=1, ρ=1.0, seed 0, over layer sets — thirds
(0–8, 9–17, 18–25), halves (0–12, 13–25), every even layer, and single layers
{0,4,8,12,16,20,24}. Read: does the block need to be everywhere, or is there
a depth at which one direction suffices? Row names `arm_pca_base_<set>_k1`.

**Relocation matrix, dose (a) — partial block (16 tuned PCs × layers 10/13/16), DONE
(24 runs, rows `arm_{plain,abl_pca_tuned_L3_k16,rnd_L3_k16}_rho{0.5,0.8,0.95,1.0}_s{0,1}`).**
OOD, projection on, seed 0 / seed 1:

| ρ | plain | ablated (partial) | random (matched) |
|---|---|---|---|
| 1.0 | 0.127 / 0.095 | 0.051 / 0.171 | 0.184 / 0.149 |
| 0.95 | 0.750 / 0.646 | 0.658 / 0.560 | 0.804 / 0.699 |
| 0.8 | 0.978 / 0.975 | 0.990 / 0.975 | 0.987 / 0.981 |
| 0.5 | 1.000 / 0.987 | 1.000 / 0.994 | 1.000 / 0.994 |

ID ≥ 0.975 everywhere. Reads: (1) The plain ρ curve is steep: the shortcut is
fully adopted only at ρ=1.0; at ρ=0.95 (31 counterexamples) OOD is already
0.65–0.75, at 0.8 it is ~0.98. (2) The partial block is routed around at
every ρ — it never beats plain; at ρ=0.95 it is below plain in both seeds
(−0.09) while random is above plain in both (+0.05). Two seeds; the seed
spread of plain at 0.95 is 0.10, so the ordering is suggestive, not a
result. Routing-around is not visibly pressure-gated in this range. (3) The
test-time complement ablation (rank-1 diff-of-means in the complement at
the three layers) moves OOD by ≤ 0.05 in every row, INCLUDING the plain
arm at ρ=1.0 (0.127 → 0.133), which was the positive control. So the
complement measurement as designed is not sensitive enough to detect where
the concept lives — consistent with the training-time finding that rank-1
at three layers does nothing. The complement rows are uninformative and the
write-up must say so; the relocation claim rests on the training-time
depth-coverage result, not on these.

**Relocation matrix, dose (b) — full block (1 base PC × all 26 layers), DONE
(16 runs, rows `arm_{abl_pca_base_L26_k1,rnd_L26_k1}_rho*_s*`; plain shared with (a)).**
OOD, projection on, seed 0 / seed 1 (hooks-off in parentheses for the ablated arm):

| ρ | plain | ablated (full) | random (matched) |
|---|---|---|---|
| 1.0 | 0.127 / 0.095 | **0.991 / 1.000** (0.915 / 0.968) | 0.127 / 0.117 |
| 0.95 | 0.750 / 0.646 | **1.000 / 0.994** (0.965 / 0.902) | 0.734 / 0.557 |
| 0.8 | 0.978 / 0.975 | 1.000 / 0.997 (0.987 / 0.946) | 1.000 / 0.975 |
| 0.5 | 1.000 / 0.987 | 1.000 / 1.000 (0.968 / 0.987) | 1.000 / 0.994 |

ID ≥ 0.994 for the ablated arm at every ρ. Read: the full-depth rank-1 block
is ρ-independent — OOD ≥ 0.99 at every level of data pressure including
ρ=1.0, where plain is 0.10–0.13 and the matched random control is
indistinguishable from plain. There is no relocation into the complement
that the model can find under any ρ tested when every layer is blocked, and
the block does not damage grammar learning at ρ=0.5. Deployed without the
projection the model keeps 0.90–0.99. Together with (a): relocation in
this setup is a matter of *depth coverage* — a partial-depth block is
routed around at every ρ; a full-depth block is not routed around at any ρ.
Data pressure did not gate either outcome in the range tested (2 seeds).

**Layer-position sweep DONE (02:28 UTC; 13 runs `arm_pca_base_<set>_k1_rho1.0_s0`, base PCs, k=1, ρ=1.0, seed 0).**
OOD projection on (hooks off):

| layer set | n | OOD |
|---|---|---|
| 0–8 (early third) | 9 | **0.994** (0.943) |
| 9–17 (mid third) | 9 | 0.082 (0.070) |
| 18–25 (late third) | 8 | 0.133 (0.133) |
| 0–12 (first half) | 13 | **0.987** (0.943) |
| 13–25 (second half) | 13 | 0.082 (0.095) |
| even layers | 13 | **0.994** (0.949) |
| layer 0 only | 1 | **0.981** (0.956) |
| layer 4 only | 1 | 0.487 (0.266) |
| layer 8 / 12 / 16 / 20 / 24 only | 1 | 0.133 / 0.158 / 0.152 / 0.082 / 0.152 |

ID ≥ 0.994 everywhere. Read: "depth coverage" was the wrong abstraction —
what matters is whether the block includes the **entry layer**. One
profession-correlated direction at layer 0 (PC1 there is essentially the
doctor−nurse token-identity axis, |corr| 0.71) removes the shortcut's
usefulness for the whole fine-tune; by layer 4 the same rank-1 block is half
effective, by layer 8 it is routed around entirely. Every 26-layer success
in the dose sweep was carried by layer 0. Rank-64 at layers 10/13/16 failed
because by then the cue is spread across the residual basis.

**Pre-declared follow-ups (02:30 UTC):** (i) seed 1 for {only0, only4,
early0-8} and seed 0 for {only1, only2, only6} to bound the entry window;
(ii) diagnostic: with the layer-0 block on, is profession still linearly
present downstream (d′ of doctor−nurse at layers 4/8/16/24, base model, with
vs without hook)? This distinguishes "blocked from use" from "deleted".
Reported as information, not use (CLAUDE.md).

**Diagnostic result (`diag_layer0.py`, `data/diag_layer0_profession_info.json`) — REINTERPRETATION.**
Base model, ρ=1.0 train, d′ of doctor-vs-nurse along the diff-of-means axis,
with vs without the layer-0 rank-1 block:

| layer | profession token d′ off → on | final token d′ off → on |
|---|---|---|
| 0 | 278 → 0.23 | 2.2 → 2.2 |
| 4 | 25 → 0.16 | 5.5 → 1.1 |
| 8 | 22 → 0.16 | 3.1 → 0.7 |
| 16 | 8.9 → 0.15 | 2.5 → 0.7 |
| 24 | 5.5 → 0.16 | 2.7 → 0.6 |

The layer-0 block does not block *use* of the cue; it **deletes** the cue.
Two tokens differ along essentially one direction, PC1 at layer 0 is that
direction, and mean-ablating it makes doctor and nurse indistinguishable at
every subsequent layer (d′ ≈ 0.16). "Layer 0 alone blocks the shortcut" is
therefore the trivial end of the dose — equivalent to swapping in a
profession-neutral token — and every 26-layer success in the dose sweep was
carried by layer 0. This must be stated plainly in the write-up.

What remains substantive: once the cue has been processed (layer ≥ 5), the
profession information is spread across the residual basis, and neither
rank-1 at 21 downstream layers (sets 9–17, 13–25, 18–25 all fail) nor
rank-64 at three layers blocks it. The untested and decisive case is
**high rank at all downstream layers**.

**Pre-declared (02:35 UTC), queued:** `run_layers3.py`: base PCs, ρ=1.0,
seed 0: layers 5–25 at k ∈ {1, 16, 64}; layers 1–25 at k=1 (is layer 0 the
whole entry window?). Then `diag_layer0.py` generalised (`DIAG_HOOK`) under
the layer-4 block and under the 5–25/k=64 block, to check whether the cue
survives downstream in those cases. Reads: if 5–25/k=64 blocks with the cue
still present downstream, the concept CAN be blocked from use after
distribution; if it fails, a 2B model routes a distributed 1-bit concept
around a 64-dim-per-layer block at every processed layer.

**Layer follow-up DONE (02:34 UTC, `run_layers2.py`).** Seed 1 replicates:
only0 0.994 (s0 0.981), only4 0.589 (s0 0.487), early0-8 0.994 (s0 0.994).
Boundary, seed 0: only1 0.981, only2 0.902, only6 0.358 (only8 0.133). So a
single rank-1 block's effectiveness decays smoothly with depth: ≈1.0 at
layers 0–1, 0.90 at 2, ~0.5 at 4, 0.36 at 6, plain-level by 8. Read: the
profession cue leaves its one-dimensional (token-identity) form over the
first ~6 layers; the decay curve is a direct measure of how quickly the
concept becomes distributed. Hooks-off tracks hooks-on closely for the
early layers (0.94/0.89/0.73 at L0/1/2).

**Downstream-only, rank 64 at layers 5–25 (`arm_pca_base_L5-25_k64_rho1.0_s0`, 02:35 UTC):**
ID 0.994, OOD **0.386** hooks on (0.139 hooks off). Cells: nominative_doctor
0.625, object_doctor 0.875, nominative_nurse 0.000, object_nurse 0.026 — the
block is asymmetric: doctor→male is largely suppressed, nurse→female fully
retained. Frontier: OOD 0.42 (step 10) → 0.58 (20) → 0.57 (40) → 0.49 (50) →
0.39 (70, plateau) while loss → 0: grammar is learned first, then the
shortcut is partially re-adopted through the block — a routing-around
trajectory over training, the same shape as the session-1 LoRA transient.
Read: with 64 profession-correlated directions removed at every processed
layer, a 2B model still recovers about half of the shortcut's benefit.
Removing the projection at test time returns it to plain (0.139), so what
was learned under the block depends on the block. This is the substantive
downstream number; the k=16 / k=1 downstream and the 1–25/k=1 runs bound
it, and the diagnostics say whether the cue survives beneath it.

**Downstream-only sweep + diagnostics DONE (02:37 UTC).**
`arm_pca_base_L5-25_k{1,16,64}_rho1.0_s0`: OOD 0.367 / 0.310 / 0.386 (hooks
off 0.285 / 0.218 / 0.139); `L1-25_k1`: 0.968 (off 0.763). Rank is irrelevant
downstream (1 → 64 all ≈ 0.31–0.39), and 21 downstream layers at rank 1
(0.367) ≈ layer 6 alone (0.358) ≫ layers 9–17 or 13–25 (0.08): the whole
downstream partial effect comes from layers 5–8, the tail of the entry window.

Diagnostics (`data/diag_profinfo_L4-4_k1.json`, `..._L5-25_k64.json`; d′ of
doctor-vs-nurse, base model):
- Layer-4 rank-1 block: profession-token d′ 25 → 0.38 at L4 and ≈0.2–0.3
  downstream — but the **final token already carries the cue at layer 4**
  (d′ 5.47 → 5.42, unchanged) and keeps it (L8 2.99, L16 1.68, L24 1.77).
  The cue has been copied to the answer position before layer 4; blocking
  the profession position there is too late for half of it.
- Layers 5–25, rank 64: profession-token d′ ≈ 0.3 downstream (removed), but
  final-token d′ survives at 2.84 (L8), 1.82 (L16), 2.13 (L24) — inside the
  complement of a 64-dim block at every layer. The fine-tune under that
  block reaches OOD 0.39 by reading exactly this surviving information (the
  frontier shows grammar learned first, then the shortcut re-adopted).

**Final reading.** Whether training-time linear ablation blocks a 1-bit
spurious concept is decided by *where in depth* the concept is still
low-dimensional, not by rank, and not by data pressure ρ. At layers 0–1 it
is the token-identity axis and a rank-1 mean-ablation deletes it (trivial,
and it is what carried every "26-layer" success). By layer 4 it has been
copied to the answer position; by layer 8 it is distributed such that
removing 64 profession-correlated directions per layer at every remaining
layer leaves it at d′ ≈ 2 in the complement, and the model uses it (OOD
0.31–0.39 vs plain 0.13, random ≤ 0.13). The concept did not *relocate*
during training — it was already in the complement at initialisation; the
block simply fails to cover it. Across ρ, a partial block is routed around
at every ρ and a (deleting) full block holds at every ρ: no pressure gating.
The test-time complement measurement failed its positive control and is
uninformative.

**State at stop (02:40 UTC):** 130+ rows in runs.jsonl; kernel holds a clean
base gemma; disk ~45G of 50G (two 4.9G checkpoints kept:
gate1_gemma_rho1.0_s0, arm_ablated_rho1.0_s0). Figures: dose_response.png,
layer_sweep.png, ood_vs_rho.png, complement.png, frontier_rho1.png.
Write-up: writeup_draft.md.
