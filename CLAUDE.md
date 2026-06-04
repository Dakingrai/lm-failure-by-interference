# CLAUDE.md — lm-failure-by-interference

Codebase for the paper *“Failure by Interference: Language Models Make Balanced
Parentheses Errors When Faulty Mechanisms Overshadow Sound Ones”* (RASTEER).
Top-down mechanistic interpretability + a ranking/steering method, on the balanced-
parentheses task across GPT-2 {Small, Medium, Large, XL}, CodeLlama-7b, Llama2-7b,
Pythia-6.9b. We are refactoring it from a flat pile of scripts into a `src/lmfi/`
package + thin, config-driven experiment entry points.

## Status (where the migration is)

- Branch: `refactor/lmfi-migration` (off `main`; `main` is untouched).
- **Migration COMPLETE.** All experiments (00, 02–06) migrated into `src/lmfi/` +
  `experiments/` + `configs/`, one commit each, each validated original ≡ migrated
  on gpt2 (byte-identical where deterministic; seedless-data variation noted where it
  applies). The flat root scripts and `utils/` are gone; only `data/` (canonical),
  `src/`, `experiments/`, `configs/`, `results/` (gitignored) remain.
- Commits: cleanup eb12222 / StopForward 86e4155 / scaffold 75d6ab8 / accuracy 5aa22c5 /
  projection 1ebf8fd / generalizability b1735e6 / component-figures aceec3f /
  steering 3a707cb / synthesis (this commit).
- Remaining (optional, not blocking): regenerate the full multi-model projection/
  steering outputs (GPU) for pythia/CodeLlama/Llama-2 to validate their Table 1 / Fig 4
  anchors; build the Fig 6 generator ("deferred-and-easy"). Fig 4(c)/(f) & Fig 5 stay
  out of scope (circuit baseline not in this repo).

## Prime directive

Refactor **with understanding, one experiment per commit**, on `refactor/lmfi-migration`.
Per experiment: lift logic into `src/lmfi/` → add a thin `experiments/NN_name/run.py`
that loads a config → verify outputs against the regression anchor → commit. Do not
touch other experiments in the same step. Keep commits single-purpose (deletions-only
vs additions-only vs one experiment’s migration). **If a change would alter a number in
the paper, STOP and surface it** — every change is anchored to the figure provenance below.

## Migration order

```
00  done   cleanup (deletions only)                     — eb12222
01  now    scaffold src/ + configs/ + this file (adds)  — current
02         accuracy           get_accuracy.py            — canary: validates the src/ data+model layer
03         projection         proj_attn.py, proj_neuron.py   — hub: feeds 04 & 06
04         generalizability   analyze_attn_proj.py, analyze_neuron.py, attention_acc_distribution.py (Fig 1)
05         component figures  precision_recall_plot.py (Fig 3 + Fig 8), plot_dual_neuron.py (Fig 2)
06         steering           steer_{attn,neuron,both}.py (+ *_test) + analyze_dev_steer.py -> one run.py
           (synthesize_data.py -> src/lmfi/data; migrate anytime — committed data/ is canonical)
```

**Out of scope — no code in this repo. Do NOT create folders for these:** the circuit-
discovery baseline (§4.2 / App C), arithmetic (§5.4.3), HumanEval. Consequently Fig 4(c)/(f)
and Fig 5 (which depend on the circuit baseline) cannot be reproduced here. Fig 6 has no
committed generator either, but needs only RASteer top-60 F1 rankings the pipeline already
produces — tag it “deferred-and-easy”; reserve a slot, build later, do not block on it.

## Target structure

```
src/lmfi/
  io.py              # read_json / save_file / create_results_dir / clear_cache — SINGLE source
  data/              # synthesis (seeded), loaders, SubTask (the _0..3 <-> 1..4-paren mapping)
  models/            # load_model + per-arch HookedTransformer configs (keep ALL branches)
  analysis/          # proj processing, generalizability, precision/recall (analyze_* consolidated)
  steering/
    interventions.py # the 4 used Intervene* classes (from activation_patching.py)
    ranking.py       # recall / precision / F1 component ranking
    run.py           # ONE driver: component_type {attn,neuron,both} x split {dev,test} x metric
  viz/               # figures incl. plot_dual_neuron; route through plot-wong
configs/             # per-model + per-experiment; holds the slice defaults that reproduce the paper
experiments/00_synthesize/ 01_accuracy/ 02_projection/ 03_generalizability/
            04_component_figures/ 05_steering/   # each: thin run.py + config
data/                # unchanged, committed (84 JSON)
results/             # gitignored
tests/               # regression anchors
```

No `circuits/`, no faithfulness metric (no circuit code in repo).

## Hard rules

1. **No config in `main()`.** Models, sub-tasks (`n_paren`), metric, multiplier range,
   top-k, split, and all paths come from config/CLI. Never reintroduce slices like
   `[:2]` or commented-out config toggles.
1. **Slices become config with paper-reproducing defaults** (see provenance). PRESERVE
   the intentional ones as documented defaults: `proj_neuron`’s `[:50]` (neuron metadata
   is computed on <=200 examples/model — “to reduce the time”); `steer_neuron`‘s `[-3:]`
   models + subtasks 3/4 only. The `[:2]`s are debug leftovers — set them to the figure’s
   real model set, do not freeze: `analyze_attn_proj`, `attention_acc_distribution`,
   `steer_attn` -> all 7; `precision_recall_plot` -> two presets (rule 9).
1. **One steering driver.** `steering/run.py`, parameterized by component_type
   {attn,neuron,both} x split {dev,test} x metric. The neuron-test path is **built** here
   (mirror `steer_neuron.py` + a `coeffs_mlp` bridge), **not** refactored from the deleted
   `steer_neuron_test.py`. `steer_neuron.py` (reads test data, `[-3:]`, subtasks 3/4) was
   the actual neuron run that produced Fig 4(b)/(e).
1. **One `SubTask` abstraction.** Encode the 0-indexed file suffix (`_0..3`) <-> 1-indexed
   paren-count (1..4) mapping in exactly one place. No inline `n-1` / `f"_{n}"` arithmetic.
1. **One copy of every shared helper.** `read_json`/`save_file`/`create_results_dir`/
   `clear_cache`/`load_model`/`load_data` live in `src/lmfi/` (`io.py`, `models/`) and are
   imported. The duplicated copies inside the `steer_*` files get dropped as those migrate.
1. **No machine-specific paths.** Model caches come from the registry/env — strip
   `../ziyuyao/...` and `load_model`’s personal default. All output paths derive from one
   run-dir root in config; no scattered `f"results/..."` literals.
1. **Determinism.** Seed data synthesis. Keep the committed `data/` as canonical ground
   truth — regenerable, not regenerated.
1. **Preserve the science.** Keep `load_model`‘s per-arch HookedTransformer branches
   (`fold_ln`, `center_writing_weights`, `use_attn_result`, fp16); the additive logit-lens
   decomposition; Algorithm 1 (selective promotion vs distractors) and Algorithm 2
   (thresholded, tau=0.5) semantics; the dual-sign neuron treatment. Refactors are
   structural — they must not change what is computed. Don’t blindly delete
   `clear_cache`’s `time.sleep` — reduce/parameterize it (it lets GPU memory settle).
1. **`precision_recall_plot` is two presets, not one pair.** Fig 3 = {gpt2, CodeLlama-7b-hf};
   Fig 8 (App B.2) = {gpt2-medium, gpt2-large, gpt2-xl, Llama-2-7b}. It cannot reproduce
   either until its `final_v/` read-path mismatch is fixed (see broken items).

## Broken items to fix during migration (verified against main.pdf)

- `analyze_neuron.py` stage-2 `NameError` (`for m in models` with `models` undefined). That
  stage writes `neuron_coeffs.json`, which **nothing consumes** — Fig 2 / `plot_dual_neuron.py`
  reads proj data directly. So it’s a dead stage: fix the one-line scope bug or drop it;
  either way it changes no paper number.
- Path mismatches to reconcile under the new config path scheme: `steer_neuron.py` writes
  `.../mlp/<metric>/last_paren/new/...` but `analyze_dev_steer.py` reads `.../mlp/dev/<metric>/dev/...`;
  `precision_recall_plot.py` reads an unwritten `.../final_v/...`.

## Figure -> generator provenance (anchor for “don’t change a paper number”)

- Table 3 / App A (accuracy): `get_accuracy.py`, all 7.
- Fig 1 (attn-acc dist, CodeLlama): `attention_acc_distribution.py` — the `[:2]` made the
  App Fig 7 version; default -> all 7 (Fig 1 + Fig 7 cover all 7).
- Fig 2 (dual-sign neuron L19N11, CodeLlama): `plot_dual_neuron.py((19,11))` — reads proj data.
- Fig 3 (precision-recall, CodeLlama + GPT-2 Small) + Fig 8 (App): `precision_recall_plot.py` — two presets (rule 9).
- Fig 4(a/b/d/e) (RASteer attn / FF): `steer_attn.py` / `steer_neuron.py` (latter `[-3:]` + subtasks 3/4 + test data).
- Table 1 (generalization counts, all 7): `analyze_attn_proj.py` + `analyze_neuron.py`.
- Table 5 (attention-pattern viz): was `plot_attn_pattern` in the deleted `plotly_utils.py`
  (ad hoc / uncommitted notebook) — **not reproducible from committed code**; rebuild as a
  separate viz task only if wanted.
- No committed generator: Fig 4(c)/(f), Fig 5 (need circuit baseline — out of scope); Fig 6 (deferred-and-easy).

## Data conventions

Record: `{"prompt": "#print the string 486\nprint(486", "label": ")", "label_idx": int}`.
Files: `data/{model_folder}/{train,dev,test}_labeled_last_paren_{0..3}.json`, where
`model_folder = name.split("/")[-1]` and suffix `n` => (n+1)-paren sub-task. Counts:
350 train / 150 dev / 150 test per sub-task. Only `last_paren` files are committed
(no `second_last`/`third_last` — those synthesis branches never fired for these tokenizers).

## Model registry

`utils/models.json` = list of `{"name", "cache"}`; names map to TransformerLens / HF IDs
(`gpt2`, `codellama/CodeLlama-7b-hf`, …). Strip leaked personal paths from `cache`; treat
cache location as environment-specific, not committed.

## Verification — the regression anchor

Before migrating logic, capture current-code numbers as the anchor. Cheapest = balanced-
parens accuracy (Table 3):

```
GPT-2 Small : 100 / 100 / 49 / 0    (1/2/3/4-paren)
CodeLlama-7b: 99 / 100 / 98 / 87
Pythia-6.9b : 100 / 100 / 100 / 83
```

After each step, the refactored code must reproduce the anchor for the models it ran.
**Beware:** removing a `[:2]` may make a script run all 7 — compare the *same* model set
before vs after, or you’ll mistake a config change for a regression.

## Do NOT

- Migrate multiple experiments in one commit, or edit files outside the experiment being migrated.
- Change random behavior, token-id sets (`Tneg` distractors), the last-token slice, or
  metric definitions while “tidying.”
- Create folders for out-of-scope code (circuits, arithmetic, HumanEval).
- Reintroduce hardcoded config, duplicate helpers, or personal paths.
- Delete `data/` or `results/` content. Keep `results/` gitignored.
