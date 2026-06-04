# Failure by Interference

Code for the paper *“Failure by Interference: Language Models Make Balanced
Parentheses Errors When Faulty Mechanisms Overshadow Sound Ones”* (RASTEER).

We study why language models (124M–7B) fail at the **balanced-parentheses** task —
predicting the correct number of closing parentheses in code — by looking at the
attention heads and FF neurons that contribute directly to the final logit. Some
components implement *sound* mechanisms (reliable across sub-tasks); others are
*faulty* (noisy promotion). We then introduce **RASTEER**, which Ranks components by
reliability and STEERs the reliable ones up to improve performance.

Models studied: GPT-2 {Small, Medium, Large, XL}, CodeLlama-7b, Llama2-7b, Pythia-6.9b.

## Layout

```
src/lmfi/            # the package (importable as `lmfi`)
  io.py              # read_json / save_file / create_results_dir / clear_cache
  data/              # synthesis (seeded), loaders, SubTask (file suffix _0..3 <-> 1..4-paren)
  models/            # load_model + per-arch HookedTransformer configs
  analysis/          # accuracy, per-component projection, generalizability (Table 1)
  steering/          # interventions, component ranking, the unified steering driver
  viz/               # figures (Fig 1, 2, 3/8)
configs/             # one JSON per experiment + the model registry (models.json)
experiments/         # thin, config-driven entry points (one dir per stage)
data/                # the synthesized dataset (committed, canonical)
results/             # all outputs (gitignored)
tests/               # regression notes
```

Every stage is a thin `experiments/NN_name/run.py` that loads a config from
`configs/` and calls into `src/lmfi/`. Knobs (models, sub-tasks, metrics, paths,
multiplier grid) live in the configs/CLI — not in code.

## Setup

Tested with Python 3.10 (the original was 3.9.9).

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the `lmfi` package (src layout)
```

Notes:
- `requirements.txt` pins `transformer-lens` to a specific git commit, so the install
  needs network access.
- It also lists `human_eval` (an editable git package used only for the out-of-scope
  HumanEval evaluation). If it fails to build, it is safe to skip — drop that line and
  reinstall; nothing in this repo imports it.
- If you don’t `pip install -e .`, run the experiments with `PYTHONPATH=src` instead.

### Models

`configs/models.json` is the model registry. `cache` is `null` by default, meaning the
default Hugging Face hub cache is used; set it per machine if your weights live
elsewhere. Gated models (Llama-2, CodeLlama) require accepting their license / an HF
token. Most scripts run offline once weights are cached (`HF_HUB_OFFLINE=1`).

## Data

The committed `data/<model>/{train,dev,test}_labeled_last_paren_{0..3}.json` is the
**canonical** dataset (350 / 150 / 150 examples per sub-task). You normally do not need
to regenerate it. To regenerate it reproducibly (seeded):

```
python experiments/00_synthesize/run.py            # all models
python experiments/00_synthesize/run.py --models gpt2
```

Each record is `{"prompt": "#print the string 486\nprint(486", "label": ")", "label_idx": int}`;
the file suffix `n` corresponds to the `(n+1)`-paren sub-task.

## Running the pipeline

Stages feed forward in this order. GPU stages run model forwards; analysis/figure
stages are CPU-only. Most stages take `--models <...>` to restrict the model set
(default = the paper’s set in each config).

```
# 1. Baseline accuracy (Table 3 / Appendix A)        [GPU]
python experiments/01_accuracy/run.py --models gpt2

# 2. Per-component logit projections — the hub        [GPU]
python experiments/02_projection/run.py --component both --models gpt2

# 3. Generalizability: precision/recall/F1/accuracy + generalization counts (Table 1) [CPU]
python experiments/03_generalizability/run.py --component both --models gpt2
python experiments/03_generalizability/plot_acc_distribution.py --models CodeLlama-7b-hf   # Fig 1

# 4. Component figures                                 [CPU]
python experiments/04_component_figures/precision_recall.py --preset fig3   # Fig 3 (+ fig8)
python experiments/04_component_figures/plot_dual_neuron.py                 # Fig 2 (CodeLlama L19N11)

# 5. Steering (RASTEER) — Fig 4 + Appendix D
python experiments/05_steering/run.py --component attn --mode sweep  --split dev --models gpt2   [GPU]
python experiments/05_steering/run.py --component attn --mode analyze                            [CPU]
python experiments/05_steering/run.py --component attn --mode apply  --models gpt2               [GPU]
```

The steering driver is unified: `--component {attn,neuron,both}` × `--mode
{sweep,analyze,apply}`. `sweep` runs the full coeff × #components grid on a split;
`analyze` picks the best dev coefficient per configuration (→ `coeffs_<component>.json`);
`apply` applies that coefficient on the test set.

### SLURM

Each GPU stage has a matching `*.slurm` script (`experiments/01_accuracy/accuracy.slurm`,
`02_projection/projection.slurm`, `05_steering/steering.slurm`). They activate the env,
set `PYTHONPATH`/offline caches, and request an A100; override the GPU for small models,
e.g. `--gres=gpu:1g.10gb:1` for a gpt2 canary. Example:

```
mkdir -p results/slurm-logs
sbatch --export=ALL,MODELS="gpt2",COMPONENT="both" experiments/02_projection/projection.slurm
```

## Figure → script map

| Artifact | Produced by |
| --- | --- |
| Table 3 / App A (accuracy) | `experiments/01_accuracy/run.py` |
| Table 1 (generalization counts) | `experiments/03_generalizability/run.py` |
| Fig 1 (head accuracy distribution, CodeLlama) | `experiments/03_generalizability/plot_acc_distribution.py` |
| Fig 2 (dual-sign neuron, CodeLlama L19N11) | `experiments/04_component_figures/plot_dual_neuron.py` |
| Fig 3 + Fig 8 (precision-recall) | `experiments/04_component_figures/precision_recall.py` (`--preset fig3` / `fig8`) |
| Fig 4(a,b,d,e), App D (RASTEER) | `experiments/05_steering/run.py` |

Out of scope (no code in this repo): the circuit-discovery baseline (§4.2 / App C),
the arithmetic task (§5.4.3), and HumanEval — so Fig 4(c)/(f) and Fig 5 (which depend
on the circuit baseline) are not reproducible here.

## Notes on reproducibility

The dataset was originally synthesized without a fixed seed, so re-synthesized data can
differ slightly from the committed copy (a few examples per sub-task); this does not
affect any claim in the paper and the committed `data/` is the ground truth used for the
reported numbers.
