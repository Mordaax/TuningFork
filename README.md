# TuningFork

TuningFork takes one LLM fine-tuning configuration and prepares backend-specific plans for TRL, Unsloth, LlamaFactory and TorchTune. It validates the model, dataset and training settings, then shows the command each backend would use. It can also generate a Kubeflow TrainJob manifest.

This is a standalone proof of concept, not an integration into Kubeflow Trainer. The included demonstrations are dry runs; GPU training and model downloads have not been verified here.

---

## Problem

TRL, Unsloth, LlamaFactory and TorchTune take different settings and commands. Switching between them can mean rewriting the same fine-tuning job. TuningFork keeps the common model, dataset and training settings in one YAML file and translates them into a plan for the selected backend.

Each backend adapter validates the shared configuration and prepares its native command or API call. Dry runs print the plan without downloading models or requiring a GPU.

## Architecture (at a glance)

```
                 +------------------------+
   YAML config   |   TrainingConfig       |   (pydantic schema — the contract)
  ────────────►  |  model / dataset /     |
                 |  lora / hyperparams    |
                 +-----------+------------+
                             │
                    registry │ get_backend("trl")
                             ▼
        +----------------------------------------------+
        |            TrainerBackend (ABC)              |
        |  validate(cfg) · build_plan(cfg) · train()   |
        +----+-----------+------------+-----------+----+
             │           │            │           │
          TRL        Unsloth     LlamaFactory  TorchTune     ← adapters
        (trl sft)   (py API)   (llamafactory-cli) (tune run)
             │           │            │           │
             ▼           ▼            ▼           ▼
                     TrainingPlan  (command / api_call / artifacts)
                          │
                   dry-run│ print plan      real│ guarded on optional dep
                          ▼                     ▼
                    inspectable plan       actual training (deferred)
```

- **`config.py`** — pydantic models = the single contract every backend reads.
- **`registry.py`** — `@register_backend` decorator + lookup + entry-point discovery.
- **`base.py`** — `TrainerBackend` ABC and the `TrainingPlan` dataclass.
- **`backends/*.py`** — one adapter per tool.
- **`cli.py`** — the `klt` command.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for implementation details. The existing `klt` command remains available for the CLI and generated manifests.

## Possible use with Kubeflow Trainer

Kubeflow Trainer (v2) exposes a **`TrainJob` CRD** plus **`TrainingRuntime` /
`ClusterTrainingRuntime`** templates, and a Python SDK (`TrainerClient`). A
runtime ultimately launches a container that runs *some* training entrypoint.

A possible integration would use TuningFork as that container entrypoint:

1. A user submits a unified `TrainingConfig` (e.g. as a ConfigMap / job arg).
2. The Trainer runtime's container runs `klt train --backend <X> config.yaml`.
3. The adapter resolves the config into the backend's native command and runs it
   inside the runtime's torch/accelerate environment.

Because backends are pluggable, a `TrainingRuntime` can advertise *which*
backend image it provides (e.g. an "unsloth-runtime" vs a "trl-runtime"), and
the **same `TrainingConfig`** works across all of them. The `--dry-run` plan is
also useful for the SDK/CRD layer: it lets the control plane validate and
preview a job's concrete command before scheduling a GPU pod. The mapping to
specific CRD fields is detailed in ARCHITECTURE.md.

## Install

Core install is tiny and **ML-free** (only `pydantic` + `pyyaml`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pydantic, pyyaml, pytest
# or, as a package (provides the `klt` console script):
pip install -e .
```

Optional, per-backend heavy extras (only needed for **real** training):

```bash
pip install -e '.[trl]'          # trl peft transformers torch accelerate datasets
pip install -e '.[unsloth]'      # unsloth trl peft transformers torch datasets
pip install -e '.[llamafactory]' # llamafactory
pip install -e '.[torchtune]'    # torchtune torch
```

**Offline behaviour:** `pip install -r requirements.txt` needs network access
the first time (to fetch pydantic/pyyaml wheels) or a populated wheel cache /
local index. Once the four small wheels are cached, installs are offline. The
framework itself makes **no network calls** — `klt list-backends`, `validate`,
and `plan`/`--dry-run` are fully offline.

## Run

```bash
# Treat the repo as importable without installing:
export PYTHONPATH=.
alias klt='python -m kubeflow_llm_trainer.cli'   # or use the installed `klt`

klt list-backends
klt validate examples/lora_small.yaml
klt plan  --backend trl       examples/lora_small.yaml
klt plan  --backend torchtune examples/lora_small.yaml
klt train --backend trl --dry-run examples/lora_small.yaml
```

### Example dry-run output (TRL)

```
Backend : trl
Mode    : cli
Command :
  trl sft --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
    --dataset_name yelp_review_full --dataset_train_split train \
    --output_dir ./outputs/smollm2-lora --learning_rate 0.0002 \
    --num_train_epochs 1.0 --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 --max_seq_length 512 --warmup_ratio 0.03 \
    --weight_decay 0.0 --seed 42 --bf16 --use_peft --lora_r 8 --lora_alpha 16 \
    --lora_dropout 0.05 --lora_target_modules q_proj v_proj \
    --dataset_text_field text --max_train_samples 256
Resolved:
  tool: trl sft (SFTTrainer)   model: HuggingFaceTB/SmolLM2-135M-Instruct
  peft: True   precision: bf16   effective_batch_size: 8
```

### Example dry-run output (TorchTune)

```
Backend : torchtune
Mode    : cli
Command :
  tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
    output_dir=./outputs/smollm2-lora dtype=bf16 \
    dataset.source=yelp_review_full dataset.split=train seed=42 epochs=1 \
    batch_size=2 gradient_accumulation_steps=4 optimizer.lr=0.0002 \
    tokenizer.max_seq_len=512 model.lora_rank=8 model.lora_alpha=16 \
    model.lora_dropout=0.05 model.lora_attn_modules=[q_proj,v_proj]
```

## Example: planning a small LoRA fine-tune

[`examples/lora_small.yaml`](examples/lora_small.yaml) describes a LoRA SFT of
`SmolLM2-135M-Instruct` on a 256-sample slice of `yelp_review_full` — small
enough to plan without a GPU. The commands below show how a GPU run would be started; a complete training run has not been verified for this project:

```bash
pip install -e '.[trl]'                                   # GPU box with CUDA torch
klt plan  --backend trl examples/lora_small.yaml          # inspect the plan first
klt train --backend trl examples/lora_small.yaml          # omit --dry-run to execute
```

For the verified demonstration, use `plan` or `train --dry-run`.

## Tests

```bash
PYTHONPATH=. python -m pytest -q
```

The suite covers: registry register/resolve/list, config validation (valid +
invalid), each backend's dry-run command/args, dependency-guard errors, and a
CLI smoke test via subprocess. A test explicitly asserts that no ML library is
importable, enforcing the "core is ML-free" invariant.

## Deferred (documented, not implemented)

- **Real GPU fine-tuning runs** and **real model/dataset downloads.**
- Packaging each adapter as a separately-installable `entry_point` plugin
  (the group + hooks exist; the wiring is shown in `pyproject.toml`).
- Distributed / multi-node training (FSDP/DeepSpeed, torchrun, multi-pod).

## File tree

```
TuningFork/
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
├── requirements.txt
├── examples/
│   └── lora_small.yaml
├── kubeflow_llm_trainer/
│   ├── __init__.py
│   ├── config.py          # pydantic unified schema
│   ├── registry.py        # register/resolve/list + entry_points
│   ├── base.py            # TrainerBackend ABC + TrainingPlan
│   ├── cli.py             # `klt` CLI (argparse)
│   └── backends/
│       ├── __init__.py
│       ├── trl.py
│       ├── unsloth.py
│       ├── llamafactory.py
│       └── torchtune.py
└── tests/
    ├── conftest.py
    ├── test_registry.py
    ├── test_config.py
    ├── test_backends.py
    └── test_cli.py
```
