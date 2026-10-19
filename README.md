# Kubeflow LLM Trainer — Pluggable Multi-Framework Backend

A proof-of-concept that gives **Kubeflow Trainer** a *pluggable* training-backend
system so a single, unified fine-tuning config can be executed by **any** of
several LLM fine-tuning tools — **TRL**, **Unsloth**, **LlamaFactory**, and
**TorchTune** — behind one interface.

> GSoC 2026 portfolio PoC. Real GPU runs and model downloads are intentionally
> **deferred** (documented below). Everything here imports, validates, plans,
> and tests with **zero ML libraries installed**.

---

## Problem

Kubeflow Trainer's training logic is effectively tied to one backend. As the
LLM fine-tuning ecosystem fragmented (TRL/SFTTrainer, Unsloth's fused kernels,
LlamaFactory's CLI, Meta's TorchTune recipes), users want to pick the tool that
best fits their model/hardware **without** rewriting their job spec each time.

This project introduces a **backend abstraction**: describe the job once in a
unified schema, then route it to whichever backend you want. Each backend
*adapter* knows how to translate the unified config into that tool's native
invocation (CLI args or Python API). A `--dry-run` mode prints the **exact
command/plan** that would execute — no GPU, no downloads, no ML deps required.

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

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the deep dive.

## How it plugs into Kubeflow Trainer

Kubeflow Trainer (v2) exposes a **`TrainJob` CRD** plus **`TrainingRuntime` /
`ClusterTrainingRuntime`** templates, and a Python SDK (`TrainerClient`). A
runtime ultimately launches a container that runs *some* training entrypoint.

This framework slots in at exactly that entrypoint:

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

## Example: tiny LoRA fine-tune

[`examples/lora_small.yaml`](examples/lora_small.yaml) describes a LoRA SFT of
`SmolLM2-135M-Instruct` on a 256-sample slice of `yelp_review_full` — small
enough for a single modest GPU. To run it **for real** on a GPU:

```bash
pip install -e '.[trl]'                                   # GPU box with CUDA torch
klt plan  --backend trl examples/lora_small.yaml          # inspect the plan first
klt train --backend trl examples/lora_small.yaml          # omit --dry-run to execute
```

On CPU-only / this PoC, keep `--dry-run`.

## Tests

```bash
PYTHONPATH=. python -m pytest -q     # 57 tests, all green with NO ML libs
```

The suite covers: registry register/resolve/list, config validation (valid +
invalid), each backend's dry-run command/args, dependency-guard errors, and a
CLI smoke test via subprocess. A test explicitly asserts that no ML library is
importable, enforcing the "core is ML-free" invariant.

## GSoC angle

This is a focused slice of a real Kubeflow Trainer need: **decouple the trainer
from any single fine-tuning library.** The PoC demonstrates the hard parts —
a stable unified schema, a clean adapter contract, a registry with both
in-process and entry-point plugin discovery, and dry-run planning that the
control plane can use for validation — while leaving GPU execution as a
well-scoped follow-up. It is intentionally dependency-light so it can live in
CI and be reasoned about without a GPU.

## Deferred (documented, not implemented)

- **Real GPU fine-tuning runs** and **real model/dataset downloads.**
- Packaging each adapter as a separately-installable `entry_point` plugin
  (the group + hooks exist; the wiring is shown in `pyproject.toml`).
- Distributed / multi-node training (FSDP/DeepSpeed, torchrun, multi-pod).

## File tree

```
04-kubeflow-llm-trainer/
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
