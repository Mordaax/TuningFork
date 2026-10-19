# Architecture

This document describes the internal design: the registry, the config schema,
the adapter contract, how it maps onto Kubeflow Trainer's CRDs/SDK, and the
deferred items.

## Design goals

1. **One config, many backends.** A single `TrainingConfig` is the contract;
   adapters translate it. Users never learn each tool's flag soup.
2. **ML-free core.** Importing the package, validating configs, and producing
   dry-run plans must work with only `pydantic` + `pyyaml`. Heavy ML libs are
   optional and only touched on a real run.
3. **Inspectable plans.** Dry-run yields a concrete, copy-pasteable command (or
   API call + generated script) so the plan can be reviewed, diffed, and used
   by a control plane before any GPU is allocated.
4. **Pluggable.** New backends register via a decorator *or* via Python
   entry-points (real plugin packaging).

## Registry design (`registry.py`)

- A module-level dict `_REGISTRY: {name -> BackendClass}`.
- `@register_backend(name=None)` is a class decorator. It validates the target
  is a `TrainerBackend` subclass, normalizes the name to lowercase, syncs the
  class's `name` attribute with the registry key, and refuses silent duplicate
  registrations.
- Lookups (`get_backend`, `get_backend_class`) are **case-insensitive** and
  whitespace-tolerant, and raise a `KeyError` that lists available backends.
- `get_backend` returns a **fresh instance** (adapters are cheap and stateless,
  so per-call instances avoid accidental shared state).
- **Two discovery paths:**
  1. *In-process*: importing `kubeflow_llm_trainer.backends` runs each adapter
     module, whose `@register_backend` populates the dict. This is how the four
     bundled backends load.
  2. *Entry-points*: `load_entry_point_backends()` scans the
     `kubeflow_llm_trainer.backends` entry-point group via
     `importlib.metadata` and registers any external plugin classes. A
     third-party package publishes a backend with:

     ```toml
     [project.entry-points."kubeflow_llm_trainer.backends"]
     mybackend = "my_pkg.adapter:MyBackend"
     ```

     The CLI calls this on `list-backends`, so installed plugins show up
     automatically. Bad/uninstallable plugins are skipped rather than crashing
     the host.

## Config schema (`config.py`)

All models use pydantic v2 with `extra="forbid"` so typos and stray keys are
**hard errors**, not silently ignored.

| Model | Key fields |
| --- | --- |
| `TrainingConfig` | `model_name_or_path` (required), `dataset`, `output_dir`, `precision`, `lora`, `hyperparams`, `extra` |
| `DatasetConfig` | `name_or_path` (required), `split`, `text_field`, `max_samples` |
| `LoRAConfig` | `enabled`, `r`, `alpha`, `dropout`, `target_modules`, `bias` |
| `TrainingHyperParams` | `learning_rate`, `num_epochs`, `per_device_batch_size`, `gradient_accumulation_steps`, `max_seq_len`, `warmup_ratio`, `weight_decay`, `seed` |
| `Precision` (enum) | `fp32 / fp16 / bf16 / int8 / int4` |

Validation highlights:

- Range constraints via `Field(ge=..., le=..., gt=...)` (e.g. `0 < lr`,
  `1 <= r <= 512`, `0 <= dropout <= 1`, `8 <= max_seq_len <= 131072`).
- Custom validators reject blank strings, empty `target_modules`, invalid
  `bias`.
- A **cross-field** model validator enforces the QLoRA invariant: quantized
  precision (`int4`/`int8`) requires `lora.enabled=True`.
- `extra: dict` is a deliberate escape hatch for backend-specific knobs (e.g.
  `tune_config`, `dataset_dir`) that don't belong in the unified schema.
- `from_dict` / `to_dict` give clean YAML↔model round-trips (enums serialize to
  their string values).

## Adapter contract (`base.py`)

`TrainerBackend` is an ABC. Subclasses set two class attributes and implement
one method:

```python
class MyBackend(TrainerBackend):
    name = "mybackend"
    required_dependency = "mylib"        # import name of the heavy dep

    def build_plan(self, cfg: TrainingConfig) -> TrainingPlan: ...
    def validate(self, cfg) -> None: ...   # optional, default no-op
```

Provided for free by the base class:

- `build_command(cfg)` — convenience alias returning `build_plan(cfg).command`.
- `dependency_available()` / `ensure_dependency()` — guard real execution; the
  latter raises `BackendDependencyError` with an actionable "pip install …"
  message.
- `train(cfg, dry_run=True)` — the orchestration path:
  `validate → build_plan → (dry_run? return plan : ensure_dependency → _execute)`.
  In dry-run it **never imports ML libs**.
- `_execute` — default raises (GPU runs deferred); a real backend overrides it.

`TrainingPlan` is a dataclass capturing *what would run*:

- `mode`: `"cli"` (argv in `command`) or `"api"` (Python call in `api_call`).
- `command` / `command_str` (shell-quoted) / `api_call`.
- `artifacts`: files the adapter would write first (e.g. a generated YAML or
  training script), as `{filename: contents}` — kept in-memory in dry-run.
- `resolved`: a flat, human-readable summary of the resolved knobs.
- `notes`: install hints and API equivalents.

### How each adapter maps the unified config

| Backend | Invocation | LoRA mapping | Notes |
| --- | --- | --- | --- |
| **TRL** | `trl sft --flag …` | `--use_peft --lora_r/alpha/dropout --lora_target_modules` | bf16/fp16/4bit/8bit flags; mirrors `SFTTrainer`. |
| **Unsloth** | `python unsloth_train.py` (generated script artifact) | `get_peft_model(r=…, lora_alpha=…)` | API-mode; requires LoRA (validator). Emits runnable script. |
| **LlamaFactory** | `llamafactory-cli train cfg.yaml` (generated YAML artifact) | `finetuning_type: lora`, `lora_rank/alpha/dropout/target` | `quantization_bit` for int4/int8. |
| **TorchTune** | `tune run <recipe> --config <cfg> k=v …` | `model.lora_rank/alpha/dropout`, `model.lora_attn_modules` | recipe = `lora_finetune_single_device`; `quantize_base=True` for int4. |

## Mapping onto Kubeflow Trainer

Kubeflow Trainer v2 model:

- **`TrainJob` (CRD)** — the user's job request (runtime ref, trainer overrides,
  resources, dataset/model config).
- **`TrainingRuntime` / `ClusterTrainingRuntime` (CRD)** — reusable, blueprint
  templates (the container image + JobSet topology + default command).
- **`TrainerClient` SDK** — Python client to submit/manage `TrainJob`s.

Integration points for this framework:

1. **Entrypoint.** A runtime image bundles `kubeflow_llm_trainer` and sets its
   trainer command to `klt train --backend $BACKEND /config/config.yaml`. The
   unified `TrainingConfig` is mounted (ConfigMap) or passed as a job arg.
2. **Runtime = backend selection.** Each `TrainingRuntime` can correspond to a
   backend (e.g. `trl-lora-runtime`, `unsloth-runtime`). Because the config is
   unified, switching runtime/backend needs **no config rewrite** — only the
   `--backend` value (or the runtime ref) changes.
3. **Control-plane validation/preview.** The SDK or an admission step can call
   `build_plan()`/`--dry-run --json` to validate the config and preview the
   exact command *before* scheduling a GPU pod — cheap, ML-free, offline.
4. **Field mapping (illustrative).** `TrainJob.spec.trainer.numNodes` /
   `resourcesPerNode` map to launcher concerns (torchrun/accelerate) layered
   on top of the per-backend command; `modelConfig`/`datasetConfig`-style
   inputs map onto `TrainingConfig.model_name_or_path` / `dataset`. The adapter
   owns the translation from unified knobs to the framework's flags; the runtime
   owns cluster topology.

This keeps the **trainer abstraction** (how a job becomes a command) cleanly
separated from the **runtime abstraction** (where/how the pod runs).

## Deferred

- **Real GPU LoRA runs** — `_execute` is intentionally a stub; the dry-run plan
  is the exact command to run manually. Wiring `_execute` to actually `subprocess`
  the command (CLI backends) or import + call the API (Unsloth) is the natural
  next step, gated by `ensure_dependency()`.
- **Real model/dataset downloads** — none happen here; the example points at
  small public artifacts for when a GPU is available.
- **Adapters as entry-point plugin packages** — the discovery mechanism and the
  `pyproject.toml` group are in place; splitting each backend into its own
  installable distribution is future packaging work.
- **Distributed / multi-node** — FSDP/DeepSpeed configs, `torchrun`/accelerate
  launchers, and Kubeflow JobSet multi-pod topology would be added as a launcher
  layer that wraps the per-backend command produced here.
```
