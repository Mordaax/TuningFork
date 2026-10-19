"""Starter-config scaffolding for ``klt init``.

Emits a well-commented unified ``TrainingConfig`` YAML that a user can edit and
feed straight into ``klt validate`` / ``klt plan`` / ``klt manifest``. The
generated text is guaranteed to be a *valid* config (it is validated against the
schema by the test-suite and can be by the caller).

Only the standard library is used here; no ML libraries, no PyYAML required to
*produce* the template (it is a hand-authored, commented string).
"""

from __future__ import annotations

from .registry import get_backend, list_backends

# Per-backend hints injected into the scaffold so the starter config is tuned
# to the chosen tool's conventions.
_BACKEND_HINTS: dict[str, str] = {
    "trl": "TRL drives `trl sft` (SFTTrainer). bf16 + LoRA is a good default.",
    "unsloth": "Unsloth REQUIRES lora.enabled=true (LoRA/QLoRA only).",
    "llamafactory": "LlamaFactory uses `llamafactory-cli train`; int4/int8 set quantization_bit.",
    "torchtune": "TorchTune uses `tune run`; override the recipe via extra.tune_config.",
}


def starter_config_yaml(backend: str | None = None) -> str:
    """Return a commented starter ``TrainingConfig`` as YAML text.

    If ``backend`` is given it is validated against the registry and a
    backend-specific hint is included. ``int4`` precision is *commented out* by
    default so the scaffold validates as-is (int4 requires LoRA, which is the
    default here anyway, but we keep the simplest valid config uncommented).
    """
    hint_line = ""
    if backend:
        # Raises KeyError (with available list) for an unknown backend.
        get_backend(backend)
        hint = _BACKEND_HINTS.get(backend, "")
        hint_line = f"# Backend: {backend} — {hint}\n"
        if backend == "unsloth":
            lora_enabled = "true   # Unsloth requires LoRA"
        else:
            lora_enabled = "true"
    else:
        lora_enabled = "true"

    backends = ", ".join(list_backends())

    return f"""\
# ---------------------------------------------------------------------------
# Kubeflow LLM Trainer — unified training config (starter template)
#
# Edit the values below, then:
#   klt validate this.yaml
#   klt plan     --backend trl this.yaml
#   klt manifest --backend trl --kind trainjob this.yaml > job.yaml
#
# Available backends: {backends}
{hint_line}# ---------------------------------------------------------------------------

# REQUIRED: a HuggingFace Hub model id or a local path to the base model.
model_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct

# Where checkpoints / LoRA adapters are written.
output_dir: ./outputs/my-finetune

# Training precision: fp32 | fp16 | bf16 | int8 | int4
# (int4/int8 are QLoRA-style and REQUIRE lora.enabled: true)
precision: bf16

dataset:
  # REQUIRED: a HuggingFace dataset id or a local path.
  name_or_path: yelp_review_full
  split: train
  # Column holding the text/prompt (omit to use the backend's default).
  text_field: text
  # Optional cap on samples — great for a fast smoke test.
  max_samples: 256

lora:
  # Set to false for a full fine-tune (uses much more GPU memory).
  enabled: {lora_enabled}
  r: 16              # LoRA rank (1..512)
  alpha: 32          # scaling; a common rule of thumb is alpha = 2 * r
  dropout: 0.05      # 0.0 .. 1.0
  # Modules to inject LoRA into (omit to let the backend pick model defaults):
  # target_modules:
  #   - q_proj
  #   - v_proj
  bias: none         # none | all | lora_only

hyperparams:
  learning_rate: 0.0002
  num_epochs: 1
  per_device_batch_size: 2
  gradient_accumulation_steps: 4   # effective batch = batch * accum
  max_seq_len: 512                 # 8 .. 131072
  warmup_ratio: 0.03
  weight_decay: 0.0
  seed: 42

# Free-form, backend-specific escape hatch (must NOT duplicate fields above).
# Examples:
#   extra:
#     tune_config: llama3/8B_lora_single_device   # torchtune recipe config
#     dataset_dir: data                            # llamafactory data dir
extra: {{}}
"""
