"""LlamaFactory backend adapter.

LlamaFactory is driven by ``llamafactory-cli train <config.yaml>``. The dry-run
plan generates that YAML config as an artifact and emits the CLI command.
"""

from __future__ import annotations

from ..base import TrainerBackend, TrainingPlan
from ..config import Precision, TrainingConfig
from ..registry import register_backend

# LlamaFactory expects its own compute_type / quantization knobs.
_COMPUTE_TYPE = {
    Precision.BF16: "bf16",
    Precision.FP16: "fp16",
    Precision.FP32: "fp32",
    Precision.INT4: "bf16",  # quantized base weights, bf16 compute
    Precision.INT8: "bf16",
}


@register_backend()
class LlamaFactoryBackend(TrainerBackend):
    name = "llamafactory"
    required_dependency = "llamafactory"

    def build_plan(self, cfg: TrainingConfig) -> TrainingPlan:
        hp = cfg.hyperparams

        lf: dict[str, object] = {
            "model_name_or_path": cfg.model_name_or_path,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora" if cfg.lora.enabled else "full",
            "dataset": cfg.dataset.name_or_path,
            "dataset_dir": cfg.extra.get("dataset_dir", "data"),
            "cutoff_len": hp.max_seq_len,
            "output_dir": cfg.output_dir,
            "per_device_train_batch_size": hp.per_device_batch_size,
            "gradient_accumulation_steps": hp.gradient_accumulation_steps,
            "learning_rate": hp.learning_rate,
            "num_train_epochs": hp.num_epochs,
            "warmup_ratio": hp.warmup_ratio,
            "weight_decay": hp.weight_decay,
            "seed": hp.seed,
            "compute_type": _COMPUTE_TYPE[cfg.precision],
            "logging_steps": cfg.extra.get("logging_steps", 10),
        }

        if cfg.lora.enabled:
            lf["lora_rank"] = cfg.lora.r
            lf["lora_alpha"] = cfg.lora.alpha
            lf["lora_dropout"] = cfg.lora.dropout
            lf["lora_target"] = (
                ",".join(cfg.lora.target_modules)
                if cfg.lora.target_modules
                else "all"
            )

        if cfg.precision == Precision.INT4:
            lf["quantization_bit"] = 4
        elif cfg.precision == Precision.INT8:
            lf["quantization_bit"] = 8

        if cfg.dataset.max_samples is not None:
            lf["max_samples"] = cfg.dataset.max_samples

        # Render a stable, readable YAML (no external dep needed here).
        yaml_text = _render_yaml(lf)
        config_name = "llamafactory_sft.yaml"

        return TrainingPlan(
            backend=self.name,
            mode="cli",
            command=["llamafactory-cli", "train", config_name],
            artifacts={config_name: yaml_text},
            resolved={
                "tool": "llamafactory-cli train",
                "model": cfg.model_name_or_path,
                "dataset": cfg.dataset.name_or_path,
                "finetuning_type": lf["finetuning_type"],
                "compute_type": lf["compute_type"],
                "quantization_bit": lf.get("quantization_bit"),
                "effective_batch_size": hp.effective_batch_size,
            },
            notes=[
                f"Dry-run would write '{config_name}' then run: "
                f"llamafactory-cli train {config_name}",
                "Install with: pip install 'tuningfork[llamafactory]'",
            ],
        )


def _render_yaml(d: dict[str, object]) -> str:
    """Render a flat scalar dict as valid, deterministic YAML.

    Uses PyYAML (a core dependency of this package) so that string values
    containing YAML-special characters (e.g. a ``model: name`` with a colon)
    are correctly quoted and floats in scientific notation are emitted as YAML
    floats rather than strings. ``sort_keys=False`` preserves insertion order
    for deterministic, readable output (and stable tests).

    Previously this was a hand-rolled emitter that did **not** quote strings or
    normalise floats, which produced *invalid* YAML for values like
    ``org/model: weird`` and silently turned ``2e-05`` into a string instead of
    a number. Delegating to PyYAML fixes both classes of bug.
    """
    import yaml

    return yaml.safe_dump(
        d,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
