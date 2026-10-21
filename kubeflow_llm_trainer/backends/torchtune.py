"""TorchTune backend adapter.

TorchTune is driven by ``tune run <recipe> --config <recipe.yaml> k=v ...``.
For LoRA single-device SFT the recipe is ``lora_finetune_single_device``. The
dry-run plan emits that CLI plus the key config overrides.
"""

from __future__ import annotations

from ..base import TrainerBackend, TrainingPlan
from ..config import Precision, TrainingConfig
from ..registry import register_backend

# TorchTune uses torch dtype strings.
_DTYPE = {
    Precision.BF16: "bf16",
    Precision.FP16: "fp16",
    Precision.FP32: "fp32",
    Precision.INT4: "bf16",
    Precision.INT8: "bf16",
}


@register_backend()
class TorchTuneBackend(TrainerBackend):
    name = "torchtune"
    required_dependency = "torchtune"

    def build_plan(self, cfg: TrainingConfig) -> TrainingPlan:
        hp = cfg.hyperparams

        recipe = (
            "lora_finetune_single_device"
            if cfg.lora.enabled
            else "full_finetune_single_device"
        )
        # A model-family config name; users override via extra["tune_config"].
        tune_config = cfg.extra.get("tune_config", "llama3/8B_lora_single_device")

        overrides: list[str] = [
            f"output_dir={cfg.output_dir}",
            f"dtype={_DTYPE[cfg.precision]}",
            f"dataset.source={cfg.dataset.name_or_path}",
            f"dataset.split={cfg.dataset.split}",
            f"seed={hp.seed}",
            f"epochs={int(hp.num_epochs) if hp.num_epochs.is_integer() else hp.num_epochs}",
            f"batch_size={hp.per_device_batch_size}",
            f"gradient_accumulation_steps={hp.gradient_accumulation_steps}",
            f"optimizer.lr={hp.learning_rate}",
            f"tokenizer.max_seq_len={hp.max_seq_len}",
        ]

        if cfg.lora.enabled:
            overrides += [
                f"model.lora_rank={cfg.lora.r}",
                f"model.lora_alpha={cfg.lora.alpha}",
                f"model.lora_dropout={cfg.lora.dropout}",
            ]
            if cfg.lora.target_modules:
                modules = ",".join(cfg.lora.target_modules)
                overrides.append(f"model.lora_attn_modules=[{modules}]")

        if cfg.precision == Precision.INT4:
            # QLoRA recipe variant in torchtune.
            recipe = "lora_finetune_single_device"
            overrides.append("model.quantize_base=True")

        cmd = ["tune", "run", recipe, "--config", tune_config, *overrides]

        return TrainingPlan(
            backend=self.name,
            mode="cli",
            command=cmd,
            resolved={
                "tool": "tune run",
                "recipe": recipe,
                "config": tune_config,
                "model": cfg.model_name_or_path,
                "dataset": cfg.dataset.name_or_path,
                "dtype": _DTYPE[cfg.precision],
                "effective_batch_size": hp.effective_batch_size,
            },
            notes=[
                "model_name_or_path informs checkpoint config; TorchTune selects "
                "weights via its recipe/config (override with extra.tune_config).",
                "Install with: pip install 'tuningfork[torchtune]'",
            ],
        )
