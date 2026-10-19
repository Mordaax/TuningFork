"""TRL (Transformer Reinforcement Learning) backend adapter.

Maps the unified config onto TRL's ``trl sft`` CLI, which wraps
``SFTTrainer``. In dry-run mode we emit the exact ``trl sft ...`` argv.
"""

from __future__ import annotations

from ..base import TrainerBackend, TrainingPlan
from ..config import Precision, TrainingConfig
from ..registry import register_backend


@register_backend()
class TRLBackend(TrainerBackend):
    name = "trl"
    required_dependency = "trl"

    def build_plan(self, cfg: TrainingConfig) -> TrainingPlan:
        hp = cfg.hyperparams
        cmd: list[str] = [
            "trl",
            "sft",
            "--model_name_or_path",
            cfg.model_name_or_path,
            "--dataset_name",
            cfg.dataset.name_or_path,
            "--dataset_train_split",
            cfg.dataset.split,
            "--output_dir",
            cfg.output_dir,
            "--learning_rate",
            repr(hp.learning_rate),
            "--num_train_epochs",
            repr(hp.num_epochs),
            "--per_device_train_batch_size",
            str(hp.per_device_batch_size),
            "--gradient_accumulation_steps",
            str(hp.gradient_accumulation_steps),
            "--max_seq_length",
            str(hp.max_seq_len),
            "--warmup_ratio",
            repr(hp.warmup_ratio),
            "--weight_decay",
            repr(hp.weight_decay),
            "--seed",
            str(hp.seed),
        ]

        # Precision flags.
        if cfg.precision == Precision.BF16:
            cmd += ["--bf16"]
        elif cfg.precision == Precision.FP16:
            cmd += ["--fp16"]
        elif cfg.precision in (Precision.INT4, Precision.INT8):
            bits = "4" if cfg.precision == Precision.INT4 else "8"
            cmd += ["--load_in_4bit"] if bits == "4" else ["--load_in_8bit"]

        # LoRA / PEFT flags.
        if cfg.lora.enabled:
            cmd += [
                "--use_peft",
                "--lora_r",
                str(cfg.lora.r),
                "--lora_alpha",
                str(cfg.lora.alpha),
                "--lora_dropout",
                repr(cfg.lora.dropout),
            ]
            if cfg.lora.target_modules:
                cmd += ["--lora_target_modules", *cfg.lora.target_modules]

        if cfg.dataset.text_field:
            cmd += ["--dataset_text_field", cfg.dataset.text_field]
        if cfg.dataset.max_samples is not None:
            cmd += ["--max_train_samples", str(cfg.dataset.max_samples)]

        return TrainingPlan(
            backend=self.name,
            mode="cli",
            command=cmd,
            resolved={
                "tool": "trl sft (SFTTrainer)",
                "model": cfg.model_name_or_path,
                "dataset": cfg.dataset.name_or_path,
                "peft": cfg.lora.enabled,
                "precision": cfg.precision.value,
                "effective_batch_size": hp.effective_batch_size,
            },
            notes=[
                "Equivalent Python API: trl.SFTTrainer(model=..., args=SFTConfig(...), "
                "peft_config=peft.LoraConfig(...))",
                "Install with: pip install 'kubeflow-llm-trainer[trl]'",
            ],
        )
