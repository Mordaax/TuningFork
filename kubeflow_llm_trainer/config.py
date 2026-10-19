"""Unified, backend-agnostic training configuration schema.

This module defines the *single source of truth* for describing an LLM
fine-tuning job. Every backend adapter (TRL, Unsloth, LlamaFactory,
TorchTune) consumes the same :class:`TrainingConfig` and translates it into
that tool's native invocation (CLI args or Python API call).

Only ``pydantic`` is required here. No ML libraries are imported, so this
schema can be loaded, validated and serialized in any environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Precision(str, Enum):
    """Numeric precision used for training weights / compute."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    # Quantized base-weight loading (QLoRA style).
    INT8 = "int8"
    INT4 = "int4"


class DatasetConfig(BaseModel):
    """Where the training data comes from and how it is shaped."""

    model_config = ConfigDict(extra="forbid")

    # A HuggingFace Hub id (e.g. "tatsu-lab/alpaca") OR a local path.
    name_or_path: str = Field(
        ..., description="HF dataset id or local path to the dataset."
    )
    split: str = Field("train", description="Dataset split to train on.")
    # Column that holds the prompt/text. Adapters fall back to sensible
    # per-tool defaults when this is None.
    text_field: Optional[str] = Field(
        None, description="Name of the text/prompt column."
    )
    # Optional cap on the number of samples (useful for smoke tests).
    max_samples: Optional[int] = Field(
        None, ge=1, description="Optional cap on number of training samples."
    )

    @field_validator("name_or_path", "split")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v


class LoRAConfig(BaseModel):
    """PEFT / LoRA adapter hyper-parameters.

    Set ``enabled=False`` to request full fine-tuning instead of LoRA.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Use LoRA/PEFT instead of full FT.")
    r: int = Field(16, ge=1, le=512, description="LoRA rank.")
    alpha: int = Field(32, ge=1, description="LoRA alpha (scaling).")
    dropout: float = Field(
        0.05, ge=0.0, le=1.0, description="LoRA dropout probability."
    )
    # None => let the backend pick its model-specific defaults.
    target_modules: Optional[list[str]] = Field(
        None, description="Module names to inject LoRA into."
    )
    bias: str = Field("none", description="LoRA bias handling: none|all|lora_only.")

    @field_validator("bias")
    @classmethod
    def _valid_bias(cls, v: str) -> str:
        allowed = {"none", "all", "lora_only"}
        if v not in allowed:
            raise ValueError(f"bias must be one of {sorted(allowed)}")
        return v

    @field_validator("target_modules")
    @classmethod
    def _nonempty_modules(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is not None:
            if len(v) == 0:
                raise ValueError(
                    "target_modules must be omitted or a non-empty list"
                )
            blanks = [m for m in v if not isinstance(m, str) or not m.strip()]
            if blanks:
                raise ValueError(
                    "target_modules entries must be non-empty strings"
                )
            if len(set(v)) != len(v):
                raise ValueError(
                    "target_modules must not contain duplicates"
                )
        return v

    @model_validator(mode="after")
    def _lora_alpha_sanity(self) -> "LoRAConfig":
        # alpha < r is almost always a configuration mistake (scaling = alpha/r
        # becomes < 1, neutering the adapter). Guard with an actionable message.
        if self.enabled and self.alpha < self.r:
            raise ValueError(
                f"lora.alpha ({self.alpha}) should be >= lora.r ({self.r}); "
                "alpha<r yields a scaling factor < 1 which usually defeats the "
                "purpose of the LoRA adapter. A common choice is alpha = 2*r."
            )
        return self


class TrainingHyperParams(BaseModel):
    """Optimizer / schedule / batching hyper-parameters."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(2e-4, gt=0.0, description="Optimizer LR.")
    num_epochs: float = Field(
        1.0, gt=0.0, description="Number of training epochs."
    )
    per_device_batch_size: int = Field(
        1, ge=1, description="Per-device train batch size."
    )
    gradient_accumulation_steps: int = Field(
        1, ge=1, description="Gradient accumulation steps."
    )
    max_seq_len: int = Field(
        512, ge=8, le=131072, description="Max sequence length (tokens)."
    )
    warmup_ratio: float = Field(
        0.03, ge=0.0, le=1.0, description="LR warmup ratio."
    )
    weight_decay: float = Field(0.0, ge=0.0, description="Weight decay.")
    seed: int = Field(42, description="Random seed.")

    @property
    def effective_batch_size(self) -> int:
        """Global batch size assuming a single device."""
        return self.per_device_batch_size * self.gradient_accumulation_steps


#: Top-level config fields that must not be smuggled in via ``extra``.
#: ``extra`` is a backend-specific escape hatch; duplicating a first-class
#: knob there is almost always a typo that would be silently ignored.
_RESERVED_EXTRA_KEYS = frozenset(
    {
        "model_name_or_path",
        "dataset",
        "output_dir",
        "precision",
        "lora",
        "hyperparams",
        # common hyperparam names users might wrongly nest under extra
        "learning_rate",
        "num_epochs",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "max_seq_len",
        "warmup_ratio",
        "weight_decay",
        "seed",
    }
)


class TrainingConfig(BaseModel):
    """Top-level unified config consumed by every backend adapter."""

    model_config = ConfigDict(extra="forbid")

    # A model name is the only strictly-required field. Everything else has
    # defaults tuned for a tiny LoRA smoke run.
    model_name_or_path: str = Field(
        ..., description="HF model id or local path to the base model."
    )
    dataset: DatasetConfig
    output_dir: str = Field(
        "./outputs", description="Directory for checkpoints / adapters."
    )
    precision: Precision = Field(
        Precision.BF16, description="Training precision."
    )
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    hyperparams: TrainingHyperParams = Field(
        default_factory=TrainingHyperParams
    )
    # Free-form, backend-specific escape hatch. Adapters MAY honor keys here.
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_name_or_path", "output_dir")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _check_quant_requires_lora(self) -> "TrainingConfig":
        # QLoRA invariant: quantized base weights only make sense with PEFT.
        if self.precision in (Precision.INT4, Precision.INT8) and not self.lora.enabled:
            raise ValueError(
                f"precision={self.precision.value} (quantized) requires "
                "lora.enabled=True (full fine-tuning of quantized weights "
                "is not supported)"
            )
        return self

    @model_validator(mode="after")
    def _check_lora_mutual_exclusion(self) -> "TrainingConfig":
        # Mutually-exclusive: LoRA-only knobs must not be set when doing a full
        # fine-tune. Catching this early gives a clear message instead of those
        # flags being silently dropped by the adapters.
        if not self.lora.enabled and self.lora.target_modules is not None:
            raise ValueError(
                "lora.target_modules is set but lora.enabled=False. "
                "target_modules only applies to LoRA/PEFT runs; either set "
                "lora.enabled=True or remove lora.target_modules."
            )
        return self

    @model_validator(mode="after")
    def _check_extra_no_reserved_keys(self) -> "TrainingConfig":
        # The `extra` escape hatch must not shadow first-class config fields.
        collisions = sorted(_RESERVED_EXTRA_KEYS & set(self.extra))
        if collisions:
            raise ValueError(
                "extra must not contain reserved config keys "
                f"{collisions}: these are first-class fields and would be "
                "ignored if nested under 'extra'. Set them at the top level "
                "(or inside 'hyperparams') instead."
            )
        return self

    # -- (de)serialization helpers -------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Build a config from a plain dict (e.g. parsed YAML)."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Plain, JSON/YAML-serializable dict (enums -> values)."""
        return self.model_dump(mode="json")
