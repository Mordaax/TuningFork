"""Unified config schema: valid and invalid cases."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kubeflow_llm_trainer.config import (
    DatasetConfig,
    LoRAConfig,
    Precision,
    TrainingConfig,
    TrainingHyperParams,
)


def _minimal_dict() -> dict:
    return {
        "model_name_or_path": "sshleifer/tiny-gpt2",
        "dataset": {"name_or_path": "imdb"},
    }


# -- valid cases -------------------------------------------------------


def test_minimal_config_uses_defaults():
    cfg = TrainingConfig.from_dict(_minimal_dict())
    assert cfg.precision == Precision.BF16
    assert cfg.lora.enabled is True
    assert cfg.lora.r == 16
    assert cfg.hyperparams.learning_rate == 2e-4
    assert cfg.output_dir == "./outputs"


def test_roundtrip_to_dict_is_json_safe():
    cfg = TrainingConfig.from_dict(_minimal_dict())
    d = cfg.to_dict()
    assert d["precision"] == "bf16"  # enum serialized to value
    # Round-trips back into an identical config.
    assert TrainingConfig.from_dict(d) == cfg


def test_effective_batch_size():
    hp = TrainingHyperParams(per_device_batch_size=2, gradient_accumulation_steps=8)
    assert hp.effective_batch_size == 16


def test_qlora_int4_with_lora_is_valid():
    d = _minimal_dict()
    d["precision"] = "int4"
    d["lora"] = {"enabled": True}
    cfg = TrainingConfig.from_dict(d)
    assert cfg.precision == Precision.INT4


# -- invalid cases -----------------------------------------------------


def test_missing_model_name_raises():
    with pytest.raises(ValidationError):
        TrainingConfig.from_dict({"dataset": {"name_or_path": "imdb"}})


def test_missing_dataset_raises():
    with pytest.raises(ValidationError):
        TrainingConfig.from_dict({"model_name_or_path": "x"})


def test_blank_model_name_raises():
    d = _minimal_dict()
    d["model_name_or_path"] = "   "
    with pytest.raises(ValidationError):
        TrainingConfig.from_dict(d)


def test_unknown_field_is_rejected():
    d = _minimal_dict()
    d["totally_unknown"] = 1
    with pytest.raises(ValidationError):
        TrainingConfig.from_dict(d)


def test_negative_learning_rate_rejected():
    with pytest.raises(ValidationError):
        TrainingHyperParams(learning_rate=-1.0)


def test_lora_rank_out_of_range_rejected():
    with pytest.raises(ValidationError):
        LoRAConfig(r=0)
    with pytest.raises(ValidationError):
        LoRAConfig(r=10_000)


def test_lora_dropout_out_of_range_rejected():
    with pytest.raises(ValidationError):
        LoRAConfig(dropout=1.5)


def test_invalid_bias_rejected():
    with pytest.raises(ValidationError):
        LoRAConfig(bias="sometimes")


def test_empty_target_modules_rejected():
    with pytest.raises(ValidationError):
        LoRAConfig(target_modules=[])


def test_invalid_precision_value_rejected():
    d = _minimal_dict()
    d["precision"] = "fp9"
    with pytest.raises(ValidationError):
        TrainingConfig.from_dict(d)


def test_quantized_precision_requires_lora():
    d = _minimal_dict()
    d["precision"] = "int4"
    d["lora"] = {"enabled": False}
    with pytest.raises(ValidationError) as exc:
        TrainingConfig.from_dict(d)
    assert "quantized" in str(exc.value).lower()


def test_blank_dataset_name_rejected():
    with pytest.raises(ValidationError):
        DatasetConfig(name_or_path="")


def test_max_seq_len_too_small_rejected():
    with pytest.raises(ValidationError):
        TrainingHyperParams(max_seq_len=1)


# -- new robustness rules ----------------------------------------------


def test_lora_alpha_less_than_r_rejected():
    with pytest.raises(ValidationError) as exc:
        LoRAConfig(r=32, alpha=8)
    assert "alpha" in str(exc.value).lower()


def test_lora_alpha_equal_r_allowed():
    cfg = LoRAConfig(r=16, alpha=16)
    assert cfg.alpha == 16


def test_lora_alpha_check_skipped_when_disabled():
    # Full FT: alpha<r is irrelevant, must not error.
    cfg = LoRAConfig(enabled=False, r=32, alpha=8)
    assert cfg.enabled is False


def test_target_modules_with_full_finetune_rejected():
    d = _minimal_dict()
    d["lora"] = {"enabled": False, "target_modules": ["q_proj"]}
    with pytest.raises(ValidationError) as exc:
        TrainingConfig.from_dict(d)
    assert "target_modules" in str(exc.value)


def test_target_modules_duplicates_rejected():
    with pytest.raises(ValidationError):
        LoRAConfig(target_modules=["q_proj", "q_proj"])


def test_target_modules_blank_entry_rejected():
    with pytest.raises(ValidationError):
        LoRAConfig(target_modules=["q_proj", "  "])


@pytest.mark.parametrize(
    "bad_key", ["learning_rate", "model_name_or_path", "seed", "lora"]
)
def test_extra_reserved_key_rejected(bad_key):
    d = _minimal_dict()
    d["extra"] = {bad_key: "whatever"}
    with pytest.raises(ValidationError) as exc:
        TrainingConfig.from_dict(d)
    assert "reserved" in str(exc.value).lower()


def test_extra_non_reserved_keys_allowed():
    d = _minimal_dict()
    d["extra"] = {"tune_config": "llama3/8B", "dataset_dir": "data"}
    cfg = TrainingConfig.from_dict(d)
    assert cfg.extra["tune_config"] == "llama3/8B"
