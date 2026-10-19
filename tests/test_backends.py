"""Backend adapters: dry-run plans produce the expected commands/args.

These are the core tests. They MUST pass with no ML libraries installed.
"""

from __future__ import annotations

import pytest

from kubeflow_llm_trainer.base import BackendDependencyError, TrainingPlan
from kubeflow_llm_trainer.config import (
    DatasetConfig,
    LoRAConfig,
    Precision,
    TrainingConfig,
)
from kubeflow_llm_trainer.registry import get_backend


def test_core_runs_without_ml_libs(ml_libs_absent):
    # The whole point: the suite documents that ML libs are absent.
    assert ml_libs_absent, "These tests are meant to run with NO ML libraries."


# -- TRL ---------------------------------------------------------------


def test_trl_plan(sample_config):
    plan = get_backend("trl").build_plan(sample_config)
    assert isinstance(plan, TrainingPlan)
    assert plan.mode == "cli"
    cmd = plan.command
    assert cmd[:2] == ["trl", "sft"]
    assert "--model_name_or_path" in cmd
    assert "HuggingFaceTB/SmolLM2-135M-Instruct" in cmd
    assert "--dataset_name" in cmd
    assert "yelp_review_full" in cmd
    # LoRA flags present.
    assert "--use_peft" in cmd
    assert cmd[cmd.index("--lora_r") + 1] == "8"
    assert cmd[cmd.index("--lora_alpha") + 1] == "16"
    # bf16 precision flag.
    assert "--bf16" in cmd
    # target modules expanded.
    tm_idx = cmd.index("--lora_target_modules")
    assert cmd[tm_idx + 1 : tm_idx + 3] == ["q_proj", "v_proj"]
    # text field + max samples.
    assert cmd[cmd.index("--dataset_text_field") + 1] == "text"
    assert cmd[cmd.index("--max_train_samples") + 1] == "256"


def test_trl_int4_uses_load_in_4bit(sample_config):
    cfg = sample_config.model_copy(update={"precision": Precision.INT4})
    cmd = get_backend("trl").build_plan(cfg).command
    assert "--load_in_4bit" in cmd
    assert "--bf16" not in cmd


def test_trl_full_finetune_omits_peft():
    cfg = TrainingConfig(
        model_name_or_path="m",
        dataset=DatasetConfig(name_or_path="d"),
        lora=LoRAConfig(enabled=False),
    )
    cmd = get_backend("trl").build_plan(cfg).command
    assert "--use_peft" not in cmd


# -- Unsloth -----------------------------------------------------------


def test_unsloth_plan_emits_script(sample_config):
    plan = get_backend("unsloth").build_plan(sample_config)
    assert plan.mode == "api"
    assert plan.command == ["python", "unsloth_train.py"]
    assert "unsloth_train.py" in plan.artifacts
    script = plan.artifacts["unsloth_train.py"]
    assert "FastLanguageModel.from_pretrained" in script
    assert "get_peft_model" in script
    assert "r=8" in script
    assert "lora_alpha=16" in script
    assert "HuggingFaceTB/SmolLM2-135M-Instruct" in script
    assert "max_seq_length=512" in script
    assert plan.api_call and "FastLanguageModel" in plan.api_call


def test_unsloth_int4_sets_load_in_4bit(sample_config):
    cfg = sample_config.model_copy(update={"precision": Precision.INT4})
    plan = get_backend("unsloth").build_plan(cfg)
    assert plan.resolved["load_in_4bit"] is True
    assert "load_in_4bit=True" in plan.artifacts["unsloth_train.py"]


def test_unsloth_rejects_full_finetune():
    cfg = TrainingConfig(
        model_name_or_path="m",
        dataset=DatasetConfig(name_or_path="d"),
        lora=LoRAConfig(enabled=False),
    )
    with pytest.raises(ValueError):
        get_backend("unsloth").validate(cfg)


# -- LlamaFactory ------------------------------------------------------


def test_llamafactory_plan(sample_config):
    plan = get_backend("llamafactory").build_plan(sample_config)
    assert plan.mode == "cli"
    assert plan.command == ["llamafactory-cli", "train", "llamafactory_sft.yaml"]
    yaml_text = plan.artifacts["llamafactory_sft.yaml"]
    assert "stage: sft" in yaml_text
    assert "finetuning_type: lora" in yaml_text
    assert "lora_rank: 8" in yaml_text
    assert "lora_alpha: 16" in yaml_text
    assert "lora_target: q_proj,v_proj" in yaml_text
    assert "cutoff_len: 512" in yaml_text
    assert "compute_type: bf16" in yaml_text
    assert "model_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct" in yaml_text


def test_llamafactory_int8_quantization(sample_config):
    cfg = sample_config.model_copy(update={"precision": Precision.INT8})
    yaml_text = get_backend("llamafactory").build_plan(cfg).artifacts[
        "llamafactory_sft.yaml"
    ]
    assert "quantization_bit: 8" in yaml_text


def test_llamafactory_artifact_is_valid_yaml(sample_config):
    # Regression: the artifact must be parseable YAML and preserve values.
    import yaml

    yaml_text = get_backend("llamafactory").build_plan(sample_config).artifacts[
        "llamafactory_sft.yaml"
    ]
    parsed = yaml.safe_load(yaml_text)
    assert parsed["stage"] == "sft"
    assert parsed["lora_rank"] == 8


def test_llamafactory_yaml_handles_special_chars():
    # Regression for the hand-rolled emitter: a colon in a value and a
    # scientific-notation LR must round-trip correctly (no broken/string YAML).
    import yaml

    from kubeflow_llm_trainer.config import TrainingHyperParams

    cfg = TrainingConfig(
        model_name_or_path="org/model: weird",
        dataset=DatasetConfig(name_or_path="d"),
        hyperparams=TrainingHyperParams(learning_rate=2e-5),
    )
    yaml_text = get_backend("llamafactory").build_plan(cfg).artifacts[
        "llamafactory_sft.yaml"
    ]
    parsed = yaml.safe_load(yaml_text)
    assert parsed["model_name_or_path"] == "org/model: weird"
    assert isinstance(parsed["learning_rate"], float)
    assert parsed["learning_rate"] == 2e-5


# -- TorchTune ---------------------------------------------------------


def test_torchtune_plan(sample_config):
    plan = get_backend("torchtune").build_plan(sample_config)
    assert plan.mode == "cli"
    cmd = plan.command
    assert cmd[:3] == ["tune", "run", "lora_finetune_single_device"]
    assert "--config" in cmd
    assert any(p.startswith("model.lora_rank=8") for p in cmd)
    assert any(p.startswith("model.lora_alpha=16") for p in cmd)
    assert any(p.startswith("dataset.source=yelp_review_full") for p in cmd)
    assert any(p.startswith("optimizer.lr=") for p in cmd)
    assert any(p == "dtype=bf16" for p in cmd)
    assert any(p == "model.lora_attn_modules=[q_proj,v_proj]" for p in cmd)


def test_torchtune_int4_quantize_base(sample_config):
    cfg = sample_config.model_copy(update={"precision": Precision.INT4})
    cmd = get_backend("torchtune").build_plan(cfg).command
    assert "model.quantize_base=True" in cmd


# -- train(dry_run) and dependency guards ------------------------------


@pytest.mark.parametrize("name", ["trl", "unsloth", "llamafactory", "torchtune"])
def test_train_dry_run_returns_plan_without_executing(name, sample_config):
    plan = get_backend(name).train(sample_config, dry_run=True)
    assert isinstance(plan, TrainingPlan)
    assert plan.backend == name


@pytest.mark.parametrize("name", ["trl", "unsloth", "llamafactory", "torchtune"])
def test_real_train_raises_dependency_error_when_libs_absent(
    name, sample_config, ml_libs_absent
):
    assert ml_libs_absent
    with pytest.raises(BackendDependencyError) as exc:
        get_backend(name).train(sample_config, dry_run=False)
    msg = str(exc.value).lower()
    assert "install" in msg or "deferred" in msg


def test_command_str_is_shell_safe(sample_config):
    plan = get_backend("trl").build_plan(sample_config)
    # command_str must be a single, joinable string.
    assert isinstance(plan.command_str, str)
    assert "trl sft" in plan.command_str


def test_plan_to_dict_is_serializable(sample_config):
    plan = get_backend("llamafactory").build_plan(sample_config)
    d = plan.to_dict()
    assert d["backend"] == "llamafactory"
    assert "llamafactory_sft.yaml" in d["artifacts"]
