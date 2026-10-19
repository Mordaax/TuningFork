"""Config scaffolder (``klt init``).

The generated starter config MUST be a valid TrainingConfig and MUST be
produced with no ML libraries installed.
"""

from __future__ import annotations

import pytest
import yaml

from kubeflow_llm_trainer.config import TrainingConfig
from kubeflow_llm_trainer.scaffold import starter_config_yaml


def test_default_scaffold_is_valid_config():
    text = starter_config_yaml()
    cfg = TrainingConfig.from_dict(yaml.safe_load(text))
    assert cfg.model_name_or_path
    assert cfg.lora.enabled is True


@pytest.mark.parametrize("backend", ["trl", "unsloth", "llamafactory", "torchtune"])
def test_scaffold_per_backend_is_valid(backend):
    text = starter_config_yaml(backend)
    # Backend hint is embedded as a comment.
    assert backend in text
    cfg = TrainingConfig.from_dict(yaml.safe_load(text))
    assert cfg.lora.enabled is True


def test_scaffold_is_commented():
    text = starter_config_yaml("trl")
    # Plenty of guidance comments for a human to edit.
    assert text.count("#") > 10
    assert "REQUIRED" in text


def test_scaffold_unknown_backend_raises():
    with pytest.raises(KeyError):
        starter_config_yaml("nope")


def test_scaffold_extra_is_empty_dict_not_collision():
    # The scaffold must not put reserved keys under `extra` (would fail validation).
    cfg = TrainingConfig.from_dict(yaml.safe_load(starter_config_yaml()))
    assert cfg.extra == {}
