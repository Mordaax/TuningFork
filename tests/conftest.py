"""Shared pytest fixtures.

These tests deliberately run with NO ML libraries installed. We assert that
fact so the suite documents and enforces the "core is ML-free" invariant.
"""

from __future__ import annotations

import importlib.util

import pytest

from kubeflow_llm_trainer.config import (
    DatasetConfig,
    LoRAConfig,
    Precision,
    TrainingConfig,
    TrainingHyperParams,
)


@pytest.fixture
def sample_config() -> TrainingConfig:
    """A deterministic config used across backend plan tests."""
    return TrainingConfig(
        model_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        output_dir="./outputs/test",
        precision=Precision.BF16,
        dataset=DatasetConfig(
            name_or_path="yelp_review_full",
            split="train",
            text_field="text",
            max_samples=256,
        ),
        lora=LoRAConfig(
            enabled=True,
            r=8,
            alpha=16,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
        hyperparams=TrainingHyperParams(
            learning_rate=2e-4,
            num_epochs=1,
            per_device_batch_size=2,
            gradient_accumulation_steps=4,
            max_seq_len=512,
            seed=42,
        ),
    )


def _no_ml_libs() -> bool:
    for mod in ("torch", "transformers", "trl", "peft", "unsloth", "torchtune"):
        if importlib.util.find_spec(mod) is not None:
            return False
    return True


@pytest.fixture(scope="session")
def ml_libs_absent() -> bool:
    return _no_ml_libs()
