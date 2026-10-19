"""kubeflow_llm_trainer: a pluggable multi-framework LLM fine-tuning backend.

A single unified :class:`TrainingConfig` is translated by pluggable backend
adapters (TRL, Unsloth, LlamaFactory, TorchTune) into each tool's native
invocation. Heavy ML dependencies are optional: importing this package and
building dry-run plans requires only ``pydantic`` (and ``pyyaml`` for the CLI).
"""

from __future__ import annotations

from .base import (
    BackendDependencyError,
    TrainerBackend,
    TrainingPlan,
)
from .config import (
    DatasetConfig,
    LoRAConfig,
    Precision,
    TrainingConfig,
    TrainingHyperParams,
)
from .registry import (
    get_backend,
    get_backend_class,
    list_backends,
    load_entry_point_backends,
    register_backend,
)
from .manifests import (
    MANIFEST_KINDS,
    ManifestOptions,
    build_manifest_docs,
    render_manifest_yaml,
)
from .scaffold import starter_config_yaml

# Side-effect import: registers all bundled backends.
from . import backends  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "TrainingConfig",
    "DatasetConfig",
    "LoRAConfig",
    "TrainingHyperParams",
    "Precision",
    # base
    "TrainerBackend",
    "TrainingPlan",
    "BackendDependencyError",
    # registry
    "register_backend",
    "get_backend",
    "get_backend_class",
    "list_backends",
    "load_entry_point_backends",
    # manifests
    "ManifestOptions",
    "MANIFEST_KINDS",
    "build_manifest_docs",
    "render_manifest_yaml",
    # scaffold
    "starter_config_yaml",
]
