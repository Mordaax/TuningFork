"""Adapter contract shared by all training backends.

A *backend* knows how to turn a :class:`~kubeflow_llm_trainer.config.TrainingConfig`
into a concrete invocation of one fine-tuning tool. Every backend subclasses
:class:`TrainerBackend` and implements :meth:`build_plan`. Real execution is
optional and guarded behind import checks of the heavy ML dependency.
"""

from __future__ import annotations

import importlib.util
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import TrainingConfig


class BackendDependencyError(RuntimeError):
    """Raised when a real training run is requested but its deps are absent."""


@dataclass
class TrainingPlan:
    """A concrete, inspectable description of *what would run*.

    ``mode`` is either ``"cli"`` (``command`` is the argv to exec) or
    ``"api"`` (``api_call`` describes the Python call that would be made).
    Both are populated where it makes sense so dry-run output is readable.
    """

    backend: str
    mode: str  # "cli" | "api"
    command: list[str] = field(default_factory=list)
    api_call: Optional[str] = None
    # Flat, human-readable summary of the resolved knobs.
    resolved: dict[str, Any] = field(default_factory=dict)
    # Files the adapter would write before launching (name -> contents).
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def command_str(self) -> str:
        """The CLI command as a copy-pasteable shell string."""
        return " ".join(shlex.quote(p) for p in self.command)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "command": self.command,
            "command_str": self.command_str if self.command else None,
            "api_call": self.api_call,
            "resolved": self.resolved,
            "artifacts": list(self.artifacts.keys()),
            "notes": self.notes,
        }


class TrainerBackend(ABC):
    """Abstract base class every backend adapter must implement."""

    #: Stable, lowercase identifier used by the registry and CLI.
    name: str = ""
    #: Import name of the optional heavy dependency (e.g. "trl").
    required_dependency: str = ""

    def validate(self, cfg: TrainingConfig) -> None:
        """Backend-specific validation beyond the schema.

        Default implementation is a no-op; the pydantic schema already covers
        structural validation. Override to reject configs a backend cannot
        satisfy (and raise ``ValueError`` with a clear message).
        """
        return None

    @abstractmethod
    def build_plan(self, cfg: TrainingConfig) -> TrainingPlan:
        """Translate the unified config into a concrete :class:`TrainingPlan`."""
        raise NotImplementedError

    # Backwards-friendly alias some callers may expect.
    def build_command(self, cfg: TrainingConfig) -> list[str]:
        """Convenience: just the argv from :meth:`build_plan`."""
        return self.build_plan(cfg).command

    def dependency_available(self) -> bool:
        """True if the backend's heavy dependency can be imported."""
        if not self.required_dependency:
            return True
        return importlib.util.find_spec(self.required_dependency) is not None

    def ensure_dependency(self) -> None:
        """Raise a clear, actionable error if the dependency is missing."""
        if not self.dependency_available():
            raise BackendDependencyError(
                f"Backend '{self.name}' requires '{self.required_dependency}' "
                f"to actually train. Install it, e.g.:\n"
                f"    pip install {self.required_dependency}\n"
                f"Or use --dry-run to inspect the plan without training."
            )

    def train(self, cfg: TrainingConfig, dry_run: bool = True) -> TrainingPlan:
        """Validate, build a plan, then (optionally) execute it.

        In ``dry_run`` mode this NEVER imports ML libraries or launches work;
        it returns the plan describing what *would* run. Real execution is
        delegated to :meth:`_execute` and guarded on the dependency.
        """
        self.validate(cfg)
        plan = self.build_plan(cfg)
        if dry_run:
            return plan
        self.ensure_dependency()
        self._execute(cfg, plan)
        return plan

    def _execute(self, cfg: TrainingConfig, plan: TrainingPlan) -> None:
        """Perform the real training run. Override in subclasses.

        The default raises, because real GPU runs are intentionally deferred
        in this proof-of-concept.
        """
        raise BackendDependencyError(
            f"Real execution for backend '{self.name}' is not wired up in "
            f"this proof-of-concept (GPU runs are deferred). The dry-run plan "
            f"shows the exact command/API call to run manually."
        )
