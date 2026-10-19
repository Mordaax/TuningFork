"""Registry: register / resolve / list, plus error handling."""

from __future__ import annotations

import pytest

from kubeflow_llm_trainer.base import TrainerBackend, TrainingPlan
from kubeflow_llm_trainer.config import TrainingConfig
from kubeflow_llm_trainer.registry import (
    get_backend,
    get_backend_class,
    is_registered,
    list_backends,
    register_backend,
    unregister,
)


def test_builtin_backends_are_registered():
    names = list_backends()
    for expected in ("trl", "unsloth", "llamafactory", "torchtune"):
        assert expected in names


def test_resolve_returns_instance():
    backend = get_backend("trl")
    assert isinstance(backend, TrainerBackend)
    assert backend.name == "trl"


def test_resolve_is_case_insensitive():
    assert get_backend("TRL").name == "trl"
    assert get_backend("  TorchTune ").name == "torchtune"


def test_get_backend_class():
    cls = get_backend_class("unsloth")
    assert issubclass(cls, TrainerBackend)


def test_unknown_backend_raises_helpful_error():
    with pytest.raises(KeyError) as exc:
        get_backend("does-not-exist")
    assert "Available backends" in str(exc.value)


def test_register_and_unregister_roundtrip():
    @register_backend(name="dummy")
    class DummyBackend(TrainerBackend):
        name = "dummy"

        def build_plan(self, cfg: TrainingConfig) -> TrainingPlan:
            return TrainingPlan(backend=self.name, mode="cli", command=["echo", "hi"])

    try:
        assert is_registered("dummy")
        assert "dummy" in list_backends()
        assert get_backend("dummy").build_plan.__self__ is not None
    finally:
        unregister("dummy")
    assert not is_registered("dummy")


def test_duplicate_registration_raises():
    @register_backend(name="dup")
    class A(TrainerBackend):
        name = "dup"

        def build_plan(self, cfg: TrainingConfig) -> TrainingPlan:
            return TrainingPlan(backend=self.name, mode="cli")

    try:
        with pytest.raises(ValueError):

            @register_backend(name="dup")
            class B(TrainerBackend):  # noqa: F811
                name = "dup"

                def build_plan(self, cfg: TrainingConfig) -> TrainingPlan:
                    return TrainingPlan(backend=self.name, mode="cli")

    finally:
        unregister("dup")


def test_register_non_backend_raises():
    with pytest.raises(TypeError):

        @register_backend(name="bad")
        class NotABackend:  # type: ignore[misc]
            name = "bad"
