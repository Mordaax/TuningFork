"""Backend registry: register, resolve, and list training backends.

Two discovery mechanisms are supported:

1. **In-process registration** via the :func:`register_backend` decorator.
   This is what the bundled adapters use (see ``backends/``).

2. **Entry-point plugins** (real-world packaging). Third-party packages can
   advertise a backend under the ``kubeflow_llm_trainer.backends`` entry-point
   group; :func:`load_entry_point_backends` discovers them lazily. Example
   ``pyproject.toml`` for an external plugin::

       [project.entry-points."kubeflow_llm_trainer.backends"]
       mybackend = "my_pkg.adapter:MyBackend"
"""

from __future__ import annotations

from typing import Callable, Iterable, Type

from .base import TrainerBackend

# name -> backend class
_REGISTRY: dict[str, Type[TrainerBackend]] = {}

ENTRY_POINT_GROUP = "kubeflow_llm_trainer.backends"


def register_backend(
    name: str | None = None,
) -> Callable[[Type[TrainerBackend]], Type[TrainerBackend]]:
    """Class decorator that registers a :class:`TrainerBackend` subclass.

    ``name`` defaults to the class's ``name`` attribute. Registration is
    case-insensitive on lookup.
    """

    def _decorate(cls: Type[TrainerBackend]) -> Type[TrainerBackend]:
        if not isinstance(cls, type) or not issubclass(cls, TrainerBackend):
            raise TypeError("register_backend can only decorate TrainerBackend subclasses")
        key = (name or getattr(cls, "name", "") or "").strip().lower()
        if not key:
            raise ValueError(
                f"Backend {cls.__name__} has no name; set the 'name' class "
                "attribute or pass register_backend(name=...)."
            )
        # Keep the class attribute and the registry key in sync.
        cls.name = key
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"Backend name '{key}' is already registered.")
        _REGISTRY[key] = cls
        return cls

    return _decorate


def get_backend(name: str) -> TrainerBackend:
    """Resolve a backend name to a fresh backend *instance*."""
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown backend '{name}'. Available backends: {available}"
        )
    return _REGISTRY[key]()


def get_backend_class(name: str) -> Type[TrainerBackend]:
    """Resolve a backend name to its class (no instantiation)."""
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown backend '{name}'. Available backends: {available}")
    return _REGISTRY[key]


def list_backends() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    return (name or "").strip().lower() in _REGISTRY


def unregister(name: str) -> None:
    """Remove a backend (primarily for tests)."""
    _REGISTRY.pop((name or "").strip().lower(), None)


def clear_registry() -> None:
    """Clear all registrations (primarily for tests)."""
    _REGISTRY.clear()


def load_entry_point_backends() -> list[str]:
    """Discover and register backends advertised via Python entry-points.

    Returns the list of names that were newly registered. Safe to call
    multiple times; failures to import a plugin are skipped silently-ish
    (the offending name is simply not added).
    """
    newly: list[str] = []
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover - very old runtimes
        return newly

    try:
        eps: Iterable = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - <3.10 selection API
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

    for ep in eps:
        try:
            cls = ep.load()
        except Exception:  # pragma: no cover - bad third-party plugin
            continue
        if isinstance(cls, type) and issubclass(cls, TrainerBackend):
            key = (getattr(cls, "name", "") or ep.name).strip().lower()
            if key and key not in _REGISTRY:
                cls.name = key
                _REGISTRY[key] = cls
                newly.append(key)
    return newly
