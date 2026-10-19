"""Bundled backend adapters.

Importing this package triggers registration of all built-in backends via the
``@register_backend`` decorator on each adapter class.
"""

from __future__ import annotations

# Importing each module runs its @register_backend decorator.
from . import llamafactory, torchtune, trl, unsloth  # noqa: F401

__all__ = ["trl", "unsloth", "llamafactory", "torchtune"]
