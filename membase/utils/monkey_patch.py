from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable


def make_attr_patch(obj: Any, attr: str) -> tuple[
    Callable[[], Callable[..., Any]], 
    Callable[[Callable[..., Any]], None]
]:
    """Helper to build getter and setter closures for a given object attribute.

    Args:
        obj (`Any`):
            The instance of the object to patch.
        attr (`str`):
            The method name of the object to patch.

    Returns:
        `tuple[Callable[[], Callable[..., Any]], Callable[[Callable[..., Any]], None]]`:
            A tuple containing the getter and setter closures.

    Example::

        from collections import deque 
        from memories.utils import make_attr_patch
        q = deque()
        getter, setter = make_attr_patch(q, "append")
    """
    def getter() -> Callable[..., Any]:
        return getattr(obj, attr)

    try:
        from pydantic import BaseModel as _BaseModel
    except Exception:
        _BaseModel = None

    if _BaseModel is not None and isinstance(obj, _BaseModel):
        def setter(fn: Callable[..., Any]) -> None:
            object.__setattr__(obj, attr, fn)
    else:
        def setter(fn: Callable[..., Any]) -> None:
            setattr(obj, attr, fn)
    return getter, setter


@dataclass
class PatchSpec:
    """A single patch rule."""
    
    name: str 
    "A human-readable identifier for debugging and monitoring."

    getter: Callable[[], Callable[..., Any]] 
    "It should return the current callable to patch."

    setter: Callable[[Callable[..., Any]], None] 
    "It should set the new callable in-place."

    wrapper: Callable[[Callable[..., Any]], Callable[..., Any]] 
    "A decorator that wraps the target callable."


class MonkeyPatcher:
    """A context manager to apply user-defined patch specs on enter and 
    restore originals on exit."""

    def __init__(self, specs: list[PatchSpec]) -> None:
        """Initialize the monkey patcher with the given patch specs.

        Args:
            specs (`list[PatchSpec]`):
                A list of patch specs to apply.

        Example::

            from membase.utils import (
                MonkeyPatcher,
                make_attr_patch,
                PatchSpec,
            )
            import functools
            
            class Queue:
                def __init__(self) -> None:
                    self.items = []

                def append(self, x: int) -> None:
                    self.items.append(x)

            q = Queue()
            getter, setter = make_attr_patch(q, "append")

            def positive_only(wrapper_target):
                @functools.wraps(wrapper_target)
                def wrapped(x: int, *args, **kwargs):
                    if x > 0:
                        return wrapper_target(x, *args, **kwargs)
                    return None
                return wrapped

            spec = PatchSpec(
                name=q.__class__.__name__,
                getter=getter,
                setter=setter,
                wrapper=positive_only,
            )
            with MonkeyPatcher([spec]):
                q.append(1)
                q.append(-1)

            assert q.items == [1]
        """
        self.specs = specs
        # A mapping from the spec name to the original callable. 
        self._originals = {}
        self._active = False

    def __enter__(self) -> MonkeyPatcher:
        """Apply the patch specs on enter."""
        if self._active:
            return self
        for spec in self.specs:
            original = spec.getter()
            wrapped = spec.wrapper(original)
            spec.setter(wrapped)
            self._originals[spec.name] = original
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Restore the original callables on exit."""
        # Best-effort restore in reverse order.
        for spec in reversed(self.specs):
            if spec.name in self._originals:
                try:
                    spec.setter(self._originals[spec.name])
                except Exception:
                    pass
        self._originals.clear()
        self._active = False
        return None
