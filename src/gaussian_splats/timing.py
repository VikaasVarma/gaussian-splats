import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
from typing import Any

import torch
from torch._C import device

DEBUG = os.getenv("GAUSSIAN_SPLATS_DEBUG", "1") != "0"

# Groups timing blocks by tag
_timers = ContextVar[tuple[dict[str, dict[str, float]], device] | None]("timings", default=None)


@contextmanager
def record_timings(device: torch.device) -> Iterator[dict[str, dict[str, float]]]:
    times: dict[str, dict[str, float]] = {}
    token = _timers.set((times, device))
    try:
        yield times
    finally:
        _timers.reset(token)


def timed(title: str | Callable[..., Any] | None = None, *, tag: str = "default") -> Any:
    if callable(title):
        return _decorate(title, title.__name__.replace("_", " ").title(), tag)
    if title is None:
        return lambda function: _decorate(
            function, function.__name__.replace("_", " ").title(), tag
        )
    return _timer(title, tag)


def _decorate(function: Callable[..., Any], title: str, tag: str) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _timer(title, tag):
            return function(*args, **kwargs)

    return wrapped


@contextmanager
def _timer(title: str, tag: str) -> Iterator[None]:
    timer = _timers.get()
    if DEBUG and timer is not None:
        times, device = timer
        _synchronize(device)
        start = perf_counter()
        try:
            yield
        finally:
            _synchronize(device)
            group = times.setdefault(tag, {})
            group[title] = group.get(title, 0) + 1000 * (perf_counter() - start)
    else:
        yield


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
