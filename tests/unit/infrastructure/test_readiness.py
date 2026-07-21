from __future__ import annotations

import time
from threading import Event, Lock

from fastapi import FastAPI
from starlette.datastructures import State

from app.infrastructure.readiness import import_runtime_is_ready
from app.main import _bounded_component_probe


class HangingProbe:
    def __init__(self) -> None:
        self.release = Event()
        self.started = Event()
        self._lock = Lock()
        self.calls = 0
        self.active = 0
        self.max_active = 0

    def probe(self) -> None:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            self.release.wait(timeout=5)
        finally:
            with self._lock:
                self.active -= 1


async def test_timed_out_component_probe_is_not_stacked() -> None:
    app = FastAPI()
    app.state.import_probe_inflight = {}
    component = HangingProbe()

    first = await _bounded_component_probe(
        app,
        "import_processor",
        component,
        timeout_seconds=0.02,
    )
    assert component.started.wait(timeout=1)
    second = await _bounded_component_probe(
        app,
        "import_processor",
        component,
        timeout_seconds=0.02,
    )

    assert first is False
    assert second is False
    assert component.calls == 1
    assert component.max_active == 1

    component.release.set()
    assert component.started.wait(timeout=1)


def test_import_runtime_readiness_cache_fails_closed() -> None:
    state = State()
    state.import_coordinator = object()
    state.import_ready = True
    state.import_ready_until = time.monotonic() + 1

    assert import_runtime_is_ready(state) is True

    state.import_ready_until = time.monotonic() - 1
    assert import_runtime_is_ready(state) is False
    state.import_ready_until = float("nan")
    assert import_runtime_is_ready(state) is False
    state.import_ready_until = True
    assert import_runtime_is_ready(state) is False
