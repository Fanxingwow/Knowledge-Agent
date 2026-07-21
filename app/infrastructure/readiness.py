from __future__ import annotations

import math
import time

from starlette.datastructures import State


def object_storage_is_ready(state: State) -> bool:
    """Return a fail-closed view of the cached object-storage health."""
    ready_until = getattr(state, "minio_ready_until", 0.0)
    return (
        getattr(state, "minio_ready", False) is True
        and getattr(state, "minio_store", None) is not None
        and isinstance(ready_until, (int, float))
        and not isinstance(ready_until, bool)
        and math.isfinite(ready_until)
        and time.monotonic() < ready_until
    )


def import_runtime_is_ready(state: State) -> bool:
    """Return a fail-closed view of the cached import-runtime health."""
    ready_until = getattr(state, "import_ready_until", 0.0)
    return (
        getattr(state, "import_ready", False) is True
        and getattr(state, "import_coordinator", None) is not None
        and isinstance(ready_until, (int, float))
        and not isinstance(ready_until, bool)
        and math.isfinite(ready_until)
        and time.monotonic() < ready_until
    )
