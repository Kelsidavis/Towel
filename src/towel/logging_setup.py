"""Single source of truth for ``basicConfig`` in Towel's long-running CLI entry points.

Each of ``towel serve`` / ``towel worker`` / ``towel setup`` / ``towel launcher``
needs to surface INFO-level events in the operator's terminal. The
default Python root logger is WARNING, so without an explicit
``basicConfig`` the operator sees the startup banner once and then
nothing — even as workers connect, jobs run, and disconnects happen.

The shared helper here keeps the format consistent across processes and
keeps the per-command call sites to one line.

Scoped to entry-point code paths — never imported eagerly at module
load time, so test runs (which import these modules) don't mutate
global logging state.
"""

from __future__ import annotations

import logging

_TERMINAL_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_TERMINAL_DATEFMT = "%H:%M:%S"


def configure_terminal_logging(level: int = logging.INFO) -> None:
    """Configure ``logging.basicConfig`` for a long-running Towel command.

    Idempotent — repeated calls (e.g. when the same process re-enters a
    ``run()`` after a reload) won't add duplicate handlers because
    ``basicConfig`` is a no-op when the root logger already has any
    handlers attached.
    """
    logging.basicConfig(
        level=level,
        format=_TERMINAL_FORMAT,
        datefmt=_TERMINAL_DATEFMT,
    )
