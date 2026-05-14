"""Tests for the shared terminal-logging helper."""

from __future__ import annotations

import logging

from towel.logging_setup import configure_terminal_logging


class TestConfigureTerminalLogging:
    def test_sets_root_level_to_info_by_default(self):
        # Stash + clear handlers so basicConfig actually fires.
        root = logging.getLogger()
        saved_level = root.level
        saved_handlers = list(root.handlers)
        for h in saved_handlers:
            root.removeHandler(h)
        try:
            configure_terminal_logging()
            assert root.level == logging.INFO
            assert root.handlers, "expected at least one handler attached"
        finally:
            # Restore original state.
            for h in list(root.handlers):
                root.removeHandler(h)
            for h in saved_handlers:
                root.addHandler(h)
            root.setLevel(saved_level)

    def test_repeat_calls_are_idempotent(self):
        # basicConfig is a no-op when handlers already exist. Verify by
        # calling twice and confirming the handler count doesn't double.
        root = logging.getLogger()
        saved_level = root.level
        saved_handlers = list(root.handlers)
        for h in saved_handlers:
            root.removeHandler(h)
        try:
            configure_terminal_logging()
            handlers_after_first = len(root.handlers)
            configure_terminal_logging()
            assert len(root.handlers) == handlers_after_first
        finally:
            for h in list(root.handlers):
                root.removeHandler(h)
            for h in saved_handlers:
                root.addHandler(h)
            root.setLevel(saved_level)

    def test_accepts_custom_level(self):
        root = logging.getLogger()
        saved_level = root.level
        saved_handlers = list(root.handlers)
        for h in saved_handlers:
            root.removeHandler(h)
        try:
            configure_terminal_logging(level=logging.DEBUG)
            assert root.level == logging.DEBUG
        finally:
            for h in list(root.handlers):
                root.removeHandler(h)
            for h in saved_handlers:
                root.addHandler(h)
            root.setLevel(saved_level)
