"""Tests for .towel.md project context loading."""

from pathlib import Path

import pytest

from towel.agent.project import (
    find_project_contexts,
    load_project_context,
    CONTEXT_FILENAME,
    MAX_CONTEXT_BYTES,
)


class TestFindProjectContexts:
    def test_no_context_files(self, tmp_path):
        paths = find_project_contexts(tmp_path)
        assert paths == []

    def test_finds_in_current_dir(self, tmp_path):
        (tmp_path / CONTEXT_FILENAME).write_text("# My Project")
        paths = find_project_contexts(tmp_path)
        assert len(paths) == 1
        assert paths[0].name == CONTEXT_FILENAME

    def test_finds_in_parent(self, tmp_path):
        (tmp_path / CONTEXT_FILENAME).write_text("# Root context")
        child = tmp_path / "subdir"
        child.mkdir()
        paths = find_project_contexts(child)
        assert len(paths) == 1
        assert "Root context" in paths[0].read_text()

    def test_finds_nested_contexts(self, tmp_path):
        (tmp_path / CONTEXT_FILENAME).write_text("# Root")
        child = tmp_path / "sub"
        child.mkdir()
        (child / CONTEXT_FILENAME).write_text("# Child")

        paths = find_project_contexts(child)
        assert len(paths) == 2
        # Child should come first (most specific)
        assert "Child" in paths[0].read_text()
        assert "Root" in paths[1].read_text()

    def test_finds_dot_towel_directory(self, tmp_path):
        towel_dir = tmp_path / ".towel"
        towel_dir.mkdir()
        (towel_dir / "architecture.md").write_text("# Architecture")
        (towel_dir / "conventions.md").write_text("# Conventions")

        paths = find_project_contexts(tmp_path)
        assert len(paths) == 2

    def test_both_file_and_directory(self, tmp_path):
        (tmp_path / CONTEXT_FILENAME).write_text("# Main")
        towel_dir = tmp_path / ".towel"
        towel_dir.mkdir()
        (towel_dir / "extra.md").write_text("# Extra")

        paths = find_project_contexts(tmp_path)
        assert len(paths) == 2


class TestLoadProjectContext:
    def test_empty_when_no_files(self, tmp_path):
        assert load_project_context(tmp_path) == ""

    def test_loads_single_file(self, tmp_path):
        (tmp_path / CONTEXT_FILENAME).write_text("Built with Rust and React")
        block = load_project_context(tmp_path)
        assert "Project Context" in block
        assert "Built with Rust and React" in block

    def test_loads_multiple_files(self, tmp_path):
        (tmp_path / CONTEXT_FILENAME).write_text("# Root project")
        child = tmp_path / "sub"
        child.mkdir()
        (child / CONTEXT_FILENAME).write_text("# Sub module")

        block = load_project_context(child)
        assert "Root project" in block
        assert "Sub module" in block
        assert "---" in block  # separator

    def test_truncates_large_content(self, tmp_path):
        huge = "x" * (MAX_CONTEXT_BYTES + 10000)
        (tmp_path / CONTEXT_FILENAME).write_text(huge)
        block = load_project_context(tmp_path)
        assert len(block) < MAX_CONTEXT_BYTES + 500  # some overhead for header
        assert "truncated" in block

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / CONTEXT_FILENAME).write_text("")
        assert load_project_context(tmp_path) == ""

    def test_handles_unreadable_gracefully(self, tmp_path):
        # Create a file then remove read permission
        f = tmp_path / CONTEXT_FILENAME
        f.write_text("content")
        f.chmod(0o000)
        try:
            # Should not crash
            load_project_context(tmp_path)
        finally:
            f.chmod(0o644)  # restore for cleanup
