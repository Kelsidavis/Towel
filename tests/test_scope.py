"""Tests for project scope derivation + scoped memory retrieval."""

from __future__ import annotations

import pytest

from towel.memory.scope import derive_scope, find_project_root
from towel.memory.store import MemoryStore


class TestFindProjectRoot:
    def test_finds_towel_md_marker(self, tmp_path):
        root = tmp_path / "proj"
        (root / "src").mkdir(parents=True)
        (root / ".towel.md").write_text("hi", encoding="utf-8")
        assert find_project_root(root / "src") == root

    def test_walks_up_for_git(self, tmp_path):
        root = tmp_path / "proj"
        (root / "sub" / "nested").mkdir(parents=True)
        (root / ".git").mkdir()
        assert find_project_root(root / "sub" / "nested") == root

    def test_returns_none_in_plain_dir(self, tmp_path):
        # No markers anywhere up to the tmp root — the walk should
        # terminate without crashing and return None.
        assert find_project_root(tmp_path) is None

    def test_pyproject_root(self, tmp_path):
        root = tmp_path / "py"
        root.mkdir()
        (root / "pyproject.toml").write_text("[project]\nname='x'", encoding="utf-8")
        assert find_project_root(root) == root


class TestDeriveScope:
    def test_stable_for_same_path(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / ".towel.md").write_text("x", encoding="utf-8")
        a = derive_scope(root)
        b = derive_scope(root)
        assert a == b
        assert a.startswith("proj:")

    def test_different_paths_yield_different_scopes(self, tmp_path):
        root1 = tmp_path / "a"
        root2 = tmp_path / "b"
        root1.mkdir(); root2.mkdir()
        (root1 / ".towel.md").write_text("", encoding="utf-8")
        (root2 / ".towel.md").write_text("", encoding="utf-8")
        assert derive_scope(root1) != derive_scope(root2)

    def test_empty_when_no_marker(self, tmp_path):
        # Walks above tmp_path will hit /tmp which has no markers
        # either, so the helper returns "".
        # (In practice on dev machines / could have markers; tmp does
        # not.)
        assert derive_scope(tmp_path) == ""

    def test_basename_slug_is_lowercase(self, tmp_path):
        root = tmp_path / "MyCool-Project"
        root.mkdir()
        (root / ".towel.md").write_text("", encoding="utf-8")
        scope = derive_scope(root)
        # Format: proj:<slug>:<sha8>
        slug = scope.split(":")[1]
        assert slug == "mycool-project"


class TestScopedStore:
    def test_remember_uses_default_scope(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "db", default_scope="proj:a")
        e = store.remember("k", "v")
        assert e.scope == "proj:a"

    def test_explicit_scope_overrides_default(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "db", default_scope="proj:a")
        e = store.remember("k", "v", scope="proj:b")
        assert e.scope == "proj:b"

    def test_explicit_empty_scope_is_global(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "db", default_scope="proj:a")
        e = store.remember("k", "v", scope="")
        assert e.scope == ""

    def test_update_preserves_scope_when_unspecified(self, tmp_path):
        # An entry written with scope="proj:a" shouldn't quietly
        # flip to global when a remember(...) call doesn't pass
        # scope and the new caller has a different default_scope.
        store_a = MemoryStore(store_dir=tmp_path / "db", default_scope="proj:a")
        store_a.remember("k", "first", scope="proj:a")
        store_b = MemoryStore(store_dir=tmp_path / "db", default_scope="proj:b")
        store_b.remember("k", "second")  # no scope param
        # Same DB → re-read should still report proj:a.
        store_c = MemoryStore(store_dir=tmp_path / "db")
        e = store_c.recall("k")
        assert e.scope == "proj:a"

    def test_recall_all_shows_current_scope_plus_global(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "db", default_scope="proj:a")
        store.remember("global_fact", "x", scope="")
        store.remember("a_fact", "y", scope="proj:a")
        store.remember("b_fact", "z", scope="proj:b")
        keys = {e.key for e in store.recall_all()}
        assert keys == {"global_fact", "a_fact"}  # b_fact is hidden

    def test_recall_all_no_default_shows_everything(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "db")  # default_scope=""
        store.remember("global_fact", "x", scope="")
        store.remember("a_fact", "y", scope="proj:a")
        store.remember("b_fact", "z", scope="proj:b")
        keys = {e.key for e in store.recall_all()}
        # Empty default_scope means no filter — show all scopes.
        assert keys == {"global_fact", "a_fact", "b_fact"}

    def test_search_respects_scope(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "db", default_scope="proj:a")
        store.remember("a", "elephant scope test", scope="proj:a")
        store.remember("b", "elephant scope test", scope="proj:b")
        results = store.search("elephant")
        assert [e.key for e in results] == ["a"]

    def test_fused_search_respects_scope(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "db", default_scope="proj:a")
        store.remember("a", "shared content", scope="proj:a")
        store.remember("b", "shared content", scope="proj:b")
        results = store.fused_search("shared")
        assert [e.key for e in results] == ["a"]

    def test_explicit_scope_filter_overrides_default(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "db", default_scope="proj:a")
        store.remember("a", "x", scope="proj:a")
        store.remember("b", "y", scope="proj:b")
        results = store.recall_all(scope="proj:b")
        assert [e.key for e in results] == ["b"]


class TestSetScope:
    def test_promote_to_global(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("k", "v", scope="proj:a")
        assert store.set_scope("k", "") is True
        assert store.recall("k").scope == ""

    def test_demote_to_project(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("k", "v")  # global
        assert store.set_scope("k", "proj:b") is True
        assert store.recall("k").scope == "proj:b"

    def test_noop_when_already_in_target(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("k", "v", scope="proj:a")
        assert store.set_scope("k", "proj:a") is False

    def test_unknown_key_returns_false(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        assert store.set_scope("missing", "proj:x") is False
