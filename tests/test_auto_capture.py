"""Tests for the heuristic auto-capture extractor."""

import pytest

from towel.memory.auto_capture import Capture, apply, extract
from towel.memory.store import MemoryStore


# ── extract() pattern coverage ────────────────────────────────────────


class TestRolePattern:
    def test_im_a_role(self):
        caps = extract("I'm a senior backend engineer.")
        assert any(c.key == "role" and "engineer" in c.content for c in caps)

    def test_i_am_an_acronym_role(self):
        # The role pattern requires an "an" article for vowel-initial
        # words; verify it still fires on acronyms like "SRE".
        caps = extract("I am an SRE working on observability.")
        assert any(c.key == "role" and c.content == "SRE" for c in caps)

    def test_negated_role_skipped(self):
        caps = extract("I'm not a data scientist.")
        assert not any(c.key == "role" for c in caps)

    def test_negated_then_affirmed_captures_only_affirmed(self):
        # The clause boundary (comma) resets the negation scope, so
        # the trailing "I'm a designer" still captures.
        caps = extract("I'm not a data scientist, I'm a designer.")
        roles = [c.content for c in caps if c.key == "role"]
        assert "designer" in roles
        assert "data scientist" not in roles


class TestEmployerPattern:
    def test_work_at_company(self):
        caps = extract("I work at Anthropic.")
        assert any(c.key == "employer" and c.content == "Anthropic" for c in caps)

    def test_work_for_company(self):
        caps = extract("I work for OpenAI on safety.")
        # Trailing context after the company name should be excluded
        # by the bounded match — "OpenAI" alone, not "OpenAI on safety".
        emp = [c.content for c in caps if c.key == "employer"]
        assert emp and emp[0].startswith("OpenAI")


class TestPreferencePattern:
    def test_prefer_creates_preference(self):
        caps = extract("I prefer short replies with no preamble.")
        prefs = [c for c in caps if c.memory_type == "preference"]
        assert prefs
        assert "short replies" in prefs[0].content

    def test_multiple_preferences_get_distinct_keys(self):
        caps = extract("I prefer concise output. I like dark mode in editors.")
        keys = {c.key for c in caps if c.memory_type == "preference"}
        # Both preferences should land with different slugged keys so
        # neither overwrites the other.
        assert len(keys) >= 2


class TestProjectPattern:
    def test_my_project_is(self):
        caps = extract("my project is towel, a local AI coordinator.")
        assert any(c.key == "current_project" and "towel" in c.content for c in caps)

    def test_were_building(self):
        caps = extract("we're building a memory layer with FTS5.")
        assert any(c.memory_type == "project" for c in caps)


class TestDeadlinePattern:
    def test_we_ship_by(self):
        caps = extract("we ship by March 15.")
        assert any(c.key == "deadline" and "March 15" in c.content for c in caps)

    def test_deadline_is(self):
        caps = extract("deadline is next Friday.")
        assert any(c.key == "deadline" and "next Friday" in c.content for c in caps)


class TestExplicitRemember:
    def test_remember_that(self):
        caps = extract("remember that I use vim everywhere.")
        facts = [c for c in caps if c.memory_type == "fact"]
        assert facts
        assert "vim" in facts[0].content


class TestTechStackPattern:
    def test_we_use_named_tech(self):
        caps = extract("we use Python 3.14 in production.")
        assert any(
            c.source_pattern == "tech-stack" and "Python 3.14" in c.content
            for c in caps
        )

    def test_we_run_typescript(self):
        caps = extract("we run TypeScript on Kubernetes.")
        assert any(c.source_pattern == "tech-stack" for c in caps)

    def test_generic_verb_not_grabbed(self):
        # "we use" followed by a lowercase generic noun shouldn't fire
        # — only named tech (capitalized or whitelisted lowercase).
        caps = extract("we use those for everything.")
        assert not any(c.source_pattern == "tech-stack" for c in caps)


class TestLocationPattern:
    def test_im_in_timezone(self):
        caps = extract("I'm in Pacific time.")
        assert any(c.key == "location" and "Pacific" in c.content for c in caps)

    def test_im_based_in_city(self):
        caps = extract("I'm based in Berlin, working remotely.")
        assert any(c.key == "location" and c.content == "Berlin" for c in caps)


class TestToolChoicePattern:
    def test_my_editor_is(self):
        caps = extract("my editor is neovim.")
        assert any(c.key == "editor" and c.content == "neovim" for c in caps)

    def test_compound_my_x_is_y(self):
        caps = extract("my editor is neovim and my shell is zsh.")
        keys = {c.key for c in caps if c.source_pattern == "tool-choice"}
        assert "editor" in keys
        assert "shell" in keys


class TestNoiseRobustness:
    def test_empty_input(self):
        assert extract("") == []

    def test_random_text_no_captures(self):
        assert extract("the quick brown fox jumps over the lazy dog") == []

    def test_long_paragraph_handles_multiple_patterns(self):
        text = (
            "Hi! I'm a senior engineer. I work at Anthropic. "
            "my project is towel and we ship by EOQ. "
            "I prefer terse responses."
        )
        caps = extract(text)
        keys = {c.key for c in caps}
        assert "role" in keys
        assert "employer" in keys
        assert "current_project" in keys
        assert "deadline" in keys
        # At least one preference key (slugged).
        assert any(k.startswith("preference_") for k in keys)


# ── apply() integration with the store ────────────────────────────────


@pytest.fixture
def store(tmp_path):
    return MemoryStore(store_dir=tmp_path)


class TestApply:
    def test_writes_new_captures(self, store):
        written = apply("I'm a backend engineer.", store)
        assert any(c.key == "role" for c in written)
        assert store.recall("role").content == "backend engineer"

    def test_skips_existing_keys_by_default(self, store):
        # Operator-set memory must not be trampled by heuristic refire.
        store.remember("role", "data scientist", memory_type="user")
        written = apply("I'm an SRE.", store)
        assert written == []
        assert store.recall("role").content == "data scientist"

    def test_overwrite_mode_does_replace(self, store):
        store.remember("role", "stale value", memory_type="user")
        written = apply("I'm a designer.", store, overwrite=True)
        assert any(c.key == "role" for c in written)
        assert store.recall("role").content == "designer"

    def test_no_captures_is_no_op(self, store):
        before = store.count
        assert apply("the weather is fine", store) == []
        assert store.count == before

    def test_negated_match_does_not_write(self, store):
        apply("I'm not a manager.", store)
        assert store.recall("role") is None

    def test_idempotent_on_second_call(self, store):
        first = apply("I'm a developer.", store)
        second = apply("I'm a developer.", store)
        # First call wrote, second is a no-op because the key exists.
        assert first and not second
        assert store.recall("role").content == "developer"


class TestCaptureDataclass:
    def test_capture_is_frozen(self):
        c = Capture(key="k", content="v", memory_type="fact", source_pattern="p")
        with pytest.raises(Exception):
            c.content = "mutated"  # frozen=True
