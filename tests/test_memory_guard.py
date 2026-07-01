"""Tests for memory write guard (towel.memory.guard)."""

from towel.memory.guard import reject_reason


class TestDenyKeys:
    def test_flagged_key_blocked(self):
        assert reject_reason("flagged", "some content") is not None

    def test_violation_key_blocked(self):
        assert reject_reason("violation", "anything") is not None

    def test_safety_flag_key_blocked(self):
        assert reject_reason("safety_flag", "anything") is not None

    def test_normal_key_allowed(self):
        assert reject_reason("favorite_food", "pizza") is None

    def test_deny_keys_case_insensitive(self):
        assert reject_reason("FLAGGED", "content") is not None
        assert reject_reason("Violation", "content") is not None


class TestJudgmentBlocking:
    def test_user_characterization_with_judgment_vocab_blocked(self):
        assert reject_reason(
            "note",
            "The user is attempting to request disallowed content"
        ) is not None

    def test_user_characterization_without_judgment_passes(self):
        assert reject_reason(
            "note",
            "The user is a software engineer in Berlin"
        ) is None

    def test_judgment_vocab_without_user_framing_passes(self):
        assert reject_reason(
            "work",
            "I work on terrorism research at the university"
        ) is None


class TestIntentRestatement:
    def test_restating_user_request_blocked(self):
        assert reject_reason(
            "note",
            "User requests information about network security"
        ) is not None

    def test_user_wants_blocked(self):
        assert reject_reason(
            "note",
            "User wants to learn about Python"
        ) is not None

    def test_user_is_asking_blocked(self):
        assert reject_reason(
            "note",
            "User is asking about database performance"
        ) is not None

    def test_genuine_user_fact_passes(self):
        assert reject_reason(
            "preference",
            "User prefers dark mode and vim keybindings"
        ) is None

    def test_first_person_statement_passes(self):
        assert reject_reason(
            "fact",
            "I'm a vegetarian and allergic to nuts"
        ) is None


class TestEdgeCases:
    def test_empty_key_and_content(self):
        assert reject_reason("", "") is None

    def test_none_key_and_content(self):
        assert reject_reason(None, None) is None

    def test_whitespace_key_normalization(self):
        assert reject_reason("  flagged  ", "content") is not None
