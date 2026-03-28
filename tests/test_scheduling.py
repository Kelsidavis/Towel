"""Tests for scheduled tasks."""
import pytest
from towel.agent.scheduling import (
    Schedule, add_schedule, remove_schedule, list_schedules, toggle_schedule,
    _save_schedules, SCHEDULES_FILE,
)


class TestSchedule:
    def test_roundtrip(self):
        s = Schedule(name="test", cron="*/5 * * * *", action="tool:git_status")
        d = s.to_dict()
        s2 = Schedule.from_dict(d)
        assert s2.name == "test"
        assert s2.cron == "*/5 * * * *"

    def test_defaults(self):
        s = Schedule(name="x", cron="* * * * *", action="tool:y")
        assert s.enabled is True
        assert s.run_count == 0


class TestScheduleStorage:
    @pytest.fixture(autouse=True)
    def tmp_storage(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.agent.scheduling.SCHEDULES_FILE", tmp_path / "schedules.json")

    def test_add_and_list(self):
        add_schedule("daily", "0 9 * * *", "pipeline:project-health")
        schedules = list_schedules()
        assert len(schedules) == 1
        assert schedules[0].name == "daily"

    def test_add_replaces(self):
        add_schedule("x", "* * * * *", "tool:a")
        add_schedule("x", "*/5 * * * *", "tool:b")
        assert len(list_schedules()) == 1
        assert list_schedules()[0].action == "tool:b"

    def test_remove(self):
        add_schedule("temp", "* * * * *", "tool:x")
        assert remove_schedule("temp") is True
        assert len(list_schedules()) == 0

    def test_remove_nonexistent(self):
        assert remove_schedule("nope") is False

    def test_toggle(self):
        add_schedule("t", "* * * * *", "tool:x")
        result = toggle_schedule("t")
        assert "Disabled" in result
        assert list_schedules()[0].enabled is False
        result = toggle_schedule("t")
        assert "Enabled" in result

    def test_empty_list(self):
        assert list_schedules() == []
