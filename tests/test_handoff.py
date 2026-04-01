"""Tests for graceful context handoff between workers."""

from towel.gateway.handoff import HandoffManager, HandoffReason, HandoffRecord


class TestHandoffManager:
    def test_plan_and_complete_handoff(self):
        mgr = HandoffManager()

        record = mgr.plan_handoff("s1", "w1", HandoffReason.WORKER_DRAINING)
        assert mgr.pending_count == 1

        mgr.assign_target("s1", "w2")
        assert record.to_worker_id == "w2"

        result = mgr.complete_handoff("s1", success=True)
        assert result is not None
        assert result.success is True
        assert result.duration_ms is not None
        assert mgr.pending_count == 0

    def test_failed_handoff(self):
        mgr = HandoffManager()
        mgr.plan_handoff("s1", "w1", HandoffReason.WORKER_DISCONNECTED)
        result = mgr.complete_handoff("s1", success=False, error="No workers available")

        assert result is not None
        assert result.success is False
        assert result.error == "No workers available"

    def test_sessions_needing_handoff(self):
        mgr = HandoffManager()
        session_workers = {
            "s1": "w1",
            "s2": "w1",
            "s3": "w2",
        }

        sessions = mgr.sessions_needing_handoff("w1", session_workers)
        assert sorted(sessions) == ["s1", "s2"]

    def test_excludes_already_pending(self):
        mgr = HandoffManager()
        session_workers = {"s1": "w1", "s2": "w1"}

        mgr.plan_handoff("s1", "w1", HandoffReason.WORKER_DRAINING)

        sessions = mgr.sessions_needing_handoff("w1", session_workers)
        assert sessions == ["s2"]

    def test_stats(self):
        mgr = HandoffManager()
        mgr.plan_handoff("s1", "w1", HandoffReason.WORKER_DRAINING)
        mgr.assign_target("s1", "w2")
        mgr.complete_handoff("s1", success=True)

        mgr.plan_handoff("s2", "w1", HandoffReason.WORKER_DISCONNECTED)
        mgr.complete_handoff("s2", success=False, error="No worker")

        stats = mgr.stats()
        assert stats["total"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1
        assert stats["by_reason"]["worker_draining"] == 1
        assert stats["by_reason"]["worker_disconnected"] == 1

    def test_max_history(self):
        mgr = HandoffManager(max_history=5)
        for i in range(10):
            mgr.plan_handoff(f"s{i}", "w1", HandoffReason.MANUAL_REBALANCE)
            mgr.complete_handoff(f"s{i}", success=True)

        assert len(mgr.history) == 5

    def test_recent_handoffs(self):
        mgr = HandoffManager()
        for i in range(5):
            mgr.plan_handoff(f"s{i}", "w1", HandoffReason.WORKER_DRAINING)
            mgr.assign_target(f"s{i}", "w2")
            mgr.complete_handoff(f"s{i}", success=True)

        recent = mgr.recent_handoffs(limit=3)
        assert len(recent) == 3
        assert all(r["success"] for r in recent)


class TestHandoffRecord:
    def test_to_dict(self):
        record = HandoffRecord(
            session_id="s1",
            from_worker_id="w1",
            to_worker_id="w2",
            reason=HandoffReason.WORKER_OVERLOADED,
            message_count=20,
            tokens_transferred=5000,
        )
        record.complete(success=True)

        d = record.to_dict()
        assert d["session_id"] == "s1"
        assert d["reason"] == "worker_overloaded"
        assert d["success"] is True
        assert d["duration_ms"] is not None
        assert d["message_count"] == 20
        assert d["tokens_transferred"] == 5000

    def test_incomplete_record(self):
        record = HandoffRecord(
            session_id="s1",
            from_worker_id="w1",
            to_worker_id="",
            reason=HandoffReason.CAPACITY_EXCEEDED,
        )
        d = record.to_dict()
        assert d["success"] is False
        assert "completed_at" not in d
        assert record.duration_ms is None
