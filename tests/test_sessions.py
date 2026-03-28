"""Tests for session management."""

from towel.gateway.sessions import SessionManager


def test_get_or_create():
    sm = SessionManager()
    s1 = sm.get_or_create("test")
    s2 = sm.get_or_create("test")
    assert s1 is s2
    assert len(sm) == 1


def test_remove():
    sm = SessionManager()
    sm.get_or_create("test")
    sm.remove("test")
    assert len(sm) == 0
    assert sm.get("test") is None


def test_all():
    sm = SessionManager()
    sm.get_or_create("a")
    sm.get_or_create("b")
    assert len(sm.all()) == 2
