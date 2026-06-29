"""Tests for conversation management."""

from towel.agent.conversation import Conversation, Message, Role


def test_conversation_add():
    conv = Conversation()
    msg = conv.add(Role.USER, "Hello, Towel!")
    assert msg.role == Role.USER
    assert msg.content == "Hello, Towel!"
    assert len(conv) == 1


def test_conversation_last():
    conv = Conversation()
    assert conv.last is None
    conv.add(Role.USER, "first")
    conv.add(Role.ASSISTANT, "second")
    assert conv.last is not None
    assert conv.last.content == "second"


def test_to_chat_messages():
    conv = Conversation()
    conv.add(Role.USER, "hi")
    conv.add(Role.ASSISTANT, "hello")
    msgs = conv.to_chat_messages()
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_message_id_unique():
    m1 = Message(role=Role.USER, content="a")
    m2 = Message(role=Role.USER, content="b")
    assert m1.id != m2.id


def test_message_from_dict_tolerates_missing_id_and_timestamp():
    # The coordinator injects a memory system message on the wire without an
    # id/timestamp; from_dict must fall back to defaults rather than KeyError,
    # which previously crashed the worker's job deserialization and hung the
    # request until the inference timeout.
    m = Message.from_dict(
        {"role": "system", "content": "mem", "metadata": {"source": "coord_memory_injection"}}
    )
    assert m.role == Role.SYSTEM
    assert m.content == "mem"
    assert m.id  # generated
    assert m.timestamp is not None  # defaulted to now()


def test_conversation_from_dict_with_idless_injected_message():
    conv = Conversation.from_dict(
        {
            "id": "c1",
            "channel": "cli",
            "created_at": "2026-06-29T00:00:00+00:00",
            "messages": [
                {"role": "system", "content": "mem", "metadata": {}},
                {
                    "id": "u1",
                    "role": "user",
                    "content": "hi",
                    "timestamp": "2026-06-29T00:00:01+00:00",
                    "metadata": {},
                },
            ],
        }
    )
    assert len(conv.messages) == 2
    assert conv.messages[0].role == Role.SYSTEM
    assert conv.messages[1].content == "hi"
