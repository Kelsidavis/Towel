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
