"""Write-time guard for the persistent memory store.

Models with residual safety conditioning — abliterated checkpoints in
particular — sometimes repurpose a write-capable ``remember`` tool to
log *judgments about the user* ("user is attempting X", "disallowed
request") rather than the user-stated facts the tool is meant to hold.
Those entries are worse than useless: they pollute the prompt block on
every future turn and, in a cluster deployment, replicate to every node
via ``ClusterMemorySync`` with full provenance.

This module rejects such writes at the tool boundary. It is deliberately
conservative — it blocks the narrow, recognizable shape of a third-party
judgment/safety-flag and lets everything else through, because a missed
block costs only a stray memory while a false block silently drops a
legitimate fact.

``reject_reason(key, content)`` returns a human-readable reason string
when the write should be refused, or ``None`` when it is allowed.
"""

from __future__ import annotations

import re

# Keys that only a flagging/triage behavior would choose. Operator and
# genuine user-fact keys don't look like this.
_DENY_KEYS = frozenset(
    {
        "disallowed",
        "violation",
        "policy_violation",
        "flagged",
        "flag",
        "safety_flag",
        "unsafe",
        "abuse",
        "report",
    }
)

# Safety/triage vocabulary. A judgment about the user almost always
# pairs one of these with a third-person reference to the user.
_JUDGMENT_VOCAB = re.compile(
    r"\b("
    r"disallow(?:ed|s)?|violat(?:e|es|ing|ion)|prohibit(?:ed)?|"
    r"illegal|unlawful|extremis[mt]|terroris[mt]|violent|violence|"
    r"safety\s+(?:policy|rule|concern|assessment)|policy\s+violation|"
    r"facilitat(?:e|ing)\s+(?:violent|illegal)|"
    r"attempting\s+to|requesting\s+disallowed|harmful\s+intent|"
    r"flag(?:ged|ging)?\s+(?:this|the\s+user|for)"
    r")\b",
    re.IGNORECASE,
)

# Third-person framing that characterizes the user rather than recording
# something the user stated about themselves. "User is/was/appears/seems/
# attempting/requesting/trying ..." is the tell.
_USER_CHARACTERIZATION = re.compile(
    r"\b(?:the\s+)?user\s+"
    r"(?:is|was|appears|seems|seemed|may\s+be|might\s+be|"
    r"attempt(?:s|ing|ed)?|request(?:s|ing|ed)?|"
    r"tr(?:y|ies|ying|ied)|want(?:s|ed)?\s+to|"
    r"intend(?:s|ing|ed)?|plan(?:s|ning|ned)?)\b",
    re.IGNORECASE,
)

# Third-person restatement of the *current request* — "User requests X",
# "User wants info about Y", "User is asking about Z". These are running
# commentary on the live turn, not durable facts, and a model that logs
# them is journaling the user rather than remembering anything. Narrower
# than _USER_CHARACTERIZATION on purpose: only intent/request verbs, NOT
# bare "user is ..." — so a real attribute ("user is a vegetarian",
# "user is based in Berlin") still passes.
_INTENT_RESTATEMENT = re.compile(
    r"\b(?:the\s+)?user\s+"
    r"(?:"
    r"requests?|requested|wants?|wanted|would\s+like|"
    r"is\s+(?:interested|asking|looking|requesting|seeking|inquiring)|"
    r"asks?|asked|intends?|intended|needs?|needed|seeks?|"
    r"is\s+trying\s+to|is\s+attempting\s+to|plans?\s+to"
    r")\b",
    re.IGNORECASE,
)


_PATH_TRAVERSAL = re.compile(r"\.\./|/\.\.|^/|\\")


def reject_reason(key: str, content: str) -> str | None:
    """Return why a memory write should be refused, or None to allow it.

    The block fires when the write reads as a third-party judgment or
    safety flag about the user rather than a user-stated fact:

    * the key contains path traversal sequences (``../``, leading ``/``),
    * the key is one only a flagging behavior would pick, or
    * the content characterizes the user ("user is/attempting/...") AND
      carries safety/triage vocabulary.

    Plain facts — even ones that happen to mention sensitive words in
    the user's own voice ("I work on terrorism research") — don't match
    because they lack the third-person "the user is ..." framing.
    """
    k = (key or "").strip().lower()
    c = (content or "").strip()

    if _PATH_TRAVERSAL.search(k):
        return (
            f"refused: key {key!r} contains path traversal sequences. "
            "Memory keys must be plain identifiers, not file paths."
        )

    if k in _DENY_KEYS:
        return (
            f"refused: key {key!r} looks like a safety/triage flag, not a "
            "user fact. The remember tool stores facts the user states "
            "about themselves or their work, not judgments about them."
        )

    if _USER_CHARACTERIZATION.search(c) and _JUDGMENT_VOCAB.search(c):
        return (
            "refused: content reads as a judgment about the user's intent "
            "rather than a fact they stated. The remember tool does not "
            "record safety assessments or characterizations of the user."
        )

    if _INTENT_RESTATEMENT.search(c):
        return (
            "refused: content restates the user's current request "
            f"({content!r}) instead of recording a durable fact. The "
            "remember tool is not a per-turn activity log."
        )

    return None
