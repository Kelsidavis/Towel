"""Auto-title generation for conversations.

Generates concise, descriptive titles from the first user message
without an extra model call. Uses keyword extraction heuristics.
"""

from __future__ import annotations

import re

# Common filler words to skip
STOP_WORDS = frozenset("""
a an the is are was were be been being have has had do does did will would
shall should may might can could must need dare ought to of in for on at by
with from as into through during before after above below between out off
over under again further then once here there when where why how all each
every both few more most other some such no nor not only own same so than
too very just don't doesn't didn't won't wouldn't shan't shouldn't can't
couldn't mustn't let's that's who's what's here's there's when's where's
why's how's i me my myself we our ours ourselves you your yours yourself
yourselves he him his himself she her hers herself it its itself they them
their theirs themselves what which who whom this that these those am about
up please could help me tell explain show give make write create find get
want like know think also really
""".split())

MAX_TITLE_WORDS = 6
MIN_TITLE_WORDS = 2


def generate_title(user_message: str, assistant_message: str = "") -> str:
    """Generate a short title from the first exchange.

    Returns an empty string if no meaningful title can be generated.
    """
    text = user_message

    # Strip @file references
    text = re.sub(r"@[\w./~*?:-]+", "", text)

    # Strip code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Strip inline code
    text = re.sub(r"`[^`]+`", "", text)

    # Strip URLs
    text = re.sub(r"https?://\S+", "", text)

    # Strip special characters but keep word boundaries
    text = re.sub(r"[^\w\s-]", " ", text)

    # Split into words and filter
    words = text.lower().split()
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 1]

    if len(keywords) < MIN_TITLE_WORDS:
        # Fall back to first meaningful words from original, still filtering stops
        words = re.sub(r"[^\w\s]", " ", user_message).lower().split()
        keywords = [w for w in words if w not in STOP_WORDS and len(w) > 1][:MAX_TITLE_WORDS]

    if not keywords:
        # Last resort: just take first non-trivial words
        words = re.sub(r"[^\w\s]", " ", user_message).lower().split()
        keywords = [w for w in words if len(w) > 2][:MAX_TITLE_WORDS]

    if not keywords:
        return ""

    # Take the first N keywords and title-case them
    title_words = keywords[:MAX_TITLE_WORDS]
    title = " ".join(w.capitalize() for w in title_words)

    return title
