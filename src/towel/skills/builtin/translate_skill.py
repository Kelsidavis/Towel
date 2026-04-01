"""Translate skill — detect language and translate text."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition

# Common language codes for detection heuristic
_LANG_MARKERS = {
    "es": ["el", "la", "los", "las", "de", "en", "que", "por", "para", "como", "pero", "más"],
    "fr": ["le", "la", "les", "des", "est", "une", "que", "dans", "pour", "avec", "pas", "sur"],
    "de": ["der", "die", "das", "ist", "ein", "und", "nicht", "von", "mit", "auf", "für", "den"],
    "pt": ["o", "a", "os", "as", "de", "em", "que", "para", "com", "não", "uma", "por"],
    "it": ["il", "la", "di", "che", "è", "per", "un", "una", "con", "non", "sono", "del"],
    "ja": ["の", "は", "を", "に", "が", "で", "た", "と", "も", "し"],
    "zh": ["的", "了", "在", "是", "我", "不", "这", "有", "人", "他"],
    "ko": ["은", "는", "이", "가", "를", "을", "에", "의", "로", "와"],
}


def _detect_lang(text: str) -> str:
    words = text.lower().split()[:100]
    scores: dict[str, int] = {}
    for lang, markers in _LANG_MARKERS.items():
        scores[lang] = sum(1 for w in words if w in markers)
    if not scores:
        return "en"
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= 3 else "en"


class TranslateSkill(Skill):
    @property
    def name(self) -> str:
        return "translate"

    @property
    def description(self) -> str:
        return "Detect language and prepare text for translation"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="detect_language",
                description="Detect the language of text",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze"},
                    },
                    "required": ["text"],
                },
            ),
            ToolDefinition(
                name="translation_prompt",
                description="Generate a translation prompt for the agent to execute",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to translate"},
                        "target_language": {
                            "type": "string",
                            "description": "Target language (e.g., Spanish, French, Japanese)",
                        },
                        "tone": {
                            "type": "string",
                            "description": "Tone: formal, casual, technical (default: neutral)",
                        },
                    },
                    "required": ["text", "target_language"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "detect_language":
                return self._detect(arguments["text"])
            case "translation_prompt":
                return self._prompt(
                    arguments["text"],
                    arguments["target_language"],
                    arguments.get("tone", "neutral"),
                )
            case _:
                return f"Unknown tool: {tool_name}"

    def _detect(self, text: str) -> str:
        lang = _detect_lang(text)
        names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "pt": "Portuguese",
            "it": "Italian",
            "ja": "Japanese",
            "zh": "Chinese",
            "ko": "Korean",
        }
        return f"Detected language: {names.get(lang, lang)} ({lang})"

    def _prompt(self, text: str, target: str, tone: str) -> str:
        source = _detect_lang(text)
        names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "pt": "Portuguese",
            "it": "Italian",
            "ja": "Japanese",
            "zh": "Chinese",
            "ko": "Korean",
        }
        src_name = names.get(source, source)
        tone_inst = f" Use a {tone} tone." if tone != "neutral" else ""
        return (
            f"Translate the following from {src_name} to {target}.{tone_inst} "
            f"Output only the translation.\n\n{text}"
        )
