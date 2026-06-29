"""Tests for the MLX runtime helpers and prompt rules."""

from towel.agent.conversation import Conversation, Role
from towel.agent.runtime import (
    EMPTY_TEXT_FALLBACK,
    AgentRuntime,
    GenerationResult,
    format_tool_feedback,
    mlx_tokenizer_config,
    tool_result_is_error,
)
from towel.config import TowelConfig
from towel.skills.base import Skill, ToolDefinition
from towel.skills.registry import SkillRegistry


class TestMlxTokenizerConfig:
    def test_returns_empty_config(self):
        # ``fix_mistral_regex`` was dropped because newer transformers patch
        # the regex internally and reject the kwarg as a duplicate argument.
        assert mlx_tokenizer_config() == {}


class TestToolFeedback:
    def test_classifies_common_tool_errors(self):
        assert tool_result_is_error("File not found: /tmp/missing.txt")
        assert tool_result_is_error("Unknown tool: read_files")
        assert tool_result_is_error("Not a directory: /tmp")
        assert tool_result_is_error("HTTP error: 503 backend down")
        assert tool_result_is_error("[404] resource missing")
        assert not tool_result_is_error("Written 12 bytes to /tmp/test.txt")

    def test_classifies_canonical_skill_error_format(self):
        """41 places across src/towel/skills/ return errors as
        f"Error: {e}", f"Error reading X: ...", f"Error creating
        X: ...", etc. The old patterns only caught
        "Error executing" / "Error calling" — every other skill
        error was getting silently classified as a successful result,
        and the agent then told the model "Use this result to answer
        the user concretely" on top of a literal "Error: file not
        found" string.

        The broader ^Error\\b catch-all is the fix; pin a sample of
        the real skill-return shapes here so the regression doesn't
        sneak back in if someone tightens the pattern again."""
        assert tool_result_is_error("Error: file not found")
        assert tool_result_is_error("Error: git failed")
        assert tool_result_is_error("Error reading /tmp/x.png: invalid")
        assert tool_result_is_error("Error creating archive: disk full")
        assert tool_result_is_error("Error checking host:443: timeout")
        assert tool_result_is_error("Error extracting: bad zip")
        assert tool_result_is_error("Error executing tool: boom")  # still works
        assert tool_result_is_error("Error calling Claude: 401")     # still works
        # Permission denied is a common os-level error not previously
        # caught — skills returning the bare OSError string would have
        # passed through as a success.
        assert tool_result_is_error("Permission denied: /etc/shadow")
        # \b after Error keeps "Errors" (a plain-English noun) from
        # being mislabeled — important so a tool that legitimately
        # returns analytics-style output isn't every-result-erroring.
        assert not tool_result_is_error("Errors per page: 3")
        assert not tool_result_is_error("Errored tasks summary follows...")

    def test_formats_recovery_guidance_for_errors(self):
        text = format_tool_feedback("read_file", "File not found: x.txt", is_error=True)
        assert "[read_file]" in text
        assert "status: error" in text
        # Current policy is conservative: tell the model to stop retrying after
        # an error rather than burning iterations on the same failing tool.
        assert "Do NOT retry the same tool" in text

    def test_formats_recovery_guidance_for_success(self):
        text = format_tool_feedback("read_file", "ok: 3 lines", is_error=False)
        assert "status: ok" in text
        assert "result:\nok: 3 lines" in text
        assert "Use this result to answer the user concretely" in text

    def test_typo_tool_name_gets_retry_guidance(self):
        # When the registry returns "Unknown tool: X. Did you mean: Y?",
        # the model should retry with the corrected name — NOT back off.
        text = format_tool_feedback(
            "read_files",
            "Error executing read_files: Unknown tool: read_files. Did you mean: read_file?",
            is_error=True,
        )
        assert "Retry ONCE with the corrected tool name" in text
        assert "Do NOT retry" not in text

    def test_unrecoverable_error_keeps_conservative_guidance(self):
        text = format_tool_feedback(
            "read_file", "File not found: /nope.txt", is_error=True
        )
        assert "Do NOT retry the same tool" in text


class TestSystemPrompt:
    def test_system_prompt_requires_reporting_results_back(self):
        config = TowelConfig(identity="You are Towel.")
        runtime = AgentRuntime(config)

        system = runtime._build_system_content()

        assert "always answer the user's original question" in system
        assert "explicitly report that back to the user" in system

    def test_system_prompt_includes_tool_recovery_rules(self):
        class DummySkill(Skill):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def description(self) -> str:
                return "dummy"

            def tools(self) -> list[ToolDefinition]:
                return [
                    ToolDefinition(
                        name="read_file",
                        description="Read a file",
                        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                    )
                ]

            async def execute(self, tool_name: str, arguments: dict):
                return "ok"

        config = TowelConfig(identity="You are Towel.")
        skills = SkillRegistry()
        skills.register(DummySkill())
        runtime = AgentRuntime(config, skills=skills)

        system = runtime._build_system_content()

        assert "prefer emitting just the tool call" in system
        assert "one corrected retry" in system

    def test_auto_context_uses_minimum_for_short_turn(self, monkeypatch):
        config = TowelConfig(identity="You are Towel.")
        config.model.context_window = 262144
        config.model.min_context_window = 32768
        config.model.auto_context = True
        runtime = AgentRuntime(config)
        conv = __import__("towel.agent.conversation", fromlist=["Conversation"]).Conversation()
        conv.add(__import__("towel.agent.conversation", fromlist=["Role"]).Role.USER, "hi")

        captured = {}

        def fake_fit(*, context_window, **kwargs):
            captured["context_window"] = context_window
            return kwargs["messages"], __import__(
                "towel.agent.context", fromlist=["ContextBudget"]
            ).ContextBudget(context_window=context_window, max_output_tokens=512)

        monkeypatch.setattr("towel.agent.runtime.fit_messages", fake_fit)
        runtime._build_prompt(conv)

        assert captured["context_window"] == 32768


class _ReadFileSkill(Skill):
    @property
    def name(self) -> str:
        return "fs"

    @property
    def description(self) -> str:
        return "fs"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ]

    async def execute(self, tool_name: str, arguments: dict):
        return "ok"


class TestNativeToolsChannel:
    def _runtime_with_one_tool(self) -> AgentRuntime:
        config = TowelConfig(identity="You are Towel.")
        skills = SkillRegistry()
        skills.register(_ReadFileSkill())
        return AgentRuntime(config, skills=skills)

    def test_native_path_omits_inline_tool_listing(self):
        runtime = self._runtime_with_one_tool()
        native = runtime._build_system_content(include_tools_section=False)
        # The per-tool bullet and call-format spec are gone…
        assert "- read_file" not in native
        assert "<tool_call>" not in native
        # …but behavioral guardrails remain.
        assert "Only call tools from the provided list" in native
        assert "prefer emitting just the tool call" in native
        assert "one corrected retry" in native

    def test_tools_for_chat_template_returns_openai_function_dicts(self):
        runtime = self._runtime_with_one_tool()
        tools = runtime._tools_for_chat_template()
        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

    def test_detect_native_tools_support_true_when_template_renders_tools(self):
        runtime = self._runtime_with_one_tool()

        class FakeTokenizer:
            def apply_chat_template(self, messages, tools=None, **kwargs):
                if tools:
                    names = ",".join(t["function"]["name"] for t in tools)
                    return f"<system>tools={names}</system><user>{messages[-1]['content']}</user>"
                return f"<user>{messages[-1]['content']}</user>"

        runtime._tokenizer = FakeTokenizer()
        assert runtime._detect_native_tools_support() is True

    def test_detect_native_tools_support_false_when_template_ignores_tools(self):
        runtime = self._runtime_with_one_tool()

        class IgnoresToolsTokenizer:
            def apply_chat_template(self, messages, tools=None, **kwargs):
                # Older templates silently drop the tools kwarg.
                return f"<user>{messages[-1]['content']}</user>"

        runtime._tokenizer = IgnoresToolsTokenizer()
        assert runtime._detect_native_tools_support() is False

    def test_detect_native_tools_support_false_when_template_raises(self):
        runtime = self._runtime_with_one_tool()

        class RaisesTokenizer:
            def apply_chat_template(self, messages, tools=None, **kwargs):
                if tools is not None:
                    raise TypeError("got unexpected keyword 'tools'")
                return f"<user>{messages[-1]['content']}</user>"

        runtime._tokenizer = RaisesTokenizer()
        assert runtime._detect_native_tools_support() is False


class TestEmptyTextFallback:
    def test_step_replaces_empty_final_text(self):
        import asyncio

        config = TowelConfig(identity="You are Towel.")
        runtime = AgentRuntime(config)

        async def fake_generate(_conversation):
            return GenerationResult(text="", total_tokens=0)

        runtime.generate = fake_generate
        conv = Conversation()
        conv.add(Role.USER, "hello")

        msg = asyncio.run(runtime.step(conv))

        assert msg.content == EMPTY_TEXT_FALLBACK
        assert msg.metadata["empty_text_fallback"] is True

    def test_step_streaming_replaces_empty_final_text(self):
        import asyncio

        config = TowelConfig(identity="You are Towel.")
        runtime = AgentRuntime(config)

        async def fake_stream(_conversation):
            if False:
                yield ""

        runtime.stream = fake_stream
        conv = Conversation()
        conv.add(Role.USER, "hello")

        async def collect():
            return [event async for event in runtime.step_streaming(conv)]

        events = asyncio.run(collect())

        assert events[-1].type.value == "response_complete"
        assert events[-1].data["content"] == EMPTY_TEXT_FALLBACK
        assert conv.messages[-1].content == EMPTY_TEXT_FALLBACK


class TestRegistrySuggestions:
    def test_unknown_tool_error_suggests_close_match(self):
        class DummySkill(Skill):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def description(self) -> str:
                return "dummy"

            def tools(self) -> list[ToolDefinition]:
                return [
                    ToolDefinition(
                        name="read_file",
                        description="Read a file",
                    )
                ]

            async def execute(self, tool_name: str, arguments: dict):
                return "ok"

        skills = SkillRegistry()
        skills.register(DummySkill())

        assert skills.suggest_tools("read_files") == ["read_file"]
