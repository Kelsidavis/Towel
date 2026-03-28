"""Chat slash commands — in-session control without leaving the chat.

Commands:
  /help                Show available commands
  /info                Show current session info
  /clear               Clear conversation history (start fresh)
  /agent [name]        Switch agent profile (or show current)
  /agents              List available agent profiles
  /memory              Show all memories
  /remember <k> <v>    Add a memory
  /forget <key>        Remove a memory
  /rename <title>      Set a title for this conversation
  /export [file]       Export current conversation to markdown
  /t <template> <input>  Apply a prompt template
  /templates           List available templates
  /newagent <n> <model> <prompt>  Create a new agent profile
  /delagent <name>     Delete a user-created agent
  /context             Show loaded project context (.towel.md)
  /system <prompt>     Override the system prompt
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from towel.agent.conversation import Conversation
    from towel.agent.runtime import AgentRuntime
    from towel.config import TowelConfig
    from towel.memory.store import MemoryStore
    from towel.persistence.store import ConversationStore

console = Console()


HELP_TEXT = """[bold]Chat commands:[/bold]

  [green]/help[/green]                 Show this help
  [green]/info[/green]                 Show session info
  [green]/stats[/green]                Show session statistics (tokens, speed, cost)
  [green]/undo[/green]                 Remove last exchange (your message + response)
  [green]/retry[/green]                Remove last response and regenerate
  [green]/fork[/green]                 Save current conversation and start a branch
  [green]/clear[/green]                Clear conversation history
  [green]/agent[/green] [name]         Switch agent / show current
  [green]/agents[/green]               List available agents
  [green]/memory[/green]               Show all memories
  [green]/remember[/green] <key> <val> Save a memory
  [green]/forget[/green] <key>         Remove a memory
  [green]/t[/green] <template> <input>  Apply a prompt template (e.g., /t review code here)
  [green]/templates[/green]            List available templates
  [green]/compact[/green] [n]            Summarize old messages to free context (keep last n, default 4)
  [green]/diff[/green] <id>             Compare this conversation with a saved one
  [green]/grep[/green] <query>          Search within this conversation
  [green]/pin[/green]                  Pin last response (stays in context even when old msgs drop)
  [green]/pins[/green]                 List pinned messages
  [green]/report[/green]               Generate a session summary (topics, tools, code, decisions)
  [green]/save[/green] <file> [n]       Save code block from last response to a file (n=block index)
  [green]/copy[/green] [code]           Copy last response to clipboard (or just code blocks)
  [green]/tag[/green] <name>            Add a tag to this conversation (or remove with -name)
  [green]/tags[/green]                 Show tags on this conversation
  [green]/history[/green] [n]           Browse recent conversations (default: 10)
  [green]/resume[/green] <id>          Switch to a saved conversation
  [green]/rename[/green] <title>       Set a title for this conversation
  [green]/export[/green] [file]        Export conversation to markdown (.html for HTML)
  [green]/newagent[/green] <name> <model> <prompt>  Create new agent
  [green]/delagent[/green] <name>      Delete user agent
  [green]/context[/green]              Show loaded .towel.md project context
  [green]/snippet[/green] <n> <text>    Save a reusable text snippet
  [green]/snippets[/green]             List all snippets
  [green]/s[/green] <name>              Insert a snippet into your message
  [green]/alias[/green] <name> <prompt> Create a prompt shortcut (e.g., /alias review Review this code)
  [green]/aliases[/green]              List all defined aliases
  [green]/unalias[/green] <name>       Remove an alias
  [green]/whoami[/green]              Show agent identity, model, and context budget
  [green]/delegate[/green] <role> <task> Delegate to specialist (coder, reviewer, architect...)
  [green]/health[/green]              Show agent health and error counts
  [green]/loop[/green] <interval> <prompt> Run a prompt on a recurring interval (e.g., /loop 5m check status)
  [green]/system[/green] <prompt>      Override the system prompt
"""


class SlashContext:
    """Mutable context passed to slash command handlers."""

    def __init__(
        self,
        config: TowelConfig,
        conv: Conversation,
        agent: AgentRuntime,
        memory: MemoryStore,
        store: ConversationStore,
    ) -> None:
        self.config = config
        self.conv = conv
        self.agent = agent
        self.memory = memory
        self.store = store
        self.current_agent_name: str | None = None


def handle_slash(user_input: str, ctx: SlashContext) -> bool | None:
    """Handle a slash command.

    Returns:
        True  — command consumed, skip agent
        False — command added message to conv, run agent step
        None  — not a slash command
    """
    stripped = user_input.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    match cmd:
        case "/help":
            console.print(HELP_TEXT)

        case "/info":
            _cmd_info(ctx)

        case "/stats":
            _cmd_stats(ctx)

        case "/undo":
            _cmd_undo(ctx)

        case "/retry":
            return _cmd_retry(ctx)

        case "/fork":
            _cmd_fork(ctx, arg)

        case "/clear":
            _cmd_clear(ctx)

        case "/agent":
            _cmd_agent(ctx, arg)

        case "/agents":
            _cmd_agents(ctx)

        case "/memory":
            _cmd_memory(ctx)

        case "/remember":
            _cmd_remember(ctx, arg)

        case "/forget":
            _cmd_forget(ctx, arg)

        case "/t":
            return _cmd_template(ctx, arg)  # returns False to send to agent

        case "/templates":
            _cmd_templates(ctx)

        case "/compact":
            _cmd_compact(ctx, arg)

        case "/diff":
            _cmd_diff(ctx, arg)

        case "/grep":
            _cmd_grep(ctx, arg)

        case "/pin":
            _cmd_pin(ctx, arg)

        case "/pins":
            _cmd_pins(ctx)

        case "/report":
            _cmd_report(ctx)

        case "/save":
            _cmd_save(ctx, arg)

        case "/copy":
            _cmd_copy(ctx, arg)

        case "/tag":
            _cmd_tag(ctx, arg)

        case "/tags":
            _cmd_tags(ctx)

        case "/history":
            _cmd_history(ctx, arg)

        case "/resume":
            _cmd_resume(ctx, arg)

        case "/rename":
            _cmd_rename(ctx, arg)

        case "/export":
            _cmd_export(ctx, arg)

        case "/newagent":
            _cmd_newagent(ctx, arg)

        case "/delagent":
            _cmd_delagent(ctx, arg)

        case "/context":
            _cmd_context(ctx)

        case "/delegate":
            return _cmd_delegate(ctx, arg)

        case "/whoami":
            _cmd_whoami(ctx)

        case "/health":
            _cmd_health(ctx)

        case "/loop":
            _cmd_loop(ctx, arg)

        case "/system":
            _cmd_system(ctx, arg)

        case "/snippet":
            _cmd_snippet(ctx, arg)

        case "/snippets":
            _cmd_snippets(ctx)

        case "/s":
            return _cmd_use_snippet(ctx, arg)

        case "/alias":
            _cmd_alias(ctx, arg)

        case "/aliases":
            _cmd_aliases(ctx)

        case "/unalias":
            _cmd_unalias(ctx, arg)

        case _:
            # Check if it's a user-defined alias
            alias_result = _try_alias(ctx, cmd, arg)
            if alias_result is not None:
                return alias_result
            console.print(f"[red]Unknown command:[/red] {cmd}")
            console.print("[dim]Type /help for available commands[/dim]")

    return True


def _cmd_info(ctx: SlashContext) -> None:
    from towel.agent.project import find_project_contexts

    console.print(f"  [green]Session:[/green] {ctx.conv.id}")
    console.print(f"  [green]Messages:[/green] {len(ctx.conv)}")
    console.print(f"  [green]Channel:[/green] {ctx.conv.channel}")
    console.print(f"  [green]Model:[/green] {ctx.config.model.name}")
    console.print(f"  [green]Agent:[/green] {ctx.current_agent_name or 'default'}")
    console.print(f"  [green]Memories:[/green] {ctx.memory.count}")
    console.print(f"  [green]Context window:[/green] {ctx.config.model.context_window} tokens")

    contexts = find_project_contexts()
    if contexts:
        console.print(f"  [green]Project context:[/green] {len(contexts)} file(s)")
        for p in contexts:
            console.print(f"    [dim]{p}[/dim]")
    else:
        console.print(f"  [green]Project context:[/green] [dim]none (create .towel.md)[/dim]")


def _cmd_stats(ctx: SlashContext) -> None:
    from towel.agent.conversation import Role

    msgs = ctx.conv.messages
    if not msgs:
        console.print("[dim]No messages yet.[/dim]")
        return

    user_msgs = [m for m in msgs if m.role == Role.USER]
    asst_msgs = [m for m in msgs if m.role == Role.ASSISTANT]
    tool_msgs = [m for m in msgs if m.role == Role.TOOL]

    total_tokens = 0
    tps_values: list[float] = []
    for m in asst_msgs:
        t = m.metadata.get("tokens", 0)
        if t:
            total_tokens += t
        s = m.metadata.get("tps", 0)
        if s:
            tps_values.append(s)

    # Character counts
    user_chars = sum(len(m.content) for m in user_msgs)
    asst_chars = sum(len(m.content) for m in asst_msgs)

    # Duration
    if len(msgs) >= 2:
        duration = msgs[-1].timestamp - msgs[0].timestamp
        mins = duration.total_seconds() / 60
        duration_str = f"{mins:.1f} min" if mins >= 1 else f"{duration.total_seconds():.0f} sec"
    else:
        duration_str = "—"

    console.print("[bold]Session statistics:[/bold]")
    console.print(f"  [green]Messages:[/green]     {len(user_msgs)} you, {len(asst_msgs)} towel, {len(tool_msgs)} tool")
    console.print(f"  [green]Characters:[/green]   {user_chars:,} in, {asst_chars:,} out")
    console.print(f"  [green]Tokens out:[/green]   {total_tokens:,}")
    if tps_values:
        avg_tps = sum(tps_values) / len(tps_values)
        max_tps = max(tps_values)
        console.print(f"  [green]Speed:[/green]        {avg_tps:.1f} tok/s avg, {max_tps:.1f} tok/s peak")
    console.print(f"  [green]Duration:[/green]     {duration_str}")
    console.print(f"  [green]Model:[/green]        {ctx.config.model.name}")

    # Cost comparison: show how much this would cost on cloud APIs
    if total_tokens > 0:
        # Rough estimates per 1M output tokens (as of 2026)
        cloud_costs = {
            "GPT-4o": 10.0,
            "Claude Sonnet": 15.0,
            "Claude Opus": 75.0,
        }
        console.print(f"\n  [dim]Cloud API cost comparison ({total_tokens:,} output tokens):[/dim]")
        for provider, cost_per_m in cloud_costs.items():
            est = (total_tokens / 1_000_000) * cost_per_m
            console.print(f"    [dim]{provider}: ~${est:.4f}[/dim]")
        console.print(f"    [green]Towel (local): $0.00[/green]")


def _cmd_undo(ctx: SlashContext) -> None:
    """Remove the last exchange (assistant response + user message that triggered it)."""
    from towel.agent.conversation import Role

    if not ctx.conv.messages:
        console.print("[dim]Nothing to undo.[/dim]")
        return

    removed = 0
    # Remove trailing tool/assistant messages
    while ctx.conv.messages and ctx.conv.messages[-1].role in (Role.ASSISTANT, Role.TOOL):
        ctx.conv.messages.pop()
        removed += 1

    # Remove the user message that triggered them
    if ctx.conv.messages and ctx.conv.messages[-1].role == Role.USER:
        ctx.conv.messages.pop()
        removed += 1

    console.print(f"[green]Undid {removed} message(s).[/green]")


def _cmd_retry(ctx: SlashContext) -> bool | None:
    """Remove last response and re-run the agent on the same user message.

    Returns False to signal the caller to run an agent step (the user
    message is already in the conversation).
    """
    from towel.agent.conversation import Role

    if not ctx.conv.messages:
        console.print("[dim]Nothing to retry.[/dim]")
        return True  # consumed, nothing to do

    # Remove trailing tool/assistant messages
    removed = 0
    while ctx.conv.messages and ctx.conv.messages[-1].role in (Role.ASSISTANT, Role.TOOL):
        ctx.conv.messages.pop()
        removed += 1

    if removed == 0:
        console.print("[dim]No assistant response to retry.[/dim]")
        return True

    # Check there's still a user message to re-run
    if not ctx.conv.messages or ctx.conv.messages[-1].role != Role.USER:
        console.print("[dim]No user message to retry.[/dim]")
        return True

    last_user = ctx.conv.messages[-1].content
    preview = last_user[:60] + "..." if len(last_user) > 60 else last_user
    console.print(f"[green]Retrying:[/green] {preview}")

    # Return False to signal "run agent step" — the user message is already in conv
    return False


def _cmd_fork(ctx: SlashContext, arg: str) -> None:
    """Save current conversation and start a new branch with the same history."""
    import uuid
    from towel.agent.conversation import Conversation, Message

    if not ctx.conv.messages:
        console.print("[dim]Nothing to fork (conversation is empty).[/dim]")
        return

    # Save the current conversation first
    old_id = ctx.conv.id
    old_title = ctx.conv.display_title
    ctx.store.save(ctx.conv)

    # Create a new conversation with a copy of all messages
    new_id = uuid.uuid4().hex[:16]
    fork_title = arg.strip() if arg.strip() else f"Fork of {old_title}"

    new_conv = Conversation(
        id=new_id,
        title=fork_title,
        channel=ctx.conv.channel,
        created_at=ctx.conv.created_at,
    )
    for msg in ctx.conv.messages:
        new_conv.messages.append(Message(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            metadata=dict(msg.metadata),
            id=msg.id,
        ))

    # Switch to the fork
    ctx.conv.id = new_conv.id
    ctx.conv.title = new_conv.title
    ctx.conv.messages = new_conv.messages

    console.print(f"[green]Forked![/green] Original saved as [dim]{old_id}[/dim]")
    console.print(f"  New branch: [bold]{fork_title}[/bold] ({new_id})")
    console.print(f"  {len(ctx.conv)} messages carried over")
    console.print(f"[dim]Resume original later: towel resume {old_id}[/dim]")


def _cmd_clear(ctx: SlashContext) -> None:
    from towel.agent.conversation import Conversation
    count = len(ctx.conv)
    ctx.conv.messages.clear()
    console.print(f"[green]Cleared {count} messages.[/green] Conversation reset.")


def _cmd_agent(ctx: SlashContext, arg: str) -> None:
    if not arg:
        console.print(f"  Current agent: [green]{ctx.current_agent_name or 'default'}[/green]")
        console.print(f"  Model: {ctx.config.model.name}")
        console.print(f"  [dim]Switch with: /agent <name>[/dim]")
        return

    name = arg.strip()
    profile = ctx.config.get_agent(name)
    if not profile:
        console.print(f"[red]Unknown agent:[/red] {name}")
        console.print("[dim]List agents with: /agents[/dim]")
        return

    ctx.config.model = profile.model
    ctx.config.identity = profile.effective_identity(ctx.config.identity)
    ctx.current_agent_name = name
    ctx.agent._loaded = False  # force model reload on next generation
    ctx.agent.config = ctx.config
    console.print(f"[green]Switched to agent:[/green] {name}")
    console.print(f"  Model: {profile.model.name}")
    if profile.description:
        console.print(f"  {profile.description}")
    console.print("[dim]Note: model will reload on next message.[/dim]")


def _cmd_agents(ctx: SlashContext) -> None:
    agents = ctx.config.list_agents()
    if not agents:
        console.print("[dim]No agents configured.[/dim]")
        return
    console.print("[bold]Available agents:[/bold]")
    for name, profile in agents.items():
        marker = " [green]<--[/green]" if name == ctx.current_agent_name else ""
        console.print(f"  [green]{name}[/green]{marker}")
        if profile.description:
            console.print(f"    {profile.description}")
        console.print(f"    [dim]{profile.model.name}[/dim]")


def _cmd_memory(ctx: SlashContext) -> None:
    entries = ctx.memory.recall_all()
    if not entries:
        console.print("[dim]No memories stored.[/dim]")
        return
    console.print(f"[bold]Memories ({len(entries)}):[/bold]")
    for e in entries:
        console.print(f"  [green]{e.key}[/green] [dim][{e.memory_type}][/dim] {e.content}")


def _cmd_remember(ctx: SlashContext, arg: str) -> None:
    parts = arg.split(None, 1)
    if len(parts) < 2:
        console.print("[red]Usage:[/red] /remember <key> <value>")
        return
    key, value = parts[0], parts[1]
    entry = ctx.memory.remember(key, value)
    console.print(f"[green]Remembered:[/green] {entry.key}: {entry.content}")


def _cmd_forget(ctx: SlashContext, arg: str) -> None:
    key = arg.strip()
    if not key:
        console.print("[red]Usage:[/red] /forget <key>")
        return
    if ctx.memory.forget(key):
        console.print(f"[green]Forgot:[/green] {key}")
    else:
        console.print(f"[red]Not found:[/red] {key}")


def _cmd_template(ctx: SlashContext, arg: str) -> bool:
    """Apply a template. Returns False so the expanded text gets sent to the agent."""
    from towel.templates.engine import TemplateEngine

    parts = arg.split(None, 1)
    if not parts:
        console.print("[red]Usage:[/red] /t <template> <input>")
        console.print("[dim]List templates: /templates[/dim]")
        return True  # consumed, don't send

    template_name = parts[0]
    input_text = parts[1] if len(parts) > 1 else ""

    engine = TemplateEngine()
    rendered = engine.render(template_name, input_text=input_text)
    if rendered is None:
        console.print(f"[red]Unknown template:[/red] {template_name}")
        return True

    # Expand @file references in the rendered template
    from towel.agent.refs import expand_refs
    rendered = expand_refs(rendered)

    console.print(f"[dim]  template: {template_name}[/dim]")

    # Inject into conversation — the caller will check for False and send it
    from towel.agent.conversation import Role
    ctx.conv.add(Role.USER, rendered)

    # Return False to signal "this is NOT consumed — run agent step"
    return False


def _cmd_templates(ctx: SlashContext) -> None:
    from towel.templates.engine import TemplateEngine
    engine = TemplateEngine()
    tpls = engine.list_templates()
    if not tpls:
        console.print("[dim]No templates available.[/dim]")
        return
    console.print("[bold]Templates:[/bold]")
    for name, desc in tpls.items():
        console.print(f"  [green]{name}[/green] — {desc}")


def _cmd_newagent(ctx: SlashContext, arg: str) -> None:
    parts = arg.split(None, 2)
    if len(parts) < 2:
        console.print("[red]Usage:[/red] /newagent <name> <model> [identity]")
        console.print("  Example: /newagent mybot mlx-community/Llama-3.2-8B-Instruct-4bit You are a helpful bot.")
        return

    name = parts[0]
    model = parts[1]
    identity = parts[2] if len(parts) > 2 else f"You are Towel ({name} mode). Don't Panic."

    if ctx.config.get_agent(name):
        console.print(f"[yellow]Agent '{name}' already exists.[/yellow]")
        return

    from towel.cli.agent_mgr import create_agent
    profile = create_agent(name=name, model_name=model, identity=identity)
    console.print(f"[green]Created agent:[/green] {name}")
    console.print(f"  Model: {profile.model.name}")
    console.print(f"[dim]Switch with: /agent {name}[/dim]")


def _cmd_delagent(ctx: SlashContext, arg: str) -> None:
    name = arg.strip()
    if not name:
        console.print("[red]Usage:[/red] /delagent <name>")
        return

    from towel.config import DEFAULT_AGENTS
    if name in DEFAULT_AGENTS:
        console.print(f"[yellow]Cannot delete built-in agent '{name}'.[/yellow]")
        return

    from towel.cli.agent_mgr import delete_agent
    if delete_agent(name):
        console.print(f"[green]Deleted agent:[/green] {name}")
    else:
        console.print(f"[red]Agent not found:[/red] {name}")


def _cmd_compact(ctx: SlashContext, arg: str) -> None:
    """Compress older messages into a summary to free context window space.

    Keeps the last N messages intact and replaces everything before them
    with a condensed summary. Pinned messages are always preserved.
    """
    import re
    from towel.agent.conversation import Role, Message

    keep_count = int(arg.strip()) if arg.strip().isdigit() else 4
    msgs = ctx.conv.messages

    if len(msgs) <= keep_count:
        console.print(f"[dim]Only {len(msgs)} messages — nothing to compact.[/dim]")
        return

    # Split into old (to compress) and recent (to keep)
    # But always preserve pinned messages
    old_msgs = msgs[:-keep_count]
    recent_msgs = msgs[-keep_count:]

    pinned_old = [m for m in old_msgs if m.pinned]
    compressible = [m for m in old_msgs if not m.pinned]

    if not compressible:
        console.print("[dim]All old messages are pinned — nothing to compact.[/dim]")
        return

    # Build a condensed summary from old messages
    summary_parts: list[str] = []
    summary_parts.append(f"[Compacted summary of {len(compressible)} earlier messages]\n")

    for msg in compressible:
        role = msg.role.value
        content = msg.content

        if msg.role == Role.TOOL:
            # Just note the tool was called
            if content.startswith("[") and "]" in content:
                tool_name = content[1:content.index("]")]
                summary_parts.append(f"- Tool: {tool_name}")
            continue

        if msg.role == Role.SYSTEM:
            continue

        # Extract first meaningful line
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        first_line = lines[0][:120] if lines else ""

        # Extract code blocks (preserve them fully — they're high-value)
        code_blocks = re.findall(r"```\w*\n(.*?)```", content, re.DOTALL)

        label = "Q" if msg.role == Role.USER else "A"
        summary_parts.append(f"- {label}: {first_line}")

        for block in code_blocks[:2]:  # max 2 code blocks per message
            trimmed = block.strip()
            if len(trimmed) > 300:
                trimmed = trimmed[:300] + "\n..."
            summary_parts.append(f"  ```\n  {trimmed}\n  ```")

    summary_text = "\n".join(summary_parts)

    # Replace old messages with summary + pinned
    summary_msg = Message(
        role=Role.SYSTEM,
        content=summary_text,
        metadata={"compacted": True, "original_count": len(compressible)},
    )

    ctx.conv.messages = [summary_msg] + pinned_old + recent_msgs

    old_chars = sum(len(m.content) for m in compressible)
    new_chars = len(summary_text)
    saved_pct = ((old_chars - new_chars) / old_chars * 100) if old_chars > 0 else 0

    console.print(f"[green]Compacted {len(compressible)} messages into summary.[/green]")
    console.print(f"  {old_chars:,} chars → {new_chars:,} chars ({saved_pct:.0f}% reduction)")
    console.print(f"  Kept: {keep_count} recent + {len(pinned_old)} pinned")
    console.print(f"  Total messages: {len(ctx.conv.messages)}")


def _cmd_diff(ctx: SlashContext, arg: str) -> None:
    """Compare this conversation with a saved one, showing where they diverge."""
    conv_id = arg.strip()
    if not conv_id:
        console.print("[red]Usage:[/red] /diff <conversation_id>")
        console.print("[dim]Compare with a forked or saved conversation.[/dim]")
        return

    other = ctx.store.load(conv_id)
    if not other:
        console.print(f"[red]Conversation not found:[/red] {conv_id}")
        return

    current = ctx.conv
    cur_msgs = current.messages
    oth_msgs = other.messages

    # Find divergence point
    common = 0
    limit = min(len(cur_msgs), len(oth_msgs))
    while common < limit and cur_msgs[common].content == oth_msgs[common].content:
        common += 1

    console.print(f"[bold]Diff:[/bold] current vs [green]{conv_id}[/green]\n")
    console.print(f"  Shared history: {common} message(s)")
    console.print(f"  Current:  {len(cur_msgs)} total ({len(cur_msgs) - common} unique)")
    console.print(f"  Other:    {len(oth_msgs)} total ({len(oth_msgs) - common} unique)")

    if common == len(cur_msgs) and common == len(oth_msgs):
        console.print("\n  [green]Conversations are identical.[/green]")
        return

    console.print(f"\n[bold]Diverges after message {common}:[/bold]\n")

    # Show diverging messages side by side
    max_show = 5
    cur_unique = cur_msgs[common:common + max_show]
    oth_unique = oth_msgs[common:common + max_show]

    if cur_unique:
        console.print("  [cyan]Current branch:[/cyan]")
        for msg in cur_unique:
            preview = msg.content[:80].replace("\n", " ")
            if len(msg.content) > 80:
                preview += "..."
            role_color = {"user": "cyan", "assistant": "green", "tool": "yellow"}.get(msg.role.value, "dim")
            console.print(f"    [{role_color}]{msg.role.value}[/{role_color}] {preview}")
        if len(cur_msgs) - common > max_show:
            console.print(f"    [dim]... and {len(cur_msgs) - common - max_show} more[/dim]")

    if oth_unique:
        console.print(f"\n  [green]Other ({conv_id}):[/green]")
        for msg in oth_unique:
            preview = msg.content[:80].replace("\n", " ")
            if len(msg.content) > 80:
                preview += "..."
            role_color = {"user": "cyan", "assistant": "green", "tool": "yellow"}.get(msg.role.value, "dim")
            console.print(f"    [{role_color}]{msg.role.value}[/{role_color}] {preview}")
        if len(oth_msgs) - common > max_show:
            console.print(f"    [dim]... and {len(oth_msgs) - common - max_show} more[/dim]")


def _cmd_grep(ctx: SlashContext, arg: str) -> None:
    """Search within the current conversation for a query string."""
    import re

    query = arg.strip()
    if not query:
        console.print("[red]Usage:[/red] /grep <query>")
        return

    if not ctx.conv.messages:
        console.print("[dim]No messages to search.[/dim]")
        return

    try:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    except re.error:
        console.print("[red]Invalid search pattern.[/red]")
        return

    matches = 0
    for i, msg in enumerate(ctx.conv.messages):
        if pattern.search(msg.content):
            matches += 1
            role = msg.role.value
            role_color = {"user": "cyan", "assistant": "green", "tool": "yellow", "system": "dim"}.get(role, "white")

            # Extract snippet around match
            m = pattern.search(msg.content)
            start = max(0, m.start() - 40)
            end = min(len(msg.content), m.end() + 40)
            snippet = msg.content[start:end].replace("\n", " ")
            if start > 0:
                snippet = "..." + snippet
            if end < len(msg.content):
                snippet = snippet + "..."

            # Highlight match
            snippet = pattern.sub(lambda x: f"[bold yellow]{x.group()}[/bold yellow]", snippet)

            pin_mark = " [magenta]pinned[/magenta]" if msg.pinned else ""
            console.print(f"  [dim]#{i+1}[/dim] [{role_color}]{role}[/{role_color}]{pin_mark}  {snippet}")

    if matches == 0:
        console.print(f"[dim]No matches for:[/dim] {query}")
    else:
        console.print(f"\n[dim]{matches} match(es) found.[/dim]")


def _cmd_pin(ctx: SlashContext, arg: str) -> None:
    """Pin or unpin a message. Default: toggle pin on last assistant response."""
    from towel.agent.conversation import Role

    if not ctx.conv.messages:
        console.print("[dim]No messages to pin.[/dim]")
        return

    # Find target message
    target = None
    if arg.strip():
        # Pin by message ID
        msg_id = arg.strip()
        for msg in ctx.conv.messages:
            if msg.id == msg_id:
                target = msg
                break
        if not target:
            console.print(f"[red]Message not found:[/red] {msg_id}")
            return
    else:
        # Default: last assistant message
        for msg in reversed(ctx.conv.messages):
            if msg.role == Role.ASSISTANT:
                target = msg
                break
        if not target:
            console.print("[dim]No assistant response to pin.[/dim]")
            return

    target.pinned = not target.pinned
    action = "Pinned" if target.pinned else "Unpinned"
    preview = target.content[:60] + "..." if len(target.content) > 60 else target.content
    preview = preview.replace("\n", " ")
    console.print(f"[green]{action}:[/green] {preview}")
    console.print(f"[dim]Pinned messages stay in context even when older messages are dropped.[/dim]")


def _cmd_pins(ctx: SlashContext) -> None:
    """Show all pinned messages in the conversation."""
    pinned = [m for m in ctx.conv.messages if m.pinned]
    if not pinned:
        console.print("[dim]No pinned messages. Pin with: /pin[/dim]")
        return

    console.print(f"[bold]Pinned messages ({len(pinned)}):[/bold]")
    for msg in pinned:
        preview = msg.content[:80].replace("\n", " ")
        if len(msg.content) > 80:
            preview += "..."
        role_color = "cyan" if msg.role.value == "user" else "green"
        console.print(f"  [{role_color}]{msg.role.value}[/{role_color}] {preview}")
        console.print(f"    [dim]id: {msg.id}[/dim]")


def _cmd_report(ctx: SlashContext) -> None:
    """Generate a structured session summary from the conversation."""
    import re
    from collections import Counter
    from towel.agent.conversation import Role

    msgs = ctx.conv.messages
    if not msgs:
        console.print("[dim]No messages to summarize.[/dim]")
        return

    user_msgs = [m for m in msgs if m.role == Role.USER]
    asst_msgs = [m for m in msgs if m.role == Role.ASSISTANT]
    tool_msgs = [m for m in msgs if m.role == Role.TOOL]

    # ── Topics (from user messages) ──
    topics: list[str] = []
    for m in user_msgs:
        first_line = m.content.strip().split("\n")[0]
        # Strip @file refs and code blocks
        clean = re.sub(r"@[\w./~*?:-]+", "", first_line)
        clean = re.sub(r"```.*?```", "", clean, flags=re.DOTALL).strip()
        if clean and len(clean) > 5:
            topics.append(clean[:80])

    # ── Tools used ──
    tool_names: Counter[str] = Counter()
    for m in tool_msgs:
        if m.content.startswith("[") and "]" in m.content:
            name = m.content[1:m.content.index("]")]
            tool_names[name] += 1

    # ── Code blocks in responses ──
    code_count = 0
    languages: Counter[str] = Counter()
    for m in asst_msgs:
        blocks = re.findall(r"```(\w*)\n", m.content)
        code_count += len(blocks)
        for lang in blocks:
            if lang:
                languages[lang] += 1

    # ── Token stats ──
    total_tokens = sum(m.metadata.get("tokens", 0) for m in asst_msgs)
    tps_values = [m.metadata.get("tps", 0) for m in asst_msgs if m.metadata.get("tps")]
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0

    # ── Duration ──
    if len(msgs) >= 2:
        duration = msgs[-1].timestamp - msgs[0].timestamp
        mins = duration.total_seconds() / 60
        dur_str = f"{mins:.0f} min" if mins >= 1 else f"{duration.total_seconds():.0f} sec"
    else:
        dur_str = "—"

    # ── Pinned messages ──
    pinned = [m for m in msgs if m.pinned]

    # ── Render ──
    console.print(f"\n[bold]Session Report[/bold]")
    console.print(f"  [green]{ctx.conv.display_title}[/green]")
    if ctx.conv.tags:
        console.print(f"  Tags: {' '.join(f'[yellow]#{t}[/yellow]' for t in ctx.conv.tags)}")
    console.print(f"  Duration: {dur_str} · {len(user_msgs)} questions · {len(asst_msgs)} answers")
    if total_tokens:
        console.print(f"  Tokens: {total_tokens:,} generated ({avg_tps:.1f} tok/s avg)")

    if topics:
        console.print(f"\n[bold]Topics discussed ({len(topics)}):[/bold]")
        for t in topics[:8]:
            console.print(f"  [dim]•[/dim] {t}")
        if len(topics) > 8:
            console.print(f"  [dim]... and {len(topics) - 8} more[/dim]")

    if tool_names:
        console.print(f"\n[bold]Tools used ({sum(tool_names.values())} calls):[/bold]")
        for name, count in tool_names.most_common(8):
            console.print(f"  [yellow]{name}[/yellow] ×{count}")

    if code_count:
        lang_str = ", ".join(f"{l} ({c})" for l, c in languages.most_common(5)) if languages else "unspecified"
        console.print(f"\n[bold]Code:[/bold] {code_count} blocks — {lang_str}")

    if pinned:
        console.print(f"\n[bold]Pinned ({len(pinned)}):[/bold]")
        for m in pinned:
            preview = m.content[:60].replace("\n", " ") + ("..." if len(m.content) > 60 else "")
            console.print(f"  [dim]•[/dim] {preview}")

    console.print()


def _cmd_save(ctx: SlashContext, arg: str) -> None:
    """Save a code block from the last assistant response to a file."""
    import re
    from pathlib import Path
    from towel.agent.conversation import Role

    parts = arg.strip().split()
    if not parts:
        console.print("[red]Usage:[/red] /save <filename> [block_number]")
        console.print("  /save main.py       Save first code block to main.py")
        console.print("  /save util.py 2     Save second code block")
        return

    filename = parts[0]
    block_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0

    # Find last assistant message
    last_asst = None
    for msg in reversed(ctx.conv.messages):
        if msg.role == Role.ASSISTANT:
            last_asst = msg
            break

    if not last_asst:
        console.print("[dim]No assistant response to extract from.[/dim]")
        return

    blocks = re.findall(r"```\w*\n(.*?)```", last_asst.content, re.DOTALL)
    if not blocks:
        console.print("[dim]No code blocks found in last response.[/dim]")
        return

    if block_idx < 0 or block_idx >= len(blocks):
        console.print(f"[red]Block {block_idx + 1} not found.[/red] Response has {len(blocks)} code block(s).")
        if len(blocks) > 1:
            for i, b in enumerate(blocks):
                preview = b.strip().split("\n")[0][:60]
                console.print(f"  [dim]{i+1}. {preview}...[/dim]")
        return

    content = blocks[block_idx].strip()
    target = Path(filename)

    # Safety: don't overwrite without notice
    if target.exists():
        size = target.stat().st_size
        console.print(f"[yellow]Overwriting:[/yellow] {target} ({size} bytes)")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")
        lines = content.count("\n") + 1
        console.print(f"[green]Saved:[/green] {target} ({lines} lines, {len(content)} bytes)")
    except OSError as e:
        console.print(f"[red]Failed to save:[/red] {e}")


def _cmd_copy(ctx: SlashContext, arg: str) -> None:
    """Copy the last assistant response (or just its code blocks) to clipboard."""
    import platform
    import re
    import subprocess
    from towel.agent.conversation import Role

    # Find last assistant message
    last_asst = None
    for msg in reversed(ctx.conv.messages):
        if msg.role == Role.ASSISTANT:
            last_asst = msg
            break

    if not last_asst:
        console.print("[dim]No assistant response to copy.[/dim]")
        return

    content = last_asst.content
    mode = arg.strip().lower()

    if mode == "code":
        # Extract only code blocks
        blocks = re.findall(r"```\w*\n(.*?)```", content, re.DOTALL)
        if not blocks:
            console.print("[dim]No code blocks found in last response.[/dim]")
            return
        content = "\n\n".join(b.strip() for b in blocks)
        label = f"{len(blocks)} code block(s)"
    else:
        label = f"{len(content)} characters"

    # Copy to clipboard
    if platform.system() == "Darwin":
        cmd = ["pbcopy"]
    elif platform.system() == "Linux":
        cmd = ["xclip", "-selection", "clipboard"]
    else:
        console.print("[red]Clipboard not supported on this platform.[/red]")
        return

    try:
        proc = subprocess.run(cmd, input=content.encode("utf-8"), timeout=5, capture_output=True)
        if proc.returncode == 0:
            console.print(f"[green]Copied:[/green] {label}")
        else:
            console.print(f"[red]Clipboard error:[/red] {proc.stderr.decode()}")
    except FileNotFoundError:
        console.print(f"[red]Clipboard tool not found:[/red] {cmd[0]}")
    except Exception as e:
        console.print(f"[red]Failed to copy:[/red] {e}")


def _cmd_tag(ctx: SlashContext, arg: str) -> None:
    tag = arg.strip().lower()
    if not tag:
        console.print("[red]Usage:[/red] /tag <name>  or  /tag -<name> to remove")
        return

    if tag.startswith("-"):
        # Remove tag
        remove = tag[1:]
        if remove in ctx.conv.tags:
            ctx.conv.tags.remove(remove)
            ctx.store.save(ctx.conv)
            console.print(f"[green]Removed tag:[/green] {remove}")
        else:
            console.print(f"[dim]Tag not found:[/dim] {remove}")
    else:
        # Add tag
        if tag not in ctx.conv.tags:
            ctx.conv.tags.append(tag)
            ctx.store.save(ctx.conv)
            console.print(f"[green]Tagged:[/green] {tag}")
        else:
            console.print(f"[dim]Already tagged:[/dim] {tag}")


def _cmd_tags(ctx: SlashContext) -> None:
    if ctx.conv.tags:
        tag_str = ", ".join(f"[green]{t}[/green]" for t in ctx.conv.tags)
        console.print(f"  Tags: {tag_str}")
    else:
        console.print("[dim]No tags. Add with: /tag <name>[/dim]")


def _cmd_history(ctx: SlashContext, arg: str) -> None:
    """Show recent conversations inline — browse without leaving chat."""
    limit = int(arg.strip()) if arg.strip().isdigit() else 10
    convos = ctx.store.list_conversations(limit=limit)

    if not convos:
        console.print("[dim]No saved conversations.[/dim]")
        return

    console.print(f"[bold]Recent conversations ({len(convos)}):[/bold]\n")
    for c in convos:
        marker = " [green]<-- current[/green]" if c.id == ctx.conv.id else ""
        # Load tags
        tag_str = ""
        try:
            import json as _json
            data = _json.loads(ctx.store._path_for(c.id).read_text(encoding="utf-8"))
            tags = data.get("tags", [])
            if tags:
                tag_str = " " + " ".join(f"[yellow]#{t}[/yellow]" for t in tags)
        except Exception:
            pass

        console.print(
            f"  [green]{c.id}[/green]  "
            f"[dim]{c.created_at[:16]}[/dim]  "
            f"[dim]({c.message_count} msgs)[/dim]{tag_str}{marker}"
        )
        console.print(f"    {c.summary}")

    console.print(f"\n[dim]Switch with: /resume <id>[/dim]")


def _cmd_resume(ctx: SlashContext, arg: str) -> None:
    """Switch to a different saved conversation."""
    conv_id = arg.strip()
    if not conv_id:
        console.print("[red]Usage:[/red] /resume <conversation_id>")
        console.print("[dim]Browse with: /history[/dim]")
        return

    # Save current conversation first
    if ctx.conv.messages:
        ctx.store.save(ctx.conv)

    other = ctx.store.load(conv_id)
    if not other:
        console.print(f"[red]Conversation not found:[/red] {conv_id}")
        return

    old_id = ctx.conv.id

    # Swap the conversation in-place
    ctx.conv.id = other.id
    ctx.conv.title = other.title
    ctx.conv.tags = other.tags
    ctx.conv.messages = other.messages
    ctx.conv.created_at = other.created_at
    ctx.conv.channel = other.channel

    console.print(f"[green]Resumed:[/green] {other.display_title}")
    console.print(f"  {len(other)} messages, {other.channel}")
    if other.tags:
        tag_str = " ".join(f"[yellow]#{t}[/yellow]" for t in other.tags)
        console.print(f"  Tags: {tag_str}")
    console.print(f"[dim]Previous conversation saved as {old_id}[/dim]")


def _cmd_rename(ctx: SlashContext, arg: str) -> None:
    title = arg.strip()
    if not title:
        if ctx.conv.title:
            console.print(f"  Current title: [green]{ctx.conv.title}[/green]")
        else:
            console.print(f"  [dim]No title set (using: {ctx.conv.summary})[/dim]")
        console.print("[dim]Set with: /rename My Conversation Title[/dim]")
        return
    ctx.conv.title = title
    ctx.store.save(ctx.conv)
    console.print(f"[green]Renamed:[/green] {title}")


def _cmd_context(ctx: SlashContext) -> None:
    from towel.agent.project import find_project_contexts, load_project_context

    paths = find_project_contexts()
    if not paths:
        console.print("[dim]No .towel.md found in current directory or parents.[/dim]")
        console.print("[dim]Create one to give the agent project context:[/dim]")
        console.print()
        console.print("  [green]echo '# My Project' > .towel.md[/green]")
        return

    console.print(f"[bold]Project context files ({len(paths)}):[/bold]")
    for p in paths:
        size = p.stat().st_size
        console.print(f"  [green]{p}[/green] [dim]({size} bytes)[/dim]")

    block = load_project_context()
    if block:
        # Show a preview
        preview = block[:500]
        if len(block) > 500:
            preview += "\n..."
        console.print(f"\n[dim]{preview}[/dim]")


def _cmd_export(ctx: SlashContext, arg: str) -> None:
    from towel.persistence.export import export_markdown, export_html

    if len(ctx.conv) == 0:
        console.print("[dim]Nothing to export (conversation is empty).[/dim]")
        return

    filename = arg.strip() if arg.strip() else None

    # Auto-detect format from extension
    if filename and filename.endswith((".html", ".htm")):
        content = export_html(ctx.conv, include_metadata=True)
    else:
        content = export_markdown(ctx.conv, include_metadata=True)

    if filename:
        from pathlib import Path
        Path(filename).write_text(content, encoding="utf-8")
        console.print(f"[green]Exported to:[/green] {filename}")
    else:
        print(content)


def _cmd_snippet(ctx: SlashContext, arg: str) -> None:
    from towel.cli.snippets import set_snippet, remove_snippet

    parts = arg.split(None, 1)
    if len(parts) < 1:
        console.print("[red]Usage:[/red] /snippet <name> <text>")
        console.print("  /snippet header # Project Name\\nBy Kelsi Davis")
        console.print("  /snippet -<name>  to remove")
        return

    name = parts[0].lower()

    # Remove with -name
    if name.startswith("-") and len(parts) == 1:
        if remove_snippet(name[1:]):
            console.print(f"[green]Removed snippet:[/green] {name[1:]}")
        else:
            console.print(f"[dim]Snippet not found:[/dim] {name[1:]}")
        return

    if len(parts) < 2:
        console.print("[red]Usage:[/red] /snippet <name> <text>")
        return

    content = parts[1].replace("\\n", "\n")
    set_snippet(name, content)
    lines = content.count("\n") + 1
    console.print(f"[green]Saved snippet:[/green] {name} ({lines} line(s), {len(content)} chars)")


def _cmd_snippets(ctx: SlashContext) -> None:
    from towel.cli.snippets import list_snippets

    snippets = list_snippets()
    if not snippets:
        console.print("[dim]No snippets. Create one: /snippet <name> <text>[/dim]")
        return

    console.print(f"[bold]Snippets ({len(snippets)}):[/bold]")
    for name, content in sorted(snippets.items()):
        preview = content.replace("\n", "\\n")
        if len(preview) > 60:
            preview = preview[:57] + "..."
        console.print(f"  [green]{name}[/green]  {preview}")
    console.print(f"\n[dim]Use with: /s <name> [extra text][/dim]")


def _cmd_use_snippet(ctx: SlashContext, arg: str) -> bool | None:
    """Insert a snippet into the conversation as a user message."""
    from towel.cli.snippets import get_snippet
    from towel.agent.conversation import Role
    from towel.agent.refs import expand_refs, parse_refs

    parts = arg.split(None, 1)
    if not parts:
        console.print("[red]Usage:[/red] /s <snippet_name> [additional text]")
        return True

    name = parts[0].lower()
    extra = parts[1] if len(parts) > 1 else ""

    content = get_snippet(name)
    if content is None:
        console.print(f"[red]Snippet not found:[/red] {name}")
        console.print("[dim]List snippets: /snippets[/dim]")
        return True

    # Combine snippet + extra input
    full = content
    if extra:
        full = f"{content}\n\n{extra}"

    # Expand @file refs
    refs = parse_refs(full)
    if refs:
        full = expand_refs(full)

    console.print(f"[dim]  snippet: {name}[/dim]")
    ctx.conv.add(Role.USER, full)
    return False  # signal: run agent step


def _cmd_alias(ctx: SlashContext, arg: str) -> None:
    from towel.cli.aliases import set_alias

    parts = arg.split(None, 1)
    if len(parts) < 2:
        console.print("[red]Usage:[/red] /alias <name> <prompt>")
        console.print("  Example: /alias review Review this code for bugs and improvements")
        console.print("  Then use: /review @myfile.py")
        return

    name, prompt = parts[0].lower(), parts[1]
    set_alias(name, prompt)
    console.print(f"[green]Alias created:[/green] /{name}")
    console.print(f"  [dim]{prompt[:80]}{'...' if len(prompt) > 80 else ''}[/dim]")


def _cmd_aliases(ctx: SlashContext) -> None:
    from towel.cli.aliases import list_aliases

    aliases = list_aliases()
    if not aliases:
        console.print("[dim]No aliases defined. Create one with: /alias <name> <prompt>[/dim]")
        return

    console.print(f"[bold]Aliases ({len(aliases)}):[/bold]")
    for name, prompt in sorted(aliases.items()):
        preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
        console.print(f"  [green]/{name}[/green]  {preview}")


def _cmd_unalias(ctx: SlashContext, arg: str) -> None:
    from towel.cli.aliases import remove_alias

    name = arg.strip().lower()
    if not name:
        console.print("[red]Usage:[/red] /unalias <name>")
        return

    if remove_alias(name):
        console.print(f"[green]Removed alias:[/green] /{name}")
    else:
        console.print(f"[dim]Alias not found:[/dim] /{name}")


def _try_alias(ctx: SlashContext, cmd: str, arg: str) -> bool | None:
    """Try to expand a slash command as a user alias.

    Returns False to signal agent step, True if consumed, None if not an alias.
    """
    from towel.cli.aliases import get_alias
    from towel.agent.conversation import Role
    from towel.agent.refs import expand_refs, parse_refs

    # cmd is like "/review" — strip the slash
    alias_name = cmd.lstrip("/")
    prompt_template = get_alias(alias_name)
    if prompt_template is None:
        return None

    # Build the full prompt: alias template + user input
    if arg.strip():
        full_prompt = f"{prompt_template}\n\n{arg}"
    else:
        full_prompt = prompt_template

    # Expand @file references
    refs = parse_refs(full_prompt)
    if refs:
        full_prompt = expand_refs(full_prompt)

    console.print(f"[dim]  alias: {alias_name}[/dim]")
    ctx.conv.add(Role.USER, full_prompt)

    return False  # signal: run agent step


def _cmd_system(ctx: SlashContext, arg: str) -> None:
    if not arg:
        console.print(f"  [dim]{ctx.config.identity[:100]}...[/dim]" if len(ctx.config.identity) > 100 else f"  [dim]{ctx.config.identity}[/dim]")
        console.print("[dim]Override with: /system <new prompt>[/dim]")
        return
    ctx.config.identity = arg
    ctx.agent.config = ctx.config
    console.print(f"[green]System prompt updated.[/green]")
    console.print(f"  [dim]{arg[:100]}{'...' if len(arg) > 100 else ''}[/dim]")


# ── /loop — recurring prompt execution ──

_active_loops: dict[str, bool] = {}  # name -> running


def _cmd_loop(ctx: SlashContext, arg: str) -> None:
    """Run a prompt on a recurring interval."""
    import re
    import asyncio
    import threading

    parts = arg.strip().split(None, 1)
    if len(parts) < 2:
        console.print("[red]Usage:[/red] /loop <interval> <prompt>")
        console.print("  /loop 5m check git status")
        console.print("  /loop 30s monitor this endpoint")
        console.print("  /loop stop <name>   stop a running loop")
        console.print("  /loop list          show active loops")
        return

    # Handle subcommands
    if parts[0] == "list":
        if not _active_loops:
            console.print("[dim]No active loops.[/dim]")
        else:
            for name, running in _active_loops.items():
                status = "[green]running[/green]" if running else "[dim]stopped[/dim]"
                console.print(f"  {name}: {status}")
        return

    if parts[0] == "stop":
        name = parts[1].strip() if len(parts) > 1 else ""
        if name in _active_loops:
            _active_loops[name] = False
            console.print(f"[green]Stopping loop:[/green] {name}")
        else:
            console.print(f"[dim]No loop named:[/dim] {name}")
        return

    # Parse interval
    interval_str = parts[0]
    prompt = parts[1]

    m = re.match(r"^(\d+)(s|m|h)$", interval_str)
    if not m:
        console.print(f"[red]Invalid interval:[/red] {interval_str} (use 30s, 5m, 1h)")
        return

    value = int(m.group(1))
    unit = m.group(2)
    seconds = value * {"s": 1, "m": 60, "h": 3600}[unit]

    loop_name = f"loop-{len(_active_loops) + 1}"
    _active_loops[loop_name] = True

    console.print(f"[green]Started loop:[/green] {loop_name}")
    console.print(f"  Interval: {interval_str} ({seconds}s)")
    console.print(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    console.print(f"[dim]Stop with: /loop stop {loop_name}[/dim]")

    def _run_loop():
        import time
        while _active_loops.get(loop_name):
            time.sleep(seconds)
            if not _active_loops.get(loop_name):
                break
            # Add prompt as user message and signal the chat loop
            from towel.agent.conversation import Role
            ctx.conv.add(Role.USER, f"[{loop_name}] {prompt}")
            console.print(f"\n[yellow][{loop_name}][/yellow] {prompt}")

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()


def _cmd_health(ctx: SlashContext) -> None:
    """Show agent health status."""
    from towel.agent.heartbeat import Heartbeat

    # Check if heartbeat is attached to agent
    hb = getattr(ctx.agent, '_heartbeat', None)
    if not hb:
        console.print("[dim]No heartbeat monitor active.[/dim]")
        console.print("[dim]Heartbeat starts automatically with `towel serve`.[/dim]")
        return

    status = hb.status()
    icon = "[green]HEALTHY[/green]" if status.alive else "[red]UNHEALTHY[/red]"
    console.print(f"  Status: {icon}")
    console.print(f"  Uptime: {status.uptime_seconds:.0f}s")
    console.print(f"  Generations: {status.total_generations}")
    console.print(f"  Errors: {status.total_errors} ({status.consecutive_errors} consecutive)")
    console.print(f"  Model loaded: {'yes' if status.model_loaded else 'no'}")
    console.print(f"  Generating: {'yes' if status.is_generating else 'no'}")


def _cmd_delegate(ctx: SlashContext, arg: str) -> bool | None:
    """Delegate a task to a specialist agent role."""
    from towel.agent.conversation import Role

    parts = arg.strip().split(None, 1)
    if len(parts) < 2:
        console.print("[red]Usage:[/red] /delegate <role> <task>")
        console.print("  Roles: coder, researcher, reviewer, writer, architect, tester, debugger")
        console.print("  Example: /delegate reviewer check my auth code for security issues")
        return True

    role, task = parts[0].lower(), parts[1]

    from towel.agent.orchestrator import ROLE_PROMPTS
    if role not in ROLE_PROMPTS:
        console.print(f"[red]Unknown role:[/red] {role}")
        console.print(f"[dim]Available: {', '.join(sorted(ROLE_PROMPTS.keys()))}[/dim]")
        return True

    # Inject the specialist prompt + task as user message
    specialist_prompt = ROLE_PROMPTS[role]
    full_msg = f"[Acting as {role}]\n\nSystem context: {specialist_prompt}\n\nTask: {task}"

    console.print(f"[dim]  delegating to: {role}[/dim]")
    ctx.conv.add(Role.USER, full_msg)
    return False  # signal agent step


def _cmd_whoami(ctx: SlashContext) -> None:
    """Show agent identity, model, skills, and context budget."""
    from towel.agent.context import count_tokens_fallback

    config = ctx.config
    conv = ctx.conv

    # Count tokens in system prompt
    system = config.identity
    sys_tokens = count_tokens_fallback(system)

    # Count conversation tokens
    conv_chars = sum(len(m.content) for m in conv.messages)
    conv_tokens = count_tokens_fallback(str(conv_chars))

    # Skills
    skills = ctx.agent.skills
    skill_count = len(skills) if skills else 0
    tool_count = len(skills.tool_definitions()) if skills else 0

    console.print(f"[bold]Agent identity:[/bold]")
    console.print(f"  Model: [green]{config.model.name}[/green]")
    console.print(f"  Agent: {ctx.current_agent_name or 'default'}")
    console.print(f"  Identity: [dim]{config.identity[:80]}{'...' if len(config.identity) > 80 else ''}[/dim]")
    console.print(f"\n[bold]Context budget:[/bold]")
    console.print(f"  Window:     {config.model.context_window:,} tokens")
    console.print(f"  Max output: {config.model.max_tokens:,} tokens")
    console.print(f"  System:     ~{sys_tokens:,} tokens")
    console.print(f"  Messages:   {len(conv)} ({conv_chars:,} chars)")
    pinned = sum(1 for m in conv.messages if m.pinned)
    if pinned: console.print(f"  Pinned:     {pinned}")
    console.print(f"\n[bold]Capabilities:[/bold]")
    console.print(f"  Skills: {skill_count}  Tools: {tool_count}")
    if conv.tags:
        console.print(f"  Tags: {' '.join(f'#{t}' for t in conv.tags)}")
