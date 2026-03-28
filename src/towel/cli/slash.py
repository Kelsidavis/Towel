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
  [green]/copy[/green] [code]           Copy last response to clipboard (or just code blocks)
  [green]/tag[/green] <name>            Add a tag to this conversation (or remove with -name)
  [green]/tags[/green]                 Show tags on this conversation
  [green]/rename[/green] <title>       Set a title for this conversation
  [green]/export[/green] [file]        Export conversation to markdown (.html for HTML)
  [green]/newagent[/green] <name> <model> <prompt>  Create new agent
  [green]/delagent[/green] <name>      Delete user agent
  [green]/context[/green]              Show loaded .towel.md project context
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

        case "/copy":
            _cmd_copy(ctx, arg)

        case "/tag":
            _cmd_tag(ctx, arg)

        case "/tags":
            _cmd_tags(ctx)

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

        case "/system":
            _cmd_system(ctx, arg)

        case _:
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


def _cmd_system(ctx: SlashContext, arg: str) -> None:
    if not arg:
        console.print(f"  [dim]{ctx.config.identity[:100]}...[/dim]" if len(ctx.config.identity) > 100 else f"  [dim]{ctx.config.identity}[/dim]")
        console.print("[dim]Override with: /system <new prompt>[/dim]")
        return
    ctx.config.identity = arg
    ctx.agent.config = ctx.config
    console.print(f"[green]System prompt updated.[/green]")
    console.print(f"  [dim]{arg[:100]}{'...' if len(arg) > 100 else ''}[/dim]")
