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
  [green]/clear[/green]                Clear conversation history
  [green]/agent[/green] [name]         Switch agent / show current
  [green]/agents[/green]               List available agents
  [green]/memory[/green]               Show all memories
  [green]/remember[/green] <key> <val> Save a memory
  [green]/forget[/green] <key>         Remove a memory
  [green]/t[/green] <template> <input>  Apply a prompt template (e.g., /t review code here)
  [green]/templates[/green]            List available templates
  [green]/rename[/green] <title>       Set a title for this conversation
  [green]/export[/green] [file]        Export conversation to markdown
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
