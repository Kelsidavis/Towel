"""TUI — Terminal User Interface for Towel.

A rich interactive dashboard that shows conversations, skills,
system status, and lets you chat — all in one terminal screen.
"""

from __future__ import annotations

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from towel import __version__
from towel.config import TOWEL_HOME, TowelConfig


def build_header() -> Panel:
    return Panel(
        f"[bold green]Towel v{__version__}[/bold green] — Don't Panic.  "
        f"[dim]100 skills · 263 tools · 43 commands[/dim]",
        style="green",
    )


def build_skills_panel(config: TowelConfig) -> Panel:
    from towel.skills.builtin import register_builtins
    from towel.skills.registry import SkillRegistry

    reg = SkillRegistry()
    register_builtins(reg)
    skills = sorted(reg.list_skills())

    table = Table(show_header=False, box=None, padding=(0, 1))
    # 4 columns
    cols = 4
    rows = (len(skills) + cols - 1) // cols
    for r in range(min(rows, 15)):
        row = []
        for c in range(cols):
            idx = r + c * rows
            if idx < len(skills):
                row.append(f"[green]{skills[idx]}[/green]")
            else:
                row.append("")
        table.add_row(*row)

    if len(skills) > 60:
        table.add_row(f"[dim]... and {len(skills) - 60} more[/dim]", "", "", "")

    return Panel(table, title=f"Skills ({len(skills)})", border_style="cyan")


def build_conversations_panel() -> Panel:
    from towel.persistence.store import ConversationStore

    store = ConversationStore()
    convos = store.list_conversations(limit=8)

    if not convos:
        return Panel("[dim]No conversations yet.[/dim]", title="Recent", border_style="blue")

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("ID", style="green", width=14)
    table.add_column("Title", width=30)
    table.add_column("Msgs", justify="right", width=5)

    for c in convos:
        table.add_row(c.id[:12], c.summary[:28], str(c.message_count))

    return Panel(table, title=f"Conversations ({len(convos)})", border_style="blue")


def build_system_panel(config: TowelConfig) -> Panel:
    from towel.memory.store import MemoryStore

    mem = MemoryStore()
    conv_dir = TOWEL_HOME / "conversations"
    conv_count = len(list(conv_dir.glob("*.json"))) if conv_dir.exists() else 0

    lines = [
        f"  Model:    [green]{config.model.name}[/green]",
        f"  Context:  {config.model.context_window:,} tokens",
        f"  Gateway:  ws://:{config.gateway.port} + http://:{config.gateway.port + 1}",
        f"  Memories: {mem.count}",
        f"  Saved:    {conv_count} conversations",
        f"  Home:     {TOWEL_HOME}",
    ]
    return Panel("\n".join(lines), title="System", border_style="yellow")


def build_channels_panel() -> Panel:
    channels = [
        ("CLI", "towel chat", "green"),
        ("Web", "towel serve", "green"),
        ("Discord", "towel discord -t TOKEN", "blue"),
        ("Telegram", "towel telegram -t TOKEN", "blue"),
        ("Slack", "towel slack -b/-a TOKEN", "blue"),
        ("Webhook", "towel webhook", "cyan"),
    ]
    table = Table(show_header=False, box=None, padding=(0, 1))
    for name, cmd, color in channels:
        table.add_row(f"[{color}]{name}[/{color}]", f"[dim]{cmd}[/dim]")
    return Panel(table, title="Channels (6)", border_style="magenta")


def build_commands_panel() -> Panel:
    cmds = [
        ("chat", "Interactive chat"),
        ("ask", "One-shot query"),
        ("review", "Code review"),
        ("commit", "AI commit msg"),
        ("fix", "Debug errors"),
        ("explain", "Explain code"),
        ("summarize", "Summarize text"),
        ("watch", "Live feedback"),
        ("dashboard", "System overview"),
        ("deploy", "Generate deploy files"),
    ]
    table = Table(show_header=False, box=None, padding=(0, 1))
    for cmd, desc in cmds:
        table.add_row(f"[green]towel {cmd}[/green]", f"[dim]{desc}[/dim]")
    return Panel(table, title="Key Commands", border_style="white")


def render_tui(config: TowelConfig | None = None) -> Layout:
    config = config or TowelConfig.load()

    layout = Layout()
    layout.split_column(
        Layout(build_header(), size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right", ratio=1),
    )
    layout["left"].split_column(
        Layout(build_skills_panel(config), ratio=2),
        Layout(build_channels_panel(), ratio=1),
    )
    layout["right"].split_column(
        Layout(build_system_panel(config), size=9),
        Layout(build_conversations_panel(), ratio=1),
        Layout(build_commands_panel(), ratio=1),
    )
    return layout
