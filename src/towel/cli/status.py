"""CLI command for Towel health status and diagnostics."""

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from towel.cli import cli
from towel.persistence.conversation import ConversationStore


def get_system_stats() -> dict[str, Any]:
    """Gather system-level metrics."""
    try:
        import psutil
    except ImportError:
        return {}
    
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "free_gb": round(memory.available / (1024**3), 2),
            "percent": memory.percent,
            "swap": {
                "total_gb": round(swap.total / (1024**3), 2),
                "used_gb": round(swap.used / (1024**3), 2),
                "percent": swap.percent,
            },
        },
        "cpu": {
            "percent": cpu,
            "cores": psutil.cpu_count(logical=True),
        },
        "disk": [
            {
                "mount": p.mount,
                "total_gb": round(p.total / (1024**3), 2),
                "used_gb": round(p.used / (1024**3), 2),
                "free_gb": round(p.free / (1024**3), 2),
                "percent": p.percent,
            }
            for p in psutil.disk_partitions(all=True)
        ],
        "load_avg": tuple(getattr(shutil, "getloadavg", lambda: (0, 0, 0))()),
    }


def get_python_process_stats() -> dict[str, Any]:
    """Get stats for the current Python process (and Towel agent if found)."""
    try:
        import psutil
    except ImportError:
        return {}
    
    process = psutil.Process()
    
    try:
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
    except Exception:
        memory_info = None
        memory_percent = None
    
    return {
        "pid": process.pid,
        "name": process.name(),
        "memory": {
            "rss_mb": round(memory_info.rss / (1024**2), 2) if memory_info else None,
            "vms_mb": round(memory_info.vms / (1024**2), 2) if memory_info else None,
            "percent": round(memory_percent, 1) if memory_percent else None,
        },
        "cpu_percent": process.cpu_percent(interval=0.1),
        "threads": process.num_threads(),
        "open_files": len(process.open_files()) if hasattr(process, "open_files") else 0,
    }


def get_model_stats() -> dict[str, Any]:
    """Estimate MLX model stats based on common patterns."""
    # Heuristics for common mlx-lm models
    model_stats = {
        "context_window": 8192,
        "quantization": "mlx/4bit",
        "memory_estimate_gb": 6.8,
        "kv_cache_mb_per_token": 2,
    }
    
    # Try to read actual model info if available
    try:
        from towel.agent.runtime import AgentRuntime
        runtime = AgentRuntime.instance()
        if runtime and hasattr(runtime, "model_name"):
            model_stats["model"] = runtime.model_name
    except ImportError:
        pass
    
    return model_stats


def get_tool_stats() -> dict[str, Any]:
    """Get stats on loaded tools/skills."""
    try:
        from towel.skills.registry import SkillRegistry
        registry = SkillRegistry.instance()
        
        skills = list(registry.skills.values())
        active = [s for s in skills if getattr(s, "enabled", True)]
        
        return {
            "total": len(skills),
            "active": len(active),
            "lazy": len([s for s in skills if getattr(s, "lazy", False)]),
            "memory_usage_gb": sum(
                getattr(s, "memory_estimate_gb", 0.5) for s in active
            ),
        }
    except ImportError:
        return {
            "total": 12,  # default for common skills
            "active": 8,
            "lazy": 4,
            "memory_usage_gb": 3.2,
        }


def get_session_stats() -> dict[str, Any]:
    """Gather session-specific stats."""
    store = ConversationStore()
    conversations = store.list()
    
    total_messages = sum(len(c.messages) for c in conversations)
    
    # Estimate total memory used by conversations
    total_memory_gb = sum(
        getattr(c, "memory_estimate_gb", 0.01 * len(c.messages))
        for c in conversations
    )
    
    return {
        "conversations": len(conversations),
        "messages": total_messages,
        "memory_gb": round(total_memory_gb, 2),
        "avg_messages_per_conv": round(total_messages / max(len(conversations), 1), 1),
    }


def calculate_health_score(stats: dict[str, Any]) -> int:
    """Calculate overall health score (0–100)."""
    score = 100
    
    # Memory health
    mem = stats.get("system", {}).get("memory", {})
    if mem.get("percent"):
        mem_score = max(0, 100 - int(mem["percent"] * 0.8))
        score += mem_score * 0.3
    
    # CPU health
    cpu = stats.get("system", {}).get("cpu", {})
    if cpu.get("percent"):
        cpu_score = max(0, 100 - int(cpu["percent"]))
        score += cpu_score * 0.2
    
    # Towel process health
    towel_proc = stats.get("towel_process", {})
    if towel_proc and towel_proc.get("memory", {}).get("percent"):
        proc_mem = towel_proc["memory"]["percent"]
        score += max(0, 100 - int(proc_mem * 0.7)) * 0.2
    
    # Tool health
    tools = stats.get("tools", {})
    if tools.get("active"):
        tool_score = min(100, int(100 * tools["active"] / tools["total"]))
        score += tool_score * 0.2
    
    # Session health (recent activity)
    session = stats.get("session", {})
    if session.get("messages"):
        session_score = min(100, int(100 * session["messages"] / 1000))
        score += session_score * 0.1
    
    return min(100, max(0, int(score)))


def format_health_indicator(score: int) -> tuple[str, str]:
    """Return (emoji, label) for health score."""
    if score >= 90:
        return "🟢", "Excellent"
    elif score >= 70:
        return "🟡", "Good"
    elif score >= 50:
        return "🟠", "Fair"
    else:
        return "🔴", "Needs Attention"


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed metrics")
@click.option("--history", is_flag=True, help="Compare with previous status runs")
def status(as_json: bool, verbose: bool, history: bool) -> None:
    """Display Towel health status and diagnostics."""
    console = Console()
    
    # Gather all stats
    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": get_system_stats(),
        "towel_process": get_python_process_stats(),
        "model": get_model_stats(),
        "tools": get_tool_stats(),
        "session": get_session_stats(),
    }
    
    # Add health score
    health_score = calculate_health_score(stats)
    health_emoji, health_label = format_health_indicator(health_score)
    stats["health_score"] = health_score
    stats["health_label"] = health_label
    
    # Save to history if requested
    if history:
        history_path = Path("~/.towel/status_history.json").expanduser()
        history_data = []
        if history_path.exists():
            with open(history_path) as f:
                history_data = json.load(f)
        
        # Append current status
        history_data.append(stats)
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
    
    # Output
    if as_json:
        console.print_json(json.dumps(stats, indent=2, default=str))
    else:
        # Build rich UI
        panel = Panel(
            build_status_tree(stats, verbose=verbose),
            title=f"[bold]🧼 Towel v0.7.3[/bold] • [bold]{health_label}[/bold] ([bold]{health_score}/100[/bold])",
            subtitle=f"Last updated: {stats['timestamp'][:16]}",
            padding=(1, 2),
            border_style="dim",
        )
        console.print(panel)


def build_status_tree(stats: dict[str, Any], verbose: bool = False) -> Tree:
    """Build a rich.tree.Tree for display."""
    tree = Tree("🧼 Towel Status")
    
    # System Overview
    system = tree.add("⚙️ System Overview")
    sys_info = stats.get("system", {})
    system.add(f"OS: Darwin {sys_info.get('timestamp', '')[:10]}")
    system.add(f"Load: {' '.join(str(l) for l in sys_info.get('load_avg', (0, 0, 0)))}")
    
    mem = sys_info.get("memory", {})
    memory_text = f"{mem.get('used_gb', 0):.1f}/{mem.get('total_gb', 0):.1f} GB ({mem.get('percent', 0)}%)"
    system.add(Text(f"Memory: {memory_text}", style="green" if mem.get('percent', 0) < 80 else "orange"))
    
    if verbose:
        swap = mem.get("swap", {})
        if swap:
            system.add(
                f"Swap: {swap.get('used_gb', 0):.1f}/{swap.get('total_gb', 0):.1f} GB ({swap.get('percent', 0)}%)"
            )
    
    # Towel Runtime
    towel = tree.add("⚙️ Towel Runtime")
    towel_proc = stats.get("towel_process", {})
    mem = towel_proc.get("memory", {})
    
    towel.add(
        Text(
            f"Process: PID {towel_proc.get('pid', 'N/A')} ({towel_proc.get('name', 'python')})",
            style="cyan",
        )
    )
    towel.add(
        Text(
            f"Memory: {mem.get('rss_mb', 'N/A')} MB RSS ({mem.get('percent', 'N/A')}%)",
            style="cyan",
        )
    )
    
    model = stats.get("model", {})
    towel.add(
        Text(
            f"Model: {model.get('model', 'Llama-3-8b')} ({model.get('quantization', 'mlx')})",
            style="blue",
        )
    )
    towel.add(f"Context: {model.get('context_window', 8192)} tokens")
    
    # Tools
    tools = stats.get("tools", {})
    tools_text = f"{tools.get('active', 0)}/{tools.get('total', 0)} active"
    if tools.get("lazy"):
        tools_text += f" ({tools['lazy']} lazy)"
    tools.add(Text(f"Tools: {tools_text}", style="yellow"))
    tools.add(
        Text(
            f"Tool Memory: {tools.get('memory_usage_gb', 0):.1f} GB",
            style="yellow",
        )
    )
    
    # Session
    session = stats.get("session", {})
    session.add(
        Text(
            f"Conversations: {session.get('conversations', 0)}",
            style="green",
        )
    )
    session.add(
        Text(
            f"Messages: {session.get('messages', 0)} ({session.get('avg_messages_per_conv', 0):.1f}/conv)",
            style="green",
        )
    )
    session.add(
        Text(
            f"Memory: {session.get('memory_gb', 0):.1f} GB",
            style="green",
        )
    )
    
    # Health Score
    health = tree.add(f"🌟 Health Score: {stats.get('health_score', 0)}/100")
    health.add(Text(f"Label: {stats.get('health_label', 'Unknown')}", style="green"))
    
    if verbose:
        # Detailed breakdown
        detailed = health.add("Detailed Breakdown")
        detailed.add(f"System Memory: {mem.get('percent', 0)}% → contrib: ~30")
        detailed.add(f"CPU: {towel_proc.get('cpu_percent', 0)}% → contrib: ~20")
        detailed.add(f"Towel Process Memory: {mem.get('percent', 0)}% → contrib: ~20")
        detailed.add(f"Tools Active: {tools.get('active', 0)} → contrib: ~20")
        detailed.add(f"Session Messages: {session.get('messages', 0)} → contrib: ~10")
    
    return tree


if __name__ == "__main__":
    status()
