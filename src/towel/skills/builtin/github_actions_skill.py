"""GitHub Actions skill — check workflow runs and status."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class GithubActionsSkill(Skill):
    @property
    def name(self) -> str:
        return "gh_actions"

    @property
    def description(self) -> str:
        return "Check GitHub Actions workflow runs and status"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="gh_runs",
                description="List recent workflow runs for a repo",
                parameters={
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo"},
                        "limit": {"type": "integer", "description": "Max runs (default: 5)"},
                    },
                    "required": ["repo"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "gh_runs":
            return f"Unknown: {tool_name}"
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=10,
                headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "Towel"},
            ) as c:
                resp = await c.get(
                    f"https://api.github.com/repos/{arguments['repo']}/actions/runs",
                    params={"per_page": arguments.get("limit", 5)},
                )
                runs = resp.json().get("workflow_runs", [])
                if not runs:
                    return "No workflow runs."
                lines = [f"Recent runs for {arguments['repo']}:"]
                for r in runs:
                    icon = {
                        "success": "✓",
                        "failure": "✗",
                        "cancelled": "○",
                        "in_progress": "~",
                    }.get(r.get("conclusion", ""), "?")
                    lines.append(
                        f"  [{icon}] {r['name']} #{r['run_number']}"
                        f" — {r.get('conclusion', 'running')}"
                        f" ({r['created_at'][:10]})"
                    )
                return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"
