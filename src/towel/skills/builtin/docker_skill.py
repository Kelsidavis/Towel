"""Docker skill — manage containers, images, and inspect running services."""

from __future__ import annotations

import asyncio
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class DockerSkill(Skill):
    @property
    def name(self) -> str: return "docker"
    @property
    def description(self) -> str: return "Manage Docker containers, images, and services"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="docker_ps", description="List running containers (like docker ps)",
                parameters={"type":"object","properties":{
                    "all":{"type":"boolean","description":"Include stopped containers (default: false)"},
                }}),
            ToolDefinition(name="docker_images", description="List local Docker images",
                parameters={"type":"object","properties":{}}),
            ToolDefinition(name="docker_logs", description="Get logs from a container",
                parameters={"type":"object","properties":{
                    "container":{"type":"string","description":"Container name or ID"},
                    "tail":{"type":"integer","description":"Number of lines (default: 50)"},
                },"required":["container"]}),
            ToolDefinition(name="docker_inspect", description="Get detailed info about a container",
                parameters={"type":"object","properties":{
                    "container":{"type":"string","description":"Container name or ID"},
                },"required":["container"]}),
            ToolDefinition(name="docker_stats", description="Get resource usage stats for running containers",
                parameters={"type":"object","properties":{}}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "docker_ps": return await self._run_docker(["ps", "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"] + (["--all"] if arguments.get("all") else []))
            case "docker_images": return await self._run_docker(["images", "--format", "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"])
            case "docker_logs": return await self._run_docker(["logs", "--tail", str(arguments.get("tail",50)), arguments["container"]])
            case "docker_inspect":
                raw = await self._run_docker(["inspect", "--format",
                    "Name: {{.Name}}\nImage: {{.Config.Image}}\nStatus: {{.State.Status}}\n"
                    "Created: {{.Created}}\nPorts: {{range $p, $conf := .NetworkSettings.Ports}}{{$p}} {{end}}\n"
                    "Mounts: {{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}\n"
                    "Env: {{range .Config.Env}}{{.}}\n{{end}}",
                    arguments["container"]])
                return raw
            case "docker_stats": return await self._run_docker(["stats", "--no-stream", "--format", "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"])
            case _: return f"Unknown tool: {tool_name}"

    async def _run_docker(self, args: list[str]) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", *args,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            output = stdout.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                if "Cannot connect" in err or "not found" in err.lower():
                    return "Docker is not running or not installed."
                return f"Error: {err}"
            return output or "(no output)"
        except FileNotFoundError:
            return "Docker CLI not found. Install Docker Desktop."
        except TimeoutError:
            return "Docker command timed out."
        except Exception as e:
            return f"Error: {e}"
