"""IP calculator skill — subnet math, CIDR notation, range calculation."""

from __future__ import annotations

import ipaddress
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class IpCalcSkill(Skill):
    @property
    def name(self) -> str: return "ipcalc"
    @property
    def description(self) -> str: return "IP subnet calculator — CIDR, ranges, netmasks"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="ipcalc_info", description="Get detailed info about an IP address or CIDR subnet",
                parameters={"type":"object","properties":{
                    "address":{"type":"string","description":"IP or CIDR (e.g., 192.168.1.0/24, 10.0.0.1)"},
                },"required":["address"]}),
            ToolDefinition(name="ipcalc_contains", description="Check if an IP is within a subnet",
                parameters={"type":"object","properties":{
                    "subnet":{"type":"string","description":"Subnet in CIDR (e.g., 10.0.0.0/8)"},
                    "ip":{"type":"string","description":"IP to check"},
                },"required":["subnet","ip"]}),
            ToolDefinition(name="ipcalc_split", description="Split a subnet into smaller subnets",
                parameters={"type":"object","properties":{
                    "subnet":{"type":"string","description":"Subnet to split (e.g., 10.0.0.0/24)"},
                    "new_prefix":{"type":"integer","description":"New prefix length (e.g., 26 to split /24 into /26s)"},
                },"required":["subnet","new_prefix"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "ipcalc_info": return self._info(arguments["address"])
            case "ipcalc_contains": return self._contains(arguments["subnet"], arguments["ip"])
            case "ipcalc_split": return self._split(arguments["subnet"], arguments["new_prefix"])
            case _: return f"Unknown tool: {tool_name}"

    def _info(self, addr: str) -> str:
        try:
            if "/" in addr:
                net = ipaddress.ip_network(addr, strict=False)
                return (f"Network: {net}\n"
                        f"  Netmask: {net.netmask}\n"
                        f"  Wildcard: {net.hostmask}\n"
                        f"  Broadcast: {net.broadcast_address}\n"
                        f"  First host: {net.network_address + 1}\n"
                        f"  Last host: {net.broadcast_address - 1}\n"
                        f"  Hosts: {net.num_addresses - 2}\n"
                        f"  Private: {net.is_private}")
            else:
                ip = ipaddress.ip_address(addr)
                return (f"Address: {ip}\n"
                        f"  Version: IPv{ip.version}\n"
                        f"  Private: {ip.is_private}\n"
                        f"  Loopback: {ip.is_loopback}\n"
                        f"  Link-local: {ip.is_link_local}\n"
                        f"  Multicast: {ip.is_multicast}\n"
                        f"  Integer: {int(ip)}\n"
                        f"  Binary: {ip.packed.hex()}")
        except ValueError as e:
            return f"Invalid address: {e}"

    def _contains(self, subnet: str, ip: str) -> str:
        try:
            net = ipaddress.ip_network(subnet, strict=False)
            addr = ipaddress.ip_address(ip)
            if addr in net:
                return f"{ip} IS within {subnet}"
            return f"{ip} is NOT within {subnet}"
        except ValueError as e:
            return f"Error: {e}"

    def _split(self, subnet: str, new_prefix: int) -> str:
        try:
            net = ipaddress.ip_network(subnet, strict=False)
            subnets = list(net.subnets(new_prefix=new_prefix))
            lines = [f"Split {subnet} into /{new_prefix} ({len(subnets)} subnets):"]
            for s in subnets[:32]:
                lines.append(f"  {s} ({s.num_addresses - 2} hosts)")
            if len(subnets) > 32:
                lines.append(f"  ... and {len(subnets) - 32} more")
            return "\n".join(lines)
        except ValueError as e:
            return f"Error: {e}"
