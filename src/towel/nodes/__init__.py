"""Towel nodes — device capability providers.

A node represents a physical or virtual machine in the Towel LAN cluster.
It tracks hardware resources (VRAM, RAM, CPU), loaded models, and active
context windows so the controller can make informed scheduling decisions.
"""

from towel.nodes.capability import NodeCapability, NodeResources
from towel.nodes.tracker import NodeTracker

__all__ = ["NodeCapability", "NodeResources", "NodeTracker"]
