"""Tool-gating policy — the enforcement point for dangerous capabilities.

The audit log records what a tool *did*; this decides whether it may run
*at all*. It is the durable answer to "a rogue model has shell, exfil,
and persistence tools wide open" — a single policy, evaluated at the
registry chokepoint before every tool executes.

Modes (env ``TOWEL_TOOL_POLICY``):

* ``audit`` (default) — allow everything; rely on the audit log. Chosen
  as the default so dropping this in does not change a running system's
  behavior. Flip to ``enforce`` deliberately.
* ``enforce`` — refuse any tool whose risk tier is in the blocked set
  (env ``TOWEL_BLOCKED_RISKS``, default ``exec,exfil,secret,persist,
  lateral``), unless the specific tool is named in the allowlist.

Overrides (comma-separated tool names):

* ``TOWEL_ALLOW_TOOLS`` — always allow these, even under enforce.
* ``TOWEL_DENY_TOOLS`` — always refuse these, in any mode.

Risk tiers come from ``towel.audit.risk_tag`` so the gating model and the
audit model never drift apart. A refusal is returned as a string the
caller surfaces to the model (and the audit log tags ``blocked``); it is
never a silent drop.
"""

from __future__ import annotations

import logging
import os

from towel.audit import risk_tag

log = logging.getLogger("towel.policy")

# Tiers blocked by default once enforcement is on. 'memory' and 'low' are
# intentionally NOT here — memory writes are already guarded by content,
# and low-risk tools are the bulk of normal operation.
_DEFAULT_BLOCKED = "exec,exfil,secret,persist,lateral"


def _split(env_value: str | None) -> set[str]:
    if not env_value:
        return set()
    return {item.strip() for item in env_value.split(",") if item.strip()}


class ToolPolicy:
    """Evaluates whether a tool call is permitted under the current policy."""

    def __init__(
        self,
        mode: str = "audit",
        blocked_risks: set[str] | None = None,
        allow_tools: set[str] | None = None,
        deny_tools: set[str] | None = None,
    ) -> None:
        self.mode = mode
        self.blocked_risks = (
            blocked_risks if blocked_risks is not None else _split(_DEFAULT_BLOCKED)
        )
        self.allow_tools = allow_tools or set()
        self.deny_tools = deny_tools or set()

    @classmethod
    def from_env(cls, config: object | None = None) -> ToolPolicy:
        """Build a policy from saved config, with env vars taking priority.

        ``config`` is an optional ``SecurityConfig``-shaped object (has
        ``tool_policy``/``blocked_risks``/``allow_tools``/``deny_tools``).
        Settings-menu values live there; environment variables override
        them so an operator can lock down a deployment without editing
        config. Env unset → fall back to config → fall back to defaults.
        """
        sec = config
        cfg_mode = getattr(sec, "tool_policy", None) if sec else None
        cfg_blocked = getattr(sec, "blocked_risks", None) if sec else None
        cfg_allow = getattr(sec, "allow_tools", None) if sec else None
        cfg_deny = getattr(sec, "deny_tools", None) if sec else None

        env_mode = os.environ.get("TOWEL_TOOL_POLICY")
        env_blocked = os.environ.get("TOWEL_BLOCKED_RISKS")
        env_allow = os.environ.get("TOWEL_ALLOW_TOOLS")
        env_deny = os.environ.get("TOWEL_DENY_TOOLS")

        return cls(
            mode=(env_mode or cfg_mode or "audit").strip().lower(),
            blocked_risks=(
                _split(env_blocked) if env_blocked
                else (set(cfg_blocked) if cfg_blocked is not None else _split(_DEFAULT_BLOCKED))
            ),
            allow_tools=(
                _split(env_allow) if env_allow is not None
                else set(cfg_allow or [])
            ),
            deny_tools=(
                _split(env_deny) if env_deny is not None
                else set(cfg_deny or [])
            ),
        )

    def evaluate(self, tool_name: str) -> str | None:
        """Return a refusal reason if the tool is blocked, else None."""
        # Explicit deny wins in every mode.
        if tool_name in self.deny_tools:
            return (
                f"refused: tool {tool_name!r} is denied by policy "
                "(TOWEL_DENY_TOOLS)."
            )
        if self.mode != "enforce":
            return None
        if tool_name in self.allow_tools:
            return None
        tier = risk_tag(tool_name)
        if tier in self.blocked_risks:
            return (
                f"refused: tool {tool_name!r} (risk tier {tier!r}) is blocked "
                "by the active tool policy. Add it to TOWEL_ALLOW_TOOLS to "
                "permit, or relax TOWEL_BLOCKED_RISKS."
            )
        return None


_policy: ToolPolicy | None = None


def get_policy() -> ToolPolicy:
    """Return the process-wide policy.

    Built once on first use from the saved security config (settings
    menu → ``~/.towel/config.toml``), with environment variables taking
    priority. Falls back to env-only defaults if config can't be read.
    """
    global _policy
    if _policy is None:
        security = None
        try:
            from towel.config import TowelConfig

            security = TowelConfig.load().security
        except Exception as exc:  # config optional — never block startup
            log.debug("policy: could not load config, using env/defaults (%s)", exc)
        _policy = ToolPolicy.from_env(security)
        if _policy.mode == "enforce":
            log.warning(
                "Tool policy ENFORCE active: blocking risk tiers %s",
                sorted(_policy.blocked_risks),
            )
    return _policy


def set_policy(policy: ToolPolicy) -> None:
    """Override the process-wide policy (tests / programmatic config)."""
    global _policy
    _policy = policy
