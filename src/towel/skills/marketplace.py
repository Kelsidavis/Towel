"""Skill marketplace — discover, install, and manage community skills.

Skills are Python files hosted on GitHub (or any URL) that implement
the Skill base class. The marketplace provides a curated registry
and one-command installation.

Usage:
    towel marketplace                  browse available skills
    towel marketplace search web       search for skills
    towel marketplace install <name>   install a skill
    towel marketplace remove <name>    uninstall a skill
"""

from __future__ import annotations

import logging

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.skills.marketplace")

SKILLS_DIR = TOWEL_HOME / "skills"
REGISTRY_FILE = TOWEL_HOME / "marketplace_cache.json"

# Built-in registry of community skills (can be extended via remote fetch)
COMMUNITY_SKILLS: list[dict[str, str]] = [
    {"name": "weather", "description": "Get weather forecasts by city", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/weather_skill.py", "author": "towel-ai", "tags": "weather,forecast,api"},
    {"name": "wikipedia", "description": "Search and summarize Wikipedia articles", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/wikipedia_skill.py", "author": "towel-ai", "tags": "wiki,search,knowledge"},
    {"name": "hackernews", "description": "Browse Hacker News top stories and comments", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/hackernews_skill.py", "author": "towel-ai", "tags": "news,tech,hn"},
    {"name": "reddit", "description": "Browse Reddit posts and comments from any subreddit", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/reddit_skill.py", "author": "towel-ai", "tags": "social,reddit,news"},
    {"name": "latex", "description": "Render LaTeX math expressions to Unicode", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/latex_skill.py", "author": "towel-ai", "tags": "math,latex,formatting"},
    {"name": "figlet", "description": "Generate ASCII art text banners", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/figlet_skill.py", "author": "towel-ai", "tags": "ascii,art,text"},
    {"name": "ip-geo", "description": "IP geolocation and network info lookup", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/ipgeo_skill.py", "author": "towel-ai", "tags": "network,ip,geo"},
    {"name": "currency", "description": "Real-time currency exchange rates", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/currency_skill.py", "author": "towel-ai", "tags": "finance,currency,exchange"},
    {"name": "pomodoro", "description": "Pomodoro timer for focus sessions", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/pomodoro_skill.py", "author": "towel-ai", "tags": "productivity,timer,focus"},
    {"name": "changelog-gen", "description": "Generate changelogs from git history", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/changelog_skill.py", "author": "towel-ai", "tags": "git,changelog,release"},
    {"name": "openapi", "description": "Parse and explore OpenAPI/Swagger specs", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/openapi_skill.py", "author": "towel-ai", "tags": "api,openapi,swagger"},
    {"name": "csv-analyzer", "description": "Advanced CSV analysis with pivot tables and charts", "url": "https://raw.githubusercontent.com/towel-ai/skills/main/csv_analyzer_skill.py", "author": "towel-ai", "tags": "data,csv,analysis"},
]


def search_marketplace(query: str) -> list[dict[str, str]]:
    q = query.lower()
    return [s for s in COMMUNITY_SKILLS
            if q in s["name"].lower() or q in s["description"].lower() or q in s.get("tags", "").lower()]


def list_installed() -> list[str]:
    if not SKILLS_DIR.exists():
        return []
    return [f.stem.replace("_skill", "") for f in SKILLS_DIR.glob("*_skill.py")]


async def install_skill(name: str) -> str:
    """Download and install a community skill."""
    skill = next((s for s in COMMUNITY_SKILLS if s["name"] == name), None)
    if not skill:
        return f"Skill not found: {name}"

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{name.replace('-', '_')}_skill.py"
    target = SKILLS_DIR / filename

    if target.exists():
        return f"Already installed: {name} ({target})"

    try:
        import httpx
        resp = httpx.get(skill["url"], timeout=10, follow_redirects=True)
        if resp.status_code != 200:
            return f"Download failed: HTTP {resp.status_code}"
        target.write_text(resp.text, encoding="utf-8")
        return f"Installed: {name} -> {target}\nRestart Towel to load the skill."
    except ImportError:
        return "httpx not installed — cannot download skills"
    except Exception as e:
        return f"Install failed: {e}"


def remove_skill(name: str) -> str:
    """Remove an installed community skill."""
    filename = f"{name.replace('-', '_')}_skill.py"
    target = SKILLS_DIR / filename
    if target.exists():
        target.unlink()
        return f"Removed: {name}"
    return f"Not installed: {name}"
