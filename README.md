# Towel

[![CI](https://github.com/Kelsidavis/Towl/actions/workflows/ci.yml/badge.svg)](https://github.com/Kelsidavis/Towl/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX-black?logo=apple)](https://ml-explore.github.io/mlx/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/tests-544%20passing-brightgreen)]()
[![Skills](https://img.shields.io/badge/skills-15%20built--in-blue)]()

**Don't Panic.**

A local AI assistant powered by MLX on Apple Silicon. Private, fast, extensible — runs entirely on your Mac with no cloud dependency.

## Quick Start

```bash
pip install -e ".[all]"
towel init
towel chat
```

## What Can Towel Do?

### Chat

```bash
towel chat                  # interactive chat with streaming
towel ask "explain monads"  # one-shot query (pipeable)
cat code.py | towel ask "review this"
```

### Developer Tools

```bash
towel review                # AI code review of git changes
towel review --staged       # review only staged changes
towel review -f security    # focus on security issues
towel commit                # generate commit message + commit
towel commit -a             # stage all and commit
towel watch src/*.py        # live feedback on file changes
```

### Conversation Management

```bash
towel history               # list conversations
towel history --tag work    # filter by tag
towel log                   # activity timeline
towel log --today           # today's sessions
towel search "auth bug"     # search across all conversations
towel show <id>             # view a conversation
towel export <id> -f html   # export (markdown/text/json/html)
towel import backup.json    # import conversations
towel gc                    # clean up old conversations
```

### In-Chat Commands

40+ slash commands for power users:

| Command | Description |
|---------|-------------|
| `/undo` | Remove last exchange |
| `/retry` | Regenerate last response |
| `/fork [title]` | Branch the conversation |
| `/diff <id>` | Compare with another conversation |
| `/compact` | Compress old messages to free context |
| `/pin` / `/pins` | Pin important messages (survive context eviction) |
| `/grep <query>` | Search within conversation |
| `/save file.py` | Extract code blocks to files |
| `/copy` / `/copy code` | Copy response to clipboard |
| `/tag` / `/tags` | Organize with tags |
| `/stats` | Session statistics + cloud cost comparison |
| `/report` | Full session summary |
| `/history` / `/resume` | Switch conversations inline |
| `/alias` / `/snippet` | Custom shortcuts and text blocks |
| `/t review @file` | Apply prompt templates |

### 15 Built-in Skills

The agent has tools for:

| Skill | Tools |
|-------|-------|
| **filesystem** | read, write, list files |
| **shell** | execute commands |
| **git** | status, diff, log, commit, branch |
| **web** | fetch URLs |
| **search** | grep-like recursive search |
| **data** | JSON/CSV parsing, math |
| **memory** | persistent cross-session memory |
| **clipboard** | read/write system clipboard |
| **system** | CPU, memory, disk, processes |
| **time** | current time, timezones, duration calc |
| **network** | DNS lookup, port check, HTTP ping, whois |
| **hash** | MD5/SHA/base64/URL encoding |
| **env** | environment variables, PATH, which |
| **regex** | test, match, replace, split |

### Web UI

```bash
towel serve    # starts gateway + web UI
# Open http://127.0.0.1:18743
```

- 4 themes (Deep Space, Frost, Matrix, Solarized)
- Command palette (Ctrl+P)
- Keyboard shortcuts (Ctrl+N/K/L/E/T)
- Conversation sidebar with search
- Real-time streaming

### API

```bash
# Simple ask endpoint
curl -X POST http://127.0.0.1:18743/api/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'

# OpenAI-compatible endpoint
curl http://127.0.0.1:18743/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}'
```

### Extensible

```bash
towel skill-init my_tool    # generate a skill skeleton
# Edit ~/.towel/skills/my_tool_skill.py
# Restart towel — skill auto-loaded
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                  Gateway                      │
│     WebSocket + HTTP + OpenAI-compat API     │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Sessions  │  │ Routing  │  │ 15 Skills │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└──────────┬───────────────────────┬───────────┘
           │                       │
    ┌──────┴──────┐         ┌──────┴──────┐
    │   Agent     │         │  Channels   │
    │  Runtime    │         │             │
    │  (MLX)      │         │  CLI        │
    │             │         │  WebChat    │
    │  Streaming  │         │  HTTP API   │
    │  Tool loop  │         │  ...        │
    └─────────────┘         └─────────────┘
```

## Configuration

```bash
towel config        # show current settings
towel config --json # machine-readable
towel doctor        # diagnose your setup
towel bench         # benchmark model speed
```

Config lives in `~/.towel/config.toml`. Three built-in agent profiles: **coder**, **researcher**, **writer**.

## License

MIT

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
