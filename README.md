# Towel

[![CI](https://github.com/Kelsidavis/Towel/actions/workflows/ci.yml/badge.svg)](https://github.com/Kelsidavis/Towel/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX-black?logo=apple)](https://ml-explore.github.io/mlx/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/tests-1350%2B%20passing-brightgreen)]()
[![Skills](https://img.shields.io/badge/skills-100%2B%20built--in-blue)]()

**Don't Panic.**

> **Towel** — **T**ool **O**riented **W**orker **E**xecution **L**ink

A local AI assistant powered by MLX. Private, fast, extensible, and local-first.
Towel is best on Apple Silicon, and also supports Linux.

## Quick Start

```bash
pip install -e ".[all]"
towel setup    # browser GUI — pick backend (MLX / Ollama / llama-server / Claude) + model
towel chat     # start chatting
```

Skipping setup? `towel init` writes a starter `~/.towel/config.toml` you can edit by hand.
Or run `towel doctor` any time to verify the environment.

Launch scripts:

```bash
./launch.sh       # Linux (also works on macOS)
./launch.command  # macOS double-click friendly
```

## What Can Towel Do?

### Chat

```bash
towel chat                  # interactive chat with streaming
towel ask "explain monads"  # one-shot query (pipeable)
cat code.py | towel ask "review this"
```

### Developer CLI

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

### 60 Built-in Skills

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
| **convert** | length, weight, volume, temperature, speed, data, time |
| **json_tools** | diff, flatten, schema generation, validate |
| **diff** | compare files and text, similarity stats |
| **archive** | create/list/extract zip and tar archives |
| **cron** | explain, preview, and build cron expressions |
| **markdown** | tables, TOC, checklists, JSON-to-markdown |
| **http** | full HTTP requests (GET/POST/PUT/DELETE, headers, JSON) |
| **sql** | query SQLite, inspect schema, explain plans |
| **image** | dimensions, format, file size (PNG/JPEG/GIF) |
| **process** | find, inspect, tree, listening ports |
| **text** | word count, stats, transforms, frequency |
| **knowledge** | personal knowledge base with tags |
| **translate** | language detection, translation prompts |
| **security** | scan for secrets, permissions, dependency audit |
| **todo** | task management with priorities and due dates |
| **scaffold** | project boilerplate (8 templates) |
| **math** | statistics, formatting, sequences (fibonacci, primes) |
| **docker** | containers, images, logs, stats |
| **calendar** | month display, business days, countdown |
| **qr** | ASCII QR code art generation |
| **jwt** | decode and inspect JSON Web Tokens |
| **color** | hex/RGB/HSL conversion, palettes, WCAG contrast |
| **uuid** | UUIDs, passwords, cryptographic tokens |
| **yaml** | parse, validate, YAML/JSON conversion |
| **codegen** | code snippet templates (10 patterns) |


### Persistent Memory

Towel remembers things across sessions in a local SQLite store with three
parallel retrieval tiers fused via Reciprocal Rank Fusion:

- **BM25** (SQLite FTS5) for keyword precision
- **Vector cosine** via `sentence-transformers` (optional: `pip install "towel-ai[embeddings]"`) for paraphrase recall
- **Graph co-retrieval** — pairs of memories that show up together get linked, so a hit on one pulls its neighbors

Auto-capture extracts user / preference / project / deadline facts from
every user turn via conservative regex patterns (clause-bounded negation
so "I'm not a data scientist, I'm a designer" only captures designer).
Decay + auto-forget prune stale, never-recalled fact memories;
user / preference / project entries are protected.

```bash
towel memory stats              # counts, recall fraction, by-source/scope, pattern health
towel memory inspect <key>      # entry detail + salience + related + recent recalls
towel memory tidy --dry-run     # see what would be pruned
towel memory tidy --apply       # actually prune
towel memory consolidate        # find + merge near-duplicates
towel memory export --out backup.json
towel memory import backup.json
towel memory backup             # timestamped snapshot + rotation
towel memory diff baseline.json # what changed since baseline
towel memory reembed            # backfill vectors after installing [embeddings]
towel memory ingest --all       # backfill captures from every saved conversation
towel memory extract --stdin    # LLM-based extraction for what regex missed
towel memory recalls --last 24  # query trail: what was asked, what came back
towel memory activity           # ASCII sparkline of capture rate
towel memory tag KEY add work   # free-form labels for grouping
towel memory list --scope all   # cross-project audit
towel memory forget --tag X     # bulk forget by tag / source / scope
towel memory nudge KEY          # mark useful — bumps recall_count
towel memory promote KEY --to global   # move between scopes
```

**Per-query introspection.** Every `to_prompt_block(query=...)` run is
logged (capped at 5000 most-recent) so `memory inspect <key>` shows
the recent queries that returned it, with rank in result. Answers
"why does the agent remember X when I asked Y?" without grepping logs.

**Auto-LLM-extract** (opt-in, `config.auto_llm_extract: true`): when
regex captures 0 on a user turn, fires a background task that runs
the local LLM extractor against the same backend. Failures are
silent; the same backend serializes the work behind the live response,
so extraction runs when the model is idle.

**Per-project scope.** Memories carry an optional `scope` string —
empty = global (visible everywhere), non-empty = restricted to
callers passing the same scope. When `towel chat` / `towel serve` /
`towel mcp` is launched inside a project (one with `.towel.md`,
`.git`, `pyproject.toml`, etc.), they auto-derive a stable scope
from the project root path. New captures land there by default;
retrieval ORs current-scope with global so universal facts still
surface. Use `--scope all` on CLI commands to audit across every
project from one terminal.

**MCP server.** Run `towel mcp` to expose the store over stdio to any
MCP-compatible client (Claude Code, Cursor, OpenCode, Gemini CLI):

```jsonc
// .mcp.json
{"mcpServers": {"towel-memory": {"command": "towel", "args": ["mcp"]}}}
```

Seven tools become available to the client: `memory_search`, `memory_recall`,
`memory_list`, `memory_remember`, `memory_forget`, `memory_related`,
`memory_stats`.

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
│  │ Sessions  │  │ Routing  │  │ 60 Skills │  │
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

**Tuning knobs** (all optional, defaults in parentheses):

| Field | Default | What it does |
|---|---|---|
| `auto_capture` | `true` | Regex auto-capture on every user turn |
| `auto_llm_extract` | `false` | Background LLM extraction when regex misses (one inference call per quiet turn) |
| `memory_recall_log_cap` | `5000` | Max rows in the per-query recall log (oldest pruned) |
| `dispatch_history_size` | `500` | Dispatch decision ring buffer — hours of audit at typical traffic |
| `worker_inference_timeout` | `300.0` | Seconds the coordinator waits for the next chunk from a remote worker before tearing down the WS. Bump for cold-loaded large models |

## Contributing

```bash
pip install -e ".[dev]"      # install dev deps (pytest, ruff, etc.)
make test                    # run the full suite (~30s, 1200+ tests)
make lint                    # ruff check src/ tests/
make fmt                     # ruff format
make help                    # see all targets
```

Tests have a 60-second per-test ceiling configured in `pyproject.toml` —
runaway loops surface as `Failed: Timeout` rather than hanging the
suite.

## License

MIT
