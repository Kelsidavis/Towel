# Towel

**Don't Panic.**

Your local AI assistant, powered by MLX on Apple Silicon.

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Initialize config
towel init

# Start chatting
towel chat

# Or start the gateway
towel serve
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                  Gateway                      │
│          ws://127.0.0.1:18742                │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Sessions  │  │ Routing  │  │  Skills   │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└──────────┬───────────────────────┬───────────┘
           │                       │
    ┌──────┴──────┐         ┌──────┴──────┐
    │   Agent     │         │  Channels   │
    │  Runtime    │         │             │
    │  (MLX)      │         │  CLI        │
    │             │         │  WebChat    │
    │  Llama      │         │  Telegram   │
    │  Mistral    │         │  Discord    │
    │  Qwen       │         │  ...        │
    └─────────────┘         └─────────────┘
```

## Project Structure

- `src/towel/agent/` — MLX model loading, inference, tool dispatch
- `src/towel/gateway/` — WebSocket control plane + HTTP API
- `src/towel/channels/` — Messaging platform adapters
- `src/towel/skills/` — The Laundromat (skills/tools registry)
- `src/towel/nodes/` — Device capability providers
- `src/towel/canvas/` — Agent-driven visual workspace
- `src/towel/cli/` — Command-line interface

## License

MIT
