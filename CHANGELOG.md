# Changelog

Human-friendly summary of notable changes. Full git history is the
source of truth; this file groups commits by theme so you can tell at
a glance whether a release affects you.

## Unreleased

A heavy-development day focused on fleet coordination, model awareness,
and onboarding. ~52 commits.

### Fleet coordination — major

- **Central `Dispatcher`** (`gateway/dispatcher.py`) replaces the
  scattered worker-selection logic. Seven explicit layers (pin →
  affinity → task match → role match → general role → capability
  fallback → idle preempt) each emit a structured `DispatchDecision`
  with reason code, candidates considered, and observability flags
  (`affinity_missed`, `quality_degraded`, `preempted_idle`).
- **Fixed silent `AttributeError`** in the drain/disconnect path —
  `_select_worker()` was called but never existed; sessions were
  silently orphaned.
- **Quality gating**: dispatcher filters by per-task `min_vram_mb` /
  `min_context` from `TASK_REQUIREMENTS`. Falls back to under-spec
  workers with a `quality_degraded` flag rather than refusing —
  coordinator adapts to the fleet it has.
- **CPU pressure** folded into worker scoring (small penalty so
  capability-tied workers break in favour of the calmer one).
- **`/dispatch/recent`** and **`/dispatch/explain`** endpoints + a
  Recent-decisions section in the fleet panel.
- **Fast disconnect notify**: when a worker dies mid-job, the
  coordinator wakes any blocked waiter immediately instead of
  letting them spin on the per-call timeout.
- **Periodic stale-result sweeper** for idle-task cache (with per-task
  TTLs derived from cooldowns).

### Heterogeneous fleet awareness

- Workers self-report **`available_models`** (HF cache scan for MLX,
  `/api/tags` for Ollama, `/v1/models` for llama-server, the three SDK
  aliases for Claude), **`max_param_b_est`** (largest 4-bit quant the
  box can hold), **`disk_free_gb`** + **`disk_total_gb`** (rolled into
  live_resources).
- **`worker_quality_tier`** classifier (`high` / `medium` / `low`)
  derived from VRAM + context + backend.
- **`/fleet/inventory`** aggregates every cached model across the
  fleet — "where can I find model X?" — surfaced in the fleet panel
  with a searchable list.
- **`/fleet/suggest-targets`** ranks workers by has-cached / fits /
  disk-fits / quality-tier for a given model, with a download-size
  estimate. Both the per-worker `replace` and fleet-wide `roll` flows
  in the UI call it to show "✓ cached / ↓ will download / ✗ too small"
  before the destructive action.

### Remote lifecycle management

- **`towel launcher`** daemon — HTTP server you run on each candidate
  worker host. Coordinator can POST `/launch` (auth via
  `$TOWEL_TRIGGER_TOKEN`) to spawn a fresh `towel worker` process.
- **`/fleet/spawn`**, **`/fleet/replace-worker`**, **`/fleet/upgrade`**,
  **`/fleet/rolling-replace`** — coordinator-side orchestration on top
  of the launcher: spawn, drain+respawn, run `pip install --upgrade
  towel`, walk N workers serially with a configurable delay.
- **Worker shutdown WS message** so replace flows exit cleanly instead
  of relying on launchers to kill the process.
- **`towel worker --model <name>`** flag — overrides `config.model.name`
  at startup, the primary knob the coordinator uses to distribute
  different models to different workers.

### Native tools channel (all four backends)

- **MLX, Ollama, llama-server, Claude** runtimes now route tools
  through each backend's native API (`tools=` kwarg /
  `/api/chat` tools field / OpenAI-compat tools / Anthropic
  tool-use blocks) instead of stuffing 330 tool descriptions into the
  system prompt. Slim system prompts, structured tool-call parsing
  with text-fallback for older models.

### Setup + onboarding

- **`towel setup`** — browser GUI to pick backend + model. Reads
  available backends, lists locally-cached models, writes
  `~/.towel/config.toml`.
- **First-run hint** in `towel chat` when no config exists.
- **`launch.sh` / `launch.command`** now drop the user into setup if
  config is missing.
- Consistent messaging across README quickstart, `towel init`
  next-steps, `towel doctor` suggestions.

### Chat UX

- **`/skills` slash command** parallel to the CLI + HTTP endpoint.
- **`/dispatch/explain`** preview without side effects.
- **`/memory`** endpoint with type filter, substring search, limit,
  newest-first ordering. Plus `DELETE /memory/{key}`.
- **Streaming `<tool_call>` markup hidden** from the live token
  stream (Qwen3 native format leaked raw XML before).

### Bug fixes

- **`RAGIndex._split` infinite loop** when `chunk_overlap ≥
  chunk_size` — wedged every full-suite test run for hours until
  found.
- **Tool-error regex bugs**: five `^X:\b` patterns never fired because
  `\b` after `:` requires a word character (which is always a space).
  Tool failures like `File not found: …` were silently classified as
  successes.
- **Bracket markup in CLI output** (`[busy/enabled/ready]`) was being
  parsed as Rich style — escaped to render literally.

### Dev workflow

- **Makefile** with `test`, `lint`, `fmt`, `fix`, `doctor`, `clean`.
- **pytest-timeout** with 60s per-test ceiling so future infinite
  loops surface as clean failures.
- Lint baseline cleared across `src/towel/` and `tests/`.

### Stats

- 1207 tests passing in 27 seconds.
- Zero lint complaints (`ruff check src/towel/ tests/`).
- ~370 new tests today, mostly covering the new endpoints + dispatcher
  paths.

— Towel: Tool Oriented Worker Execution Link. Don't Panic.
