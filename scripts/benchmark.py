#!/usr/bin/env python3
"""
Benchmark Towel cluster nodes via the OpenAI-compatible API.

Measures per-node: tokens/sec, TTFT, total latency, and response quality.

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --api http://192.168.50.247:18743
  python scripts/benchmark.py --runs 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
except ImportError:
    print("httpx not found — install with: pip install httpx")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Benchmark prompts — varied in type and output length
# ---------------------------------------------------------------------------

PROMPTS = [
    {
        "id": "short_factual",
        "prompt": "What is the capital of Japan?",
        "expect": ["tokyo"],
        "max_tokens": 64,
    },
    {
        "id": "reasoning",
        "prompt": "If a train travels 120 km in 90 minutes, what is its speed in km/h?",
        "expect": ["80"],
        "max_tokens": 128,
    },
    {
        "id": "code_gen",
        "prompt": "Write a Python function that returns the nth Fibonacci number using memoization.",
        "expect": ["def", "fibonacci", "cache"],
        "max_tokens": 256,
    },
    {
        "id": "long_output",
        "prompt": "Explain how transformers work in machine learning. Be thorough.",
        "expect": ["attention", "encoder", "embedding"],
        "max_tokens": 512,
    },
    {
        "id": "instruction_follow",
        "prompt": "List exactly 5 programming languages, one per line, nothing else.",
        "expect": ["python", "rust", "go", "java", "c"],
        "max_tokens": 128,
    },
]


@dataclass
class RunResult:
    prompt_id: str
    ttft_ms: float        # time to first token (streaming)
    total_ms: float       # total wall time
    output_tokens: int
    tps: float            # tokens per second
    quality_score: float  # 0.0–1.0 keyword match
    response_snippet: str


@dataclass
class NodeResult:
    worker_id: str
    capabilities: dict[str, Any]
    runs: list[RunResult] = field(default_factory=list)
    errors: int = 0

    @property
    def avg_tps(self) -> float:
        vals = [r.tps for r in self.runs if r.tps > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_ttft_ms(self) -> float:
        vals = [r.ttft_ms for r in self.runs if r.ttft_ms > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_total_ms(self) -> float:
        vals = [r.total_ms for r in self.runs]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_quality(self) -> float:
        vals = [r.quality_score for r in self.runs]
        return statistics.mean(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quality_score(response: str, keywords: list[str]) -> float:
    text = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords) if keywords else 1.0


def run_streaming(client: httpx.Client, api_base: str, prompt: dict, session_id: str) -> RunResult:
    """Run a single prompt with streaming and capture TTFT + TPS."""
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "max_tokens": prompt["max_tokens"],
        "stream": True,
        "session_id": session_id,
    }

    ttft_ms = 0.0
    first_token = False
    full_text = ""
    token_count = 0
    t_start = time.perf_counter()
    t_first = t_start

    with client.stream("POST", url, json=payload, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if not first_token:
                    t_first = time.perf_counter()
                    ttft_ms = (t_first - t_start) * 1000
                    first_token = True
                full_text += content
                token_count += 1

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000
    elapsed_after_first = (t_end - t_first) if first_token else (t_end - t_start)
    tps = token_count / elapsed_after_first if elapsed_after_first > 0 else 0.0

    return RunResult(
        prompt_id=prompt["id"],
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        output_tokens=token_count,
        tps=tps,
        quality_score=quality_score(full_text, prompt.get("expect", [])),
        response_snippet=full_text[:120].replace("\n", " "),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Towel cluster nodes")
    parser.add_argument("--api", default="http://127.0.0.1:18743", help="Towel HTTP API base URL")
    parser.add_argument("--runs", type=int, default=1, help="Runs per prompt per node")
    parser.add_argument("--warmup", action="store_true", help="Send a warmup request first")
    args = parser.parse_args()

    api = args.api.rstrip("/")

    with httpx.Client(timeout=30) as client:
        # Fetch workers
        try:
            workers_resp = client.get(f"{api}/workers")
            workers_resp.raise_for_status()
            workers_data = workers_resp.json()
        except Exception as exc:
            print(f"Cannot reach gateway at {api}: {exc}")
            sys.exit(1)

        workers = workers_data.get("workers", [])

    if not workers:
        print("No workers connected. Start workers with: towel worker --controller <url>")
        sys.exit(1)

    print(f"\nTowel Cluster Benchmark")
    print(f"API: {api}")
    print(f"Workers: {len(workers)}")
    print(f"Prompts: {len(PROMPTS)} × {args.runs} run(s) each\n")
    print("─" * 72)

    node_results: list[NodeResult] = []

    for worker in workers:
        wid = worker["id"]
        caps = worker.get("capabilities", {})
        model = caps.get("model", "unknown")
        backend = caps.get("backend", "?")
        gpu_info = ""
        if gpus := caps.get("gpus"):
            gpu_info = f" [{gpus[0]['name']} {gpus[0]['vram_mb']//1024}GB]"
        elif "apple" in caps.get("hostname", "").lower() or backend == "mlx":
            gpu_info = " [Apple Silicon]"

        print(f"\nNode: {wid}  |  {backend}  |  {model}{gpu_info}")
        print(f"{'Prompt':<22} {'TTFT':>8} {'Total':>8} {'Tokens':>7} {'TPS':>7} {'Quality':>8}")
        print("─" * 72)

        node = NodeResult(worker_id=wid, capabilities=caps)

        # Use a pinned session so all prompts go to this worker
        session_id = f"bench-{wid}"

        # Pin session to this worker
        with httpx.Client(timeout=30) as client:
            try:
                client.post(
                    f"{api}/workers/{wid}/state",
                    json={"pin_session": session_id},
                )
            except Exception:
                pass  # pin endpoint may not exist, routing will still work

        with httpx.Client(timeout=30) as client:
            if args.warmup:
                try:
                    client.post(
                        f"{api}/v1/chat/completions",
                        json={
                            "model": "default",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 8,
                            "stream": False,
                            "session_id": session_id,
                        },
                        timeout=60,
                    )
                except Exception:
                    pass

            for prompt in PROMPTS:
                run_results = []
                for _ in range(args.runs):
                    try:
                        result = run_streaming(client, api, prompt, session_id)
                        run_results.append(result)
                        node.runs.append(result)
                    except Exception as exc:
                        node.errors += 1
                        print(f"  ERROR on {prompt['id']}: {exc}")

                if run_results:
                    avg = RunResult(
                        prompt_id=prompt["id"],
                        ttft_ms=statistics.mean(r.ttft_ms for r in run_results),
                        total_ms=statistics.mean(r.total_ms for r in run_results),
                        output_tokens=int(statistics.mean(r.output_tokens for r in run_results)),
                        tps=statistics.mean(r.tps for r in run_results),
                        quality_score=statistics.mean(r.quality_score for r in run_results),
                        response_snippet=run_results[-1].response_snippet,
                    )
                    q_bar = "█" * int(avg.quality_score * 5) + "░" * (5 - int(avg.quality_score * 5))
                    print(
                        f"  {prompt['id']:<20} "
                        f"{avg.ttft_ms:>7.0f}ms "
                        f"{avg.total_ms:>7.0f}ms "
                        f"{avg.output_tokens:>7} "
                        f"{avg.tps:>6.1f}/s "
                        f"  {q_bar} {avg.quality_score:.0%}"
                    )

        node_results.append(node)

    # Summary
    print("\n" + "═" * 72)
    print("SUMMARY")
    print("═" * 72)
    print(f"{'Node':<20} {'Backend':<8} {'Avg TPS':>9} {'Avg TTFT':>10} {'Quality':>9} {'Errors':>7}")
    print("─" * 72)
    for node in node_results:
        caps = node.capabilities
        print(
            f"  {node.worker_id:<18} "
            f"{caps.get('backend','?'):<8} "
            f"{node.avg_tps:>8.1f}/s "
            f"{node.avg_ttft_ms:>9.0f}ms "
            f"{node.avg_quality:>8.0%} "
            f"{node.errors:>7}"
        )

    if len(node_results) > 1:
        fastest = max(node_results, key=lambda n: n.avg_tps)
        print(f"\nFastest node: {fastest.worker_id} @ {fastest.avg_tps:.1f} tok/s")

    print()


if __name__ == "__main__":
    main()
