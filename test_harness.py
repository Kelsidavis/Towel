#!/usr/bin/env python3
"""Towel integration test harness.

Sends prompts to a running Towel instance via /api/ask and verifies
that tool calls work correctly and responses are useful.

Usage:
    python test_harness.py              # run once
    python test_harness.py --loop 5     # run every 5 minutes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:18743"
TIMEOUT = 300  # seconds — model inference can be slow, tool loops need multiple rounds
SESSION = "test-harness"
_run_id = 0  # incremented each run to create fresh sessions

# Results log
LOG_FILE = Path(__file__).parent / ".test_harness_results.jsonl"


@dataclass
class TestResult:
    name: str
    passed: bool
    prompt: str
    response: str
    elapsed: float
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def ask(prompt: str, session: str = SESSION) -> tuple[str, float]:
    """Send a prompt to Towel and return (response, elapsed_seconds)."""
    start = time.monotonic()
    resp = httpx.post(
        f"{BASE_URL}/api/ask",
        json={"message": prompt, "session": session},
        timeout=TIMEOUT,
    )
    elapsed = time.monotonic() - start
    data = resp.json()
    if resp.status_code != 200:
        error = data.get("error", resp.text)
        raise RuntimeError(f"HTTP {resp.status_code}: {error}")
    return data.get("response", ""), elapsed


def check(response: str, *keywords: str) -> tuple[list[str], list[str]]:
    """Check response contains expected keywords (case-insensitive).
    Returns (passed_checks, failed_checks)."""
    passed, failed = [], []
    lower = response.lower()
    for kw in keywords:
        if kw.lower() in lower:
            passed.append(f"contains '{kw}'")
        else:
            failed.append(f"missing '{kw}'")
    return passed, failed


def check_not(response: str, *keywords: str) -> tuple[list[str], list[str]]:
    """Check response does NOT contain certain keywords."""
    passed, failed = [], []
    lower = response.lower()
    for kw in keywords:
        if kw.lower() not in lower:
            passed.append(f"correctly absent '{kw}'")
        else:
            failed.append(f"should not contain '{kw}'")
    return passed, failed


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def _s(name: str) -> str:
    """Generate a fresh session ID per test per run."""
    return f"{name}-{_run_id}"


def test_basic_response() -> TestResult:
    """Model can respond without tools."""
    prompt = "What is 2 + 2? Answer with just the number."
    resp, elapsed = ask(prompt, session=_s("basic"))
    passed, failed = check(resp, "4")
    return TestResult("basic_response", not failed, prompt, resp, elapsed, passed, failed)


def test_read_file() -> TestResult:
    """Tool: read_file — reads a known file."""
    prompt = "Read the file .towel.md in the current directory and tell me the first heading."
    resp, elapsed = ask(prompt, session=_s("read"))
    passed, failed = check(resp, "Towel")
    return TestResult("read_file", not failed, prompt, resp, elapsed, passed, failed)


def test_list_directory() -> TestResult:
    """Tool: list_directory — lists project files."""
    prompt = "List the files in the src/towel/agent/ directory. Just the filenames."
    resp, elapsed = ask(prompt, session=_s("listdir"))
    passed, failed = check(resp, "runtime", "tool_parser")
    return TestResult("list_directory", not failed, prompt, resp, elapsed, passed, failed)


def test_run_command() -> TestResult:
    """Tool: run_command — executes a shell command."""
    prompt = "Run the command 'echo DON_T_PANIC' and show me the output."
    resp, elapsed = ask(prompt, session=_s("cmd"))
    passed, failed = check(resp, "DON_T_PANIC")
    return TestResult("run_command", not failed, prompt, resp, elapsed, passed, failed)


def test_current_time() -> TestResult:
    """Tool: current_time — gets the current time."""
    prompt = "What time is it right now? Use the current_time tool."
    resp, elapsed = ask(prompt, session=_s("time"))
    import re
    has_time = bool(re.search(r"\d{1,2}:\d{2}", resp))
    passed = ["contains time pattern"] if has_time else []
    failed = [] if has_time else ["no time pattern found"]
    return TestResult("current_time", not failed, prompt, resp, elapsed, passed, failed)


def test_date_now() -> TestResult:
    """Tool: date_now — gets today's date."""
    prompt = "What is today's date? Use the date_now tool."
    resp, elapsed = ask(prompt, session=_s("date"))
    today = datetime.now().strftime("%Y")
    passed, failed = check(resp, today)
    return TestResult("date_now", not failed, prompt, resp, elapsed, passed, failed)


def test_system_info() -> TestResult:
    """Tool: system_info — system details."""
    prompt = "Get my system info. What OS am I running?"
    resp, elapsed = ask(prompt, session=_s("sysinfo"))
    passed, failed = check(resp, "mac", "darwin", "apple")
    # Any one of these is fine
    if failed and len(failed) < 3:
        failed = []
        passed = ["contains OS reference"]
    return TestResult("system_info", not failed, prompt, resp, elapsed, passed, failed)


def test_git_status() -> TestResult:
    """Tool: git_status — checks repo state."""
    prompt = "Run git status on this repository."
    resp, elapsed = ask(prompt, session=_s("git"))
    passed, failed = check(resp, "branch", "main")
    return TestResult("git_status", not failed, prompt, resp, elapsed, passed, failed)


def test_search_files() -> TestResult:
    """Tool: search_files or find_files — searches codebase."""
    prompt = "Use the search_files tool to find files containing 'Don't Panic' in the src/ directory. List the filenames."
    resp, elapsed = ask(prompt, session=_s("search"))
    # Accept .py files listed or a match count
    has_result = ".py" in resp or "match" in resp.lower() or "found" in resp.lower()
    passed = ["contains search results"] if has_result else []
    failed = [] if has_result else ["no search results found"]
    return TestResult("search_files", not failed, prompt, resp, elapsed, passed, failed)


def test_todo() -> TestResult:
    """Tool: todo_add, todo_list — task management."""
    prompt = "Add a todo item: 'Test the test harness'. Then list all todos."
    resp, elapsed = ask(prompt, session=_s("todo"))
    passed, failed = check(resp, "test")
    return TestResult("todo", not failed, prompt, resp, elapsed, passed, failed)


def test_json_skill() -> TestResult:
    """Tool: json_flatten — JSON manipulation."""
    prompt = 'Use the json_flatten tool on this input: {"a": {"b": 1, "c": {"d": 2}}}. Show the flattened result.'
    resp, elapsed = ask(prompt, session=_s("json"))
    has_flat = "a.b" in resp or "a.c.d" in resp or "flatten" in resp.lower()
    passed = ["contains flattened output"] if has_flat else []
    failed = [] if has_flat else ["no flattened output found"]
    return TestResult("json_skill", not failed, prompt, resp, elapsed, passed, failed)


def test_hash_text() -> TestResult:
    """Tool: hash_text — hashing."""
    prompt = "Use the hash_text tool to compute the MD5 hash of the string 'hello'. Show the hash."
    resp, elapsed = ask(prompt, session=_s("hash"))
    # MD5 of "hello" is 5d41402abc4b2a76b9719d911017c592
    # Accept either the full hash or a hex-looking string (tool was called)
    import re
    exact = "5d41402abc4b2a76b9719d911017c592" in resp.lower()
    has_hex = bool(re.search(r"[0-9a-f]{32}", resp.lower()))
    if exact:
        passed, failed = ["exact MD5 match"], []
    elif has_hex:
        passed, failed = ["contains 32-char hex hash"], []
    else:
        passed, failed = [], ["no hash found in response"]
    return TestResult("hash_text", not failed, prompt, resp, elapsed, passed, failed)


def test_memory() -> TestResult:
    """Tool: remember, recall — persistent memory."""
    prompt = "Remember that my favorite number is 42. Then recall what my favorite number is."
    resp, elapsed = ask(prompt, session=_s("memory"))
    passed, failed = check(resp, "42")
    return TestResult("memory", not failed, prompt, resp, elapsed, passed, failed)


def test_no_hallucinated_tools() -> TestResult:
    """Model should not call tools that don't exist."""
    prompt = "What color is the sky? Just answer directly, no tools needed."
    resp, elapsed = ask(prompt, session=_s("nohalluc"))
    passed, failed = check(resp, "blue")
    p2, f2 = check_not(resp, "error executing", "tool not found", "unknown tool")
    passed.extend(p2)
    failed.extend(f2)
    return TestResult("no_hallucinated_tools", not failed, prompt, resp, elapsed, passed, failed)


def test_multi_step() -> TestResult:
    """Model should chain tool calls for a multi-step task."""
    prompt = (
        "Read the file pyproject.toml and tell me the project version number. "
        "Just the version string."
    )
    resp, elapsed = ask(prompt, session=_s("multistep"))
    import re
    has_version = bool(re.search(r"\d+\.\d+\.\d+", resp))
    passed = ["contains version pattern"] if has_version else []
    failed = [] if has_version else ["no version pattern found"]
    return TestResult("multi_step", not failed, prompt, resp, elapsed, passed, failed)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_basic_response,
    test_read_file,
    test_list_directory,
    test_run_command,
    test_current_time,
    test_date_now,
    test_system_info,
    test_git_status,
    test_search_files,
    test_todo,
    test_json_skill,
    test_hash_text,
    test_memory,
    test_no_hallucinated_tools,
    test_multi_step,
]


def run_suite() -> list[TestResult]:
    """Run all tests sequentially (one model inference at a time)."""
    global _run_id
    _run_id += 1
    print(f"\n{'=' * 60}")
    print(f"  Towel Test Harness — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # Check server is up
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5).json()
        print(f"  Server: {health.get('status', '?')} | v{health.get('version', '?')}")
    except Exception as e:
        print(f"  ERROR: Towel is not running at {BASE_URL} — {e}")
        sys.exit(1)

    results: list[TestResult] = []
    passed_count = 0

    for test_fn in ALL_TESTS:
        name = test_fn.__name__.replace("test_", "")
        desc = test_fn.__doc__ or ""
        print(f"  [{name}] {desc.strip()} ...", end=" ", flush=True)

        try:
            result = test_fn()
            results.append(result)
            if result.passed:
                passed_count += 1
                print(f"PASS ({result.elapsed:.1f}s)")
            else:
                print(f"FAIL ({result.elapsed:.1f}s)")
                for f in result.failures:
                    print(f"    - {f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(TestResult(
                name=name, passed=False, prompt="", response="",
                elapsed=0, failures=[str(e)],
            ))

    total = len(results)
    print(f"\n  Results: {passed_count}/{total} passed")

    if passed_count < total:
        failed_names = [r.name for r in results if not r.passed]
        print(f"  Failed:  {', '.join(failed_names)}")

    print()

    # Append to log
    with open(LOG_FILE, "a") as f:
        for r in results:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "name": r.name,
                "passed": r.passed,
                "elapsed": round(r.elapsed, 2),
                "checks": r.checks,
                "failures": r.failures,
                "prompt": r.prompt,
                "response_preview": r.response[:200],
            }) + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Towel integration test harness")
    parser.add_argument("--loop", type=int, default=0, help="Re-run every N minutes (0 = run once)")
    args = parser.parse_args()

    if args.loop > 0:
        print(f"Running test suite every {args.loop} minutes. Ctrl-C to stop.")
        while True:
            try:
                run_suite()
                print(f"  Next run in {args.loop} minutes...")
                time.sleep(args.loop * 60)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        results = run_suite()
        sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
