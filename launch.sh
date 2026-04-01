#!/usr/bin/env bash
# launch.sh - Start Towel and open the web UI on Linux/macOS.

set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

PORT="${TOWEL_PORT:-18743}"
HOST="${TOWEL_HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}/"

towel serve "$@" &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for Towel to start..."
for i in $(seq 1 30); do
    if curl -sf "${URL}health" >/dev/null 2>&1; then
        echo "Towel is up - opening ${URL}"
        if [[ "$(uname -s)" == "Darwin" ]]; then
            open "$URL"
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$URL" >/dev/null 2>&1 || true
        else
            echo "No browser opener found. Open this URL manually: ${URL}"
        fi
        break
    fi
    sleep 0.5
done

wait "$SERVER_PID"
