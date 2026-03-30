#!/usr/bin/env bash
# launch.sh — Start Towel and open the web UI. Don't Panic.

set -euo pipefail

PORT="${TOWEL_PORT:-18743}"
HOST="${TOWEL_HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}/"

# Start the gateway in the background
towel serve "$@" &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for the HTTP server to come up
echo "Waiting for Towel to start..."
for i in $(seq 1 30); do
    if curl -sf "${URL}health" >/dev/null 2>&1; then
        echo "Towel is up — opening ${URL}"
        open "$URL"
        break
    fi
    sleep 0.5
done

# Keep running in foreground
wait "$SERVER_PID"
