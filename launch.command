#!/usr/bin/env bash
# launch.command — Start Towel and open the web UI. Don't Panic.

set -euo pipefail

# cd to the script's directory so double-click works from Finder
cd "$(dirname "$0")"

# Activate the venv so `towel` is available
source .venv/bin/activate

PORT="${TOWEL_PORT:-18743}"
HOST="${TOWEL_HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}/"
CONFIG_PATH="${TOWEL_HOME:-$HOME/.towel}/config.toml"

# First-run guard: drop the user into the setup wizard so they pick a
# backend + model before the chat UI starts. After saving, they press
# Enter in this terminal to continue to chat.
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "No config found at $CONFIG_PATH."
    echo "Launching the setup wizard first…"
    towel setup &
    SETUP_PID=$!
    echo "Press Enter once you've saved your configuration to continue to chat."
    read -r _
    kill "$SETUP_PID" 2>/dev/null || true
fi

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

# Keep running in foreground
wait "$SERVER_PID"
