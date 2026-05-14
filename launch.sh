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
CONFIG_PATH="${TOWEL_HOME:-$HOME/.towel}/config.toml"

# If the user hasn't run setup, open the setup wizard first so they get
# a usable backend + model picked before the chat UI greets them with
# defaults that may not match their machine.
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "No config found at $CONFIG_PATH."
    echo "Launching the setup wizard first…"
    if [[ "$(uname -s)" == "Darwin" ]]; then
        towel setup &
    else
        # Linux: --no-open since xdg-open is unreliable in some sessions;
        # user can navigate manually.
        towel setup --no-open &
        echo "Setup UI at http://127.0.0.1:18749/ — open it once setup launches."
    fi
    SETUP_PID=$!
    # Give the user a chance to complete setup before continuing.
    echo "Press Enter once you've saved your configuration to continue to chat."
    read -r _
    kill "$SETUP_PID" 2>/dev/null || true
fi

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
