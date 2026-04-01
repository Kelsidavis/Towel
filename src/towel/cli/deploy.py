"""Deploy — generate deployment artifacts for running Towel in production.

Generates:
  - Dockerfile
  - docker-compose.yml
  - systemd unit file
  - .env template
  - Procfile (Heroku)
  - fly.toml (Fly.io)
"""

from __future__ import annotations

from pathlib import Path

DOCKERFILE = '''FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir towel-ai
COPY .env* ./
EXPOSE 18742 18743
CMD ["towel", "serve"]
'''

COMPOSE = '''version: "3.8"
services:
  towel:
    build: .
    ports:
      - "18742:18742"
      - "18743:18743"
    volumes:
      - towel-data:/root/.towel
    env_file:
      - .env
    restart: unless-stopped

volumes:
  towel-data:
'''

SYSTEMD = '''[Unit]
Description=Towel AI Assistant
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory=/home/{user}
ExecStart=/usr/local/bin/towel serve
Restart=always
RestartSec=5
EnvironmentFile=/home/{user}/.towel/.env

[Install]
WantedBy=multi-user.target
'''

PROCFILE = 'web: towel serve --host 0.0.0.0 --port $PORT\n'

FLY_TOML = '''app = "towel-ai"
primary_region = "sjc"

[build]
  builder = "paketobuildpacks/builder:base"

[env]
  PORT = "8080"

[http_service]
  internal_port = 18743
  force_https = true

[[services]]
  internal_port = 18742
  protocol = "tcp"
  [[services.ports]]
    port = 18742
'''

ENV_TEMPLATE = '''# Towel configuration
# Copy to .env and fill in values

# Model (default: Llama 3.3 70B 4-bit)
# TOWEL_MODEL=mlx-community/Llama-3.3-70B-Instruct-4bit

# Channel tokens (uncomment as needed)
# DISCORD_TOKEN=your-discord-bot-token
# TELEGRAM_TOKEN=your-telegram-bot-token
# SLACK_BOT_TOKEN=xoxb-your-bot-token
# SLACK_APP_TOKEN=xapp-your-app-token

# Webhook auth
# WEBHOOK_TOKEN=your-secret-token
'''

TARGETS = {
    "docker": [("Dockerfile", DOCKERFILE), ("docker-compose.yml", COMPOSE), (".env.example", ENV_TEMPLATE)],
    "systemd": [("towel.service", SYSTEMD), (".env.example", ENV_TEMPLATE)],
    "heroku": [("Procfile", PROCFILE), (".env.example", ENV_TEMPLATE)],
    "fly": [("fly.toml", FLY_TOML), ("Dockerfile", DOCKERFILE), (".env.example", ENV_TEMPLATE)],
    "all": [],  # filled dynamically
}


def generate_deploy(target: str, output_dir: Path, user: str = "towel") -> list[Path]:
    """Generate deployment files. Returns list of created paths."""
    if target == "all":
        files_spec = []
        seen = set()
        for t in ["docker", "systemd", "heroku", "fly"]:
            for name, content in TARGETS[t]:
                if name not in seen:
                    files_spec.append((name, content))
                    seen.add(name)
    else:
        files_spec = TARGETS.get(target, [])

    if not files_spec:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for name, content in files_spec:
        content = content.replace("{user}", user)
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        created.append(path)

    return created
