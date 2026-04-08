"""Google OAuth2 helper for Gmail and Calendar skills.

Stores credentials at ~/.towel/google_credentials.json.
On first use, runs a local OAuth flow to get refresh tokens.
Subsequent calls use the stored refresh token silently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("towel.skills.google_auth")

TOWEL_HOME = Path.home() / ".towel"
CREDS_PATH = TOWEL_HOME / "google_credentials.json"
TOKEN_PATH = TOWEL_HOME / "google_token.json"

# OAuth client config — desktop app type (installed)
# Users should replace these with their own from Google Cloud Console,
# or set TOWEL_GOOGLE_CLIENT_ID / TOWEL_GOOGLE_CLIENT_SECRET env vars.
DEFAULT_CLIENT_ID = ""
DEFAULT_CLIENT_SECRET = ""

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.readonly",
]


def _load_client_config() -> dict[str, str]:
    """Load OAuth client config from env vars or credentials file."""
    import os

    client_id = os.environ.get("TOWEL_GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("TOWEL_GOOGLE_CLIENT_SECRET", "")

    if not client_id and CREDS_PATH.exists():
        data = json.loads(CREDS_PATH.read_text())
        installed = data.get("installed", data.get("web", {}))
        client_id = installed.get("client_id", "")
        client_secret = installed.get("client_secret", "")

    if not client_id:
        client_id = DEFAULT_CLIENT_ID
        client_secret = DEFAULT_CLIENT_SECRET

    return {"client_id": client_id, "client_secret": client_secret}


def get_google_credentials() -> Any:
    """Get valid Google OAuth2 credentials, refreshing if needed.

    Returns a google.oauth2.credentials.Credentials object.
    Raises RuntimeError if no credentials are configured.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = None

    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        except Exception:
            log.warning("Failed to load saved Google token, re-authenticating")

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
            return creds
        except Exception as e:
            log.warning("Token refresh failed: %s", e)

    # Need new credentials — run OAuth flow
    config = _load_client_config()
    if not config["client_id"]:
        raise RuntimeError(
            "Google OAuth not configured. Either:\n"
            "1. Place your OAuth client JSON at ~/.towel/google_credentials.json\n"
            "2. Set TOWEL_GOOGLE_CLIENT_ID and TOWEL_GOOGLE_CLIENT_SECRET env vars\n"
            "Get credentials from https://console.cloud.google.com/apis/credentials"
        )

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        },
        SCOPES,
    )
    creds = flow.run_local_server(port=0, open_browser=True)
    TOWEL_HOME.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    log.info("Google OAuth credentials saved to %s", TOKEN_PATH)
    return creds


def build_gmail_service() -> Any:
    """Build an authenticated Gmail API service."""
    from googleapiclient.discovery import build

    creds = get_google_credentials()
    return build("gmail", "v1", credentials=creds)


def build_calendar_service() -> Any:
    """Build an authenticated Google Calendar API service."""
    from googleapiclient.discovery import build

    creds = get_google_credentials()
    return build("calendar", "v3", credentials=creds)
