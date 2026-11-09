"""
box_login_and_token_storage.py
==============================

Logs in to Box with OAuth 2.0 and keeps the login tokens on disk, so you only
have to approve access in the browser once.

How Box OAuth works (short version)
-----------------------------------
1. First time only: you open an "authorize" URL in the browser, approve the
   app, and copy the ``code=...`` value from the address bar you are sent to.
2. That code is exchanged for two tokens:
     access token  — valid ~60 minutes, sent with every request
     refresh token — valid 60 days, used to get a new access token
3. Box refresh tokens are SINGLE-USE: every refresh returns a new pair, and the
   old refresh token stops working. The new pair must therefore be saved
   immediately, otherwise the next session cannot log in. The Box SDK calls
   ``JsonFileTokenStorage.store()`` every time it refreshes, and this class
   writes the new tokens to disk right away (see "Safety" below).

Compatibility
-------------
The token file has the same format ({"access_token": ..., "refresh_token":
...}) and default name ("oauth_box_auth.json") as the existing helper in
an older in-house Box login script. Both can therefore
use the same login. Avoid running both at exactly the same moment, since each
refresh invalidates the previous refresh token.

Safety
------
* The credentials (client id / secret) are read from a file OUTSIDE the
  repository and are never printed.
* Tokens are written atomically (to a temporary file that is then renamed),
  so a crash while saving cannot leave a half-written token file behind.
* Token files are made readable by the owner only (permission 600).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_FILE_NAME = "oauth_box_auth.json"


# =============================================================================
# Credentials file
# =============================================================================
def read_credentials_file(credentials_file: str | Path) -> dict[str, str]:
    """
    Read a simple KEY=VALUE text file (like a .env file).

    Blank lines and lines starting with '#' are ignored; quotes around values
    are removed. Required keys: CLIENT_ID, CLIENT_SECRET.
    """
    credentials_file = Path(credentials_file).expanduser()
    if not credentials_file.is_file():
        raise FileNotFoundError(
            f"Box credentials file not found: {credentials_file}\n"
            "Create it from box_credentials.example.env (keep it outside git)."
        )

    values: dict[str, str] = {}
    for line in credentials_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")

    missing = [key for key in ("CLIENT_ID", "CLIENT_SECRET") if not values.get(key)]
    if missing:
        raise ValueError(f"Box credentials file {credentials_file} is missing: {missing}")
    return values


def choose_token_folder(
    credentials_file: str | Path,
    credentials: dict[str, str],
    token_folder: str | Path | None = None,
) -> Path:
    """
    Decide where the token file lives, in this order:

    1. ``token_folder`` given explicitly (e.g. from the settings file)
    2. BOX_OAUTH_LOC from the credentials file, IF that folder exists here
       (it may point to another computer, e.g. /home/olduser/...)
    3. the folder that contains the credentials file
    """
    if token_folder:
        return Path(token_folder).expanduser()
    from_credentials = credentials.get("BOX_OAUTH_LOC")
    if from_credentials and Path(from_credentials).expanduser().is_dir():
        return Path(from_credentials).expanduser()
    return Path(credentials_file).expanduser().parent


# =============================================================================
# Token storage (called by the Box SDK)
# =============================================================================
def _make_token_storage_class():
    """
    Build the storage class lazily, because it must inherit from the Box SDK's
    ``TokenStorage`` and the SDK is an optional dependency.
    """
    from box_sdk_gen import AccessToken, TokenStorage

    class JsonFileTokenStorage(TokenStorage):
        """Keeps Box tokens in a small JSON file; see module docstring."""

        def __init__(self, token_file: Path):
            self.token_file = Path(token_file)

        def store(self, token: AccessToken) -> None:
            """Save a new token pair atomically and with owner-only permissions."""
            payload = {"access_token": token.access_token, "refresh_token": token.refresh_token}
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary file in the same folder, then rename it over
            # the real file: the rename is atomic on Linux/macOS.
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=self.token_file.parent, prefix=".box_token_", suffix=".tmp"
            )
            try:
                with os.fdopen(file_descriptor, "w") as handle:
                    json.dump(payload, handle, indent=4)
                os.chmod(temporary_name, 0o600)
                os.replace(temporary_name, self.token_file)
            except BaseException:
                Path(temporary_name).unlink(missing_ok=True)
                raise
            logger.info("Box tokens refreshed and saved to %s", self.token_file)

        def get(self) -> AccessToken | None:
            """Load the saved tokens, or None if there is no usable token file."""
            if not self.token_file.is_file():
                return None
            try:
                payload = json.loads(self.token_file.read_text())
            except json.JSONDecodeError:
                logger.warning("Token file %s is not valid JSON; ignoring it.", self.token_file)
                return None
            if not payload.get("refresh_token"):
                return None
            return AccessToken(
                access_token=payload.get("access_token") or "",
                refresh_token=payload.get("refresh_token"),
            )

        def clear(self) -> None:
            self.token_file.unlink(missing_ok=True)

    return JsonFileTokenStorage


# =============================================================================
# Creating a logged-in client
# =============================================================================
def create_box_client(
    credentials_file: str | Path,
    token_folder: str | Path | None = None,
    allow_interactive_login: bool = True,
):
    """
    Return a logged-in ``box_sdk_gen.BoxClient``.

    Parameters
    ----------
    credentials_file : path to the KEY=VALUE file with CLIENT_ID / CLIENT_SECRET
    token_folder : folder of ``oauth_box_auth.json`` (see ``choose_token_folder``)
    allow_interactive_login : if the saved tokens do not work, print the
        authorize URL and ask for the code (works in notebooks and terminals).
        If False, raise an error instead (useful for unattended scripts).
    """
    try:
        from box_sdk_gen import BoxClient, BoxOAuth, OAuthConfig
    except ImportError as error:
        raise ImportError(
            "Box access needs the Box SDK: pip install 'boxsdk>=10'"
        ) from error

    credentials = read_credentials_file(credentials_file)
    token_file = choose_token_folder(credentials_file, credentials, token_folder) / DEFAULT_TOKEN_FILE_NAME
    storage = _make_token_storage_class()(token_file)

    auth = BoxOAuth(
        OAuthConfig(
            client_id=credentials["CLIENT_ID"],
            client_secret=credentials["CLIENT_SECRET"],
            token_storage=storage,
        )
    )
    client = BoxClient(auth=auth)

    # --- 1) Try the saved tokens -----------------------------------------------
    if storage.get() is not None:
        try:
            # Any cheap request works; if the access token has expired, the SDK
            # refreshes it automatically and calls storage.store() with the new pair.
            user = client.users.get_user_me()
            logger.info("Connected to Box as %s", user.name)
            return client
        except Exception as error:
            logger.warning("Saved Box login did not work (%s).", type(error).__name__)

    # --- 2) Fall back to a browser login ---------------------------------------------
    if not allow_interactive_login:
        raise PermissionError(
            f"No valid Box login in {token_file}. Run create_box_client(...) once "
            "interactively (e.g. in a notebook) to log in."
        )
    authorize_url = auth.get_authorize_url()
    print(
        "Box login needed.\n"
        "1. Open this URL in your browser and approve access:\n"
        f"   {authorize_url}\n"
        "2. You will be sent to an address containing 'code=XXXX'. Copy XXXX."
    )
    authorization_code = input("Paste the code here and press Enter: ").strip()
    auth.get_tokens_authorization_code_grant(authorization_code)  # stores tokens via storage.store
    user = client.users.get_user_me()
    print(f"Connected to Box as {user.name}. Login saved to {token_file}")
    return client
