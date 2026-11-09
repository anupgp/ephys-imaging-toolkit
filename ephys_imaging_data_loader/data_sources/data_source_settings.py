"""
data_source_settings.py
=======================

Reads ``data_source_settings.toml`` and builds the data source you ask for:

    open_data_source("local")  -> LocalFolderDataSource   (this computer only)
    open_data_source("box")    -> BoxCloudDataSource      (Box, via the cache)
    open_data_source("auto")   -> LocalThenBoxDataSource  (local, else Box)
    open_data_source()         -> whatever "default_mode" says in the settings

All machine-specific information (folder paths, Box folder ids, credentials
location, cache size) lives in the settings file, NOT in the code. See
``data_source_settings.example.toml`` for a documented template.

Where the settings file is looked for (first match wins)
--------------------------------------------------------
1. the ``settings_file`` argument
2. the environment variable ``EPHYS_IMAGING_SETTINGS``
3. ``data_source_settings.toml`` in the project folder (next to pyproject.toml)
4. ``~/.config/ephys_imaging_data_loader/data_source_settings.toml``
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .base_data_source import DataSource
from .box_cloud_data_source import BoxCloudDataSource
from .box_download_cache import BoxDownloadCache
from .local_folder_data_source import LocalFolderDataSource
from .local_then_box_data_source import LocalThenBoxDataSource

SETTINGS_FILE_NAME = "data_source_settings.toml"
SETTINGS_ENVIRONMENT_VARIABLE = "EPHYS_IMAGING_SETTINGS"
# The project folder = two levels above this file (…/ephys_imaging_data_loader/).
PROJECT_FOLDER = Path(__file__).resolve().parents[2]
VALID_MODES = ("local", "box", "auto")


@dataclass
class RootSettings:
    """One named data root: where it is locally and/or on Box."""

    name: str
    description: str = ""
    local_folder: str = ""
    box_folder_id: str = ""


@dataclass
class DataSourceSettings:
    """Everything read from the settings file."""

    settings_file: Path | None
    default_mode: str = "auto"
    cache_folder: str = "~/.cache/ephys_imaging_data_loader"
    cache_size_limit_gb: float | None = 50
    box_credentials_file: str = ""
    box_token_folder: str = ""
    roots: dict[str, RootSettings] = field(default_factory=dict)
    # [catalog] section: the experiment details sheet and which roots its files live in
    catalog_csv: str = ""
    catalog_ephys_root: str = ""
    catalog_imaging_root: str = ""

    def describe(self) -> str:
        """Readable overview of the configured roots."""
        lines = [f"Settings file : {self.settings_file}", f"Default mode  : {self.default_mode}", "Roots:"]
        for root in self.roots.values():
            lines.append(f"  {root.name:<20} {root.description}")
            lines.append(f"  {'':<20} local: {root.local_folder or '-'}")
            lines.append(f"  {'':<20} box  : {root.box_folder_id or '-'}")
        return "\n".join(lines)


def find_settings_file(settings_file: str | Path | None = None) -> Path | None:
    """Return the first existing settings file (see module docstring), or None."""
    candidates = [
        settings_file,
        os.environ.get(SETTINGS_ENVIRONMENT_VARIABLE),
        PROJECT_FOLDER / SETTINGS_FILE_NAME,
        Path.home() / ".config" / "ephys_imaging_data_loader" / SETTINGS_FILE_NAME,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser()
    if settings_file:
        raise FileNotFoundError(f"Settings file not found: {settings_file}")
    return None


def load_settings(settings_file: str | Path | None = None) -> DataSourceSettings:
    """Read the settings file into a ``DataSourceSettings`` object."""
    path = find_settings_file(settings_file)
    if path is None:
        raise FileNotFoundError(
            "No data_source_settings.toml found. Copy data_source_settings.example.toml "
            f"to {PROJECT_FOLDER / SETTINGS_FILE_NAME} and edit the folder paths."
        )
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    general = raw.get("general", {})
    box = raw.get("box", {})
    catalog = raw.get("catalog", {})
    settings = DataSourceSettings(
        settings_file=path,
        default_mode=general.get("default_mode", "auto"),
        cache_folder=general.get("cache_folder", "~/.cache/ephys_imaging_data_loader"),
        cache_size_limit_gb=general.get("cache_size_limit_gb", 50),
        box_credentials_file=box.get("credentials_file", ""),
        box_token_folder=box.get("token_folder", ""),
        catalog_csv=catalog.get("experiment_details_csv", ""),
        catalog_ephys_root=catalog.get("ephys_root", ""),
        catalog_imaging_root=catalog.get("imaging_root", ""),
    )
    for name, values in raw.get("roots", {}).items():
        settings.roots[name] = RootSettings(
            name=name,
            description=values.get("description", ""),
            local_folder=values.get("local_folder", ""),
            box_folder_id=str(values.get("box_folder_id", "") or ""),
        )
    if settings.default_mode not in VALID_MODES:
        raise ValueError(f"default_mode must be one of {VALID_MODES}, not '{settings.default_mode}'.")
    return settings


def _build_local_source(settings: DataSourceSettings) -> LocalFolderDataSource:
    return LocalFolderDataSource({name: root.local_folder for name, root in settings.roots.items()})


def _build_box_source(settings: DataSourceSettings, box_client=None) -> BoxCloudDataSource:
    """Box source with a lazy login (no network access until first use)."""
    from .box_login_and_token_storage import create_box_client

    size_limit = (
        None if settings.cache_size_limit_gb in (None, 0) else int(float(settings.cache_size_limit_gb) * 1e9)
    )
    cache = BoxDownloadCache(settings.cache_folder, size_limit)

    def client_factory():
        if not settings.box_credentials_file:
            raise ValueError("No [box] credentials_file in the settings file.")
        return create_box_client(settings.box_credentials_file, settings.box_token_folder or None)

    return BoxCloudDataSource(
        {name: root.box_folder_id for name, root in settings.roots.items()},
        cache,
        client=box_client,
        client_factory=client_factory,
    )


def open_data_source(
    mode: str | None = None,
    settings_file: str | Path | None = None,
    box_client=None,
) -> DataSource:
    """
    Build a data source from the settings file.

    Parameters
    ----------
    mode : "local", "box", "auto" or None (= default_mode from the settings)
    settings_file : optional explicit settings file path
    box_client : optional already-logged-in Box client (mainly for testing)

    Examples
    --------
    >>> local = open_data_source("local")    # never touches the internet
    >>> cloud = open_data_source("box")      # always Box (cached downloads)
    >>> either = open_data_source("auto")    # local copy if present, else Box
    """
    settings = load_settings(settings_file)
    mode = (mode or settings.default_mode).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, not '{mode}'.")

    if mode == "local":
        return _build_local_source(settings)
    if mode == "box":
        return _build_box_source(settings, box_client)
    # "auto": Box part is only built if at least one root has a Box folder id.
    has_box_roots = any(root.box_folder_id for root in settings.roots.values())
    return LocalThenBoxDataSource(
        _build_local_source(settings),
        _build_box_source(settings, box_client) if has_box_roots else None,
    )
