"""Tests of the Box credentials/token helpers (no network access)."""

import json
import stat

import pytest

from ephys_imaging_data_loader.data_sources.box_login_and_token_storage import (
    _make_token_storage_class,
    choose_token_folder,
    read_credentials_file,
)


def test_read_credentials_file(tmp_path):
    credentials = tmp_path / ".box_credentials"
    credentials.write_text("# comment\nCLIENT_ID='abc'\nCLIENT_SECRET=\"def\"\n\nBOX_OAUTH_LOC=/x\n")
    values = read_credentials_file(credentials)
    assert values == {"CLIENT_ID": "abc", "CLIENT_SECRET": "def", "BOX_OAUTH_LOC": "/x"}


def test_credentials_must_contain_id_and_secret(tmp_path):
    credentials = tmp_path / ".box_credentials"
    credentials.write_text("CLIENT_ID=abc\n")
    with pytest.raises(ValueError, match="CLIENT_SECRET"):
        read_credentials_file(credentials)


def test_token_folder_falls_back_when_path_is_from_another_computer(tmp_path):
    credentials_file = tmp_path / "box" / ".box_credentials"
    credentials_file.parent.mkdir()
    # BOX_OAUTH_LOC points to a folder that does not exist on this machine
    folder = choose_token_folder(credentials_file, {"BOX_OAUTH_LOC": "/home/someone-else/OAuth2/box/"})
    assert folder == credentials_file.parent
    assert choose_token_folder(credentials_file, {}, token_folder=tmp_path) == tmp_path


def test_token_storage_is_compatible_atomic_and_private(tmp_path):
    pytest.importorskip("box_sdk_gen")
    from box_sdk_gen import AccessToken

    storage = _make_token_storage_class()(tmp_path / "oauth_box_auth.json")
    assert storage.get() is None
    storage.store(AccessToken(access_token="A1", refresh_token="R1"))

    saved = json.loads((tmp_path / "oauth_box_auth.json").read_text())
    assert saved == {"access_token": "A1", "refresh_token": "R1"}  # same format as the old helper
    mode = stat.S_IMODE((tmp_path / "oauth_box_auth.json").stat().st_mode)
    assert mode == 0o600
    assert not list(tmp_path.glob(".box_token_*"))  # no temporary files left behind
    assert storage.get().refresh_token == "R1"
