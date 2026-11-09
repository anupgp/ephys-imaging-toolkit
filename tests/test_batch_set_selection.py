"""Tests for batch_analysis/set_selection.py: the ways a batch run's sets can be given."""

import pandas as pd
import pytest

from ephys_imaging_data_loader.batch_analysis import SetKey, parse_set_text, resolve_sets


class FakeCatalog:
    """Just what resolve_sets needs: select() and distinct()."""

    table = pd.DataFrame({"expdate": ["20260322", "20260322", "20270313"], "neuronid": [1, 1, 1],
                          "dendriteid": [1, 1, 1], "group": ["pre-apv", "post-apv", "ctrl"],
                          "expname": ["a", "a", "b"]})

    def select(self, expname=None, **_):
        return self.table if expname is None else self.table[self.table.expname.isin(expname)]

    def distinct(self, column, expdate, neuronid, dendriteid, expname=None):
        rows = self.select(expname)
        return sorted(rows[(rows.expdate == str(expdate)) & (rows.neuronid == neuronid)
                           & (rows.dendriteid == dendriteid)][column].unique())


def test_parse_text():
    assert parse_set_text("20260322 1 1") == SetKey("20260322", 1, 1, None)
    assert parse_set_text("20260322,1,1,post-apv") == SetKey("20260322", 1, 1, "post-apv")
    with pytest.raises(ValueError):
        parse_set_text("20260322 1")


def test_resolve_forms(tmp_path):
    catalog = FakeCatalog()
    assert [k.group for k in resolve_sets(catalog, (20260322, 1, 1))] == ["post-apv", "pre-apv"]
    assert resolve_sets(catalog, [(20260322, 1, 1, "pre-apv"), "20270313 1 1"]) == [
        SetKey("20260322", 1, 1, "pre-apv"), SetKey("20270313", 1, 1, "ctrl")]
    path = tmp_path / "sets.txt"
    path.write_text("# my sets\n20270313,1,1\n\n20260322 1 1 pre-apv   # comment\n")
    assert [k.name for k in resolve_sets(catalog, str(path))] == ["20270313_1_1_ctrl", "20260322_1_1_pre-apv"]
    assert len(resolve_sets(catalog, None)) == 3
    assert len(resolve_sets(catalog, None, expnames=["b"])) == 1
    with pytest.raises(LookupError):
        resolve_sets(catalog, (20990101, 1, 1))
