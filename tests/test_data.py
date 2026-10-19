import os
import time
import urllib.error

import pytest

from football_predictor import data


@pytest.fixture
def fetches(monkeypatch):
    calls = []

    def fake_fetch(url, path):
        calls.append(url)
        path.write_text("new")
        return path

    monkeypatch.setattr(data, "_fetch", fake_fetch)
    return calls


def age(path, hours):
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_fresh_copy_is_reused(tmp_path, fetches):
    path = tmp_path / "T1_2627.csv"
    path.write_text("old")
    data.download("T1", "2627", tmp_path, max_age_hours=12)
    assert fetches == []


def test_stale_copy_is_refreshed(tmp_path, fetches):
    path = tmp_path / "T1_2627.csv"
    path.write_text("old")
    age(path, 13)
    data.download("T1", "2627", tmp_path, max_age_hours=12)
    assert len(fetches) == 1
    assert path.read_text() == "new"


def test_finished_season_is_never_refreshed_by_age(tmp_path, fetches):
    path = tmp_path / "T1_2425.csv"
    path.write_text("old")
    age(path, 1000)
    data.download("T1", "2425", tmp_path)
    assert fetches == []


def test_failed_refresh_falls_back_to_cache(tmp_path, monkeypatch):
    path = tmp_path / "T1_2627.csv"
    path.write_text("old")

    def offline(url, path):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(data, "_fetch", offline)
    with pytest.warns(UserWarning, match="cached copy"):
        assert data.download("T1", "2627", tmp_path, force=True) == path
    assert path.read_text() == "old"
