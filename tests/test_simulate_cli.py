import pandas as pd

import simulate
from tests.test_model import synthetic_results


def test_simulate_saves_csv(tmp_path, monkeypatch, capsys):
    history = synthetic_results()
    current = history.iloc[: len(history) // 12]
    monkeypatch.setattr(simulate, "load_results", lambda league, seasons, **_: current if len(seasons) == 1 else history)
    out = tmp_path / "season.csv"
    assert simulate.main(["--sims", "200", "--seed", "1", "--top", "1", "--relegated", "1", "--csv", str(out)]) == 0
    table = pd.read_csv(out)
    assert sorted(table["team"]) == ["A", "B", "C", "D"]
    assert table["title"].sum() > 0.99
    assert f"Saved {out}" in capsys.readouterr().out
