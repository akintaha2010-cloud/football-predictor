import predict
from football_predictor.data import LEAGUES


def test_leagues_lists_every_code_without_downloading(monkeypatch, capsys):
    monkeypatch.setattr(predict, "load_results", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")))
    assert predict.main(["--leagues"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(LEAGUES)
    assert lines[0].split()[0] in LEAGUES
