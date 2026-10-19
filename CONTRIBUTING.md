# Contributing

Bug reports, ideas and pull requests are welcome.

## Getting started

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## Pull requests

- Keep each pull request to one change, and add or update tests in `tests/`.
- Tests must not need the network: use small synthetic results like `tests/test_model.py::synthetic_results`.
- If a change affects predictions, run `python backtest.py --league T1 --season 2526` (and ideally `E0`, `SP1`) before and after, and paste the log loss and accuracy in the description.
- Update the README when you add a command or option.

## Adding a league

Any league on [football-data.co.uk](https://www.football-data.co.uk/data.php) with a `mmz4281/{season}/{code}.csv` file works. Add its code and name to `LEAGUES` in `football_predictor/data.py` and to the list in the README.
