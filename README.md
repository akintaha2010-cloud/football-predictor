# football-predictor

Estimates football match probabilities (home win / draw / away win, exact scores, over/under 2.5, both teams to score) from past results.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Predict a match

```
python predict.py Galatasaray Fenerbahce
python predict.py Arsenal Chelsea --league E0
python predict.py Galatasaray Fenerbahce --neutral   # neutral venue, e.g. a cup final
python predict.py --league T1 --table      # team ratings
```

The first run downloads the last 3 seasons from [football-data.co.uk](https://www.football-data.co.uk) into `data/`. Use `--refresh` to fetch new results. Leagues: `T1` Süper Lig, `E0` Premier League, `E1` Championship, `SP1` La Liga, `I1` Serie A, `D1` Bundesliga, `F1` Ligue 1, `N1` Eredivisie, `P1` Primeira Liga.

```
Galatasaray vs Fenerbahce
  Expected goals:   2.14 - 1.57

  Galatasaray win:           50.0%
  Draw:                      22.6%
  Fenerbahce win:            27.4%

  Over 2.5 goals:            71.5%
  Both teams score:          70.7%

  Most likely scores:
    1-1    9.2%
    2-1    8.8%
    2-2    6.9%
```

## How it works

`football_predictor/model.py` implements the Dixon-Coles (1997) model. Each team has an attack and a defence rating, and goals are Poisson distributed:

- home goals: `λ = exp(home_adv + attack[home] − defence[away])`
- away goals: `μ = exp(attack[away] − defence[home])`

On top of that:
- A correction `ρ` adjusts the 0-0, 1-0, 0-1 and 1-1 probabilities, which plain Poisson gets wrong.
- Older matches count less: `weight = exp(−ξ · days_ago)`, with `ξ = 0.0019`, a half-life of about one year.
- An L2 penalty pulls ratings toward average, so newly promoted teams with only a few matches don't get extreme ratings.

The ratings are fitted by maximum likelihood (L-BFGS-B with an analytic gradient, about 0.01 s). The score matrix (0–10 goals each) then gives every market.

## Backtest

`backtest.py` refits the model before each week of a season and predicts that week's matches. The first 4 weeks are skipped. Results for the 2025/26 season:

| League | Matches | Model log loss | Bookmaker log loss | Model accuracy | Bookmaker accuracy | Always pick the most common result |
|---|---|---|---|---|---|---|
| Süper Lig (T1) | 276 | 1.000 | 0.977 | 50.7% | 54.3% | 42.4% |
| Premier League (E0) | 350 | 1.025 | 1.016 | 47.1% | 49.7% | 42.0% |
| La Liga (SP1) | 349 | 0.980 | 0.963 | 52.1% | 54.7% | 49.3% |

The model is clearly better than guessing the most common result. It is a little behind bookmaker closing odds, which also price in injuries, line-ups and news.

```
python backtest.py --league E0 --season 2526
```

## Tests

```
python -m pytest
```
