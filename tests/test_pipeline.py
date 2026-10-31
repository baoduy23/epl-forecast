"""Synthetic-league tests: the model must recover known team strengths, and the
simulation's probabilities must be internally consistent."""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from epl_forecast.model import DixonColes
from epl_forecast.simulate import remaining_fixtures, simulate_season

TEAMS = [f"T{i:02d}" for i in range(20)]
TRUE_STRENGTH = np.linspace(0.6, -0.6, 20)   # T00 strongest


def make_league(seed=1, played_current=100):
    rng = np.random.default_rng(seed)
    rows, start = [], pd.Timestamp("2024-08-15")
    for s, season in enumerate(["2425", "2526", "2627"]):
        pairs = list(itertools.permutations(range(20), 2))
        rng.shuffle(pairs)
        if season == "2627":
            pairs = pairs[:played_current]
        for k, (i, j) in enumerate(pairs):
            lam = np.exp(0.25 + 0.5 * TRUE_STRENGTH[i] - 0.5 * TRUE_STRENGTH[j])
            mu = np.exp(0.5 * TRUE_STRENGTH[j] - 0.5 * TRUE_STRENGTH[i])
            rows.append({"date": start + pd.Timedelta(days=365 * s + k // 10 * 7),
                         "home": TEAMS[i], "away": TEAMS[j],
                         "hg": rng.poisson(lam), "ag": rng.poisson(mu), "season": season})
    return pd.DataFrame(rows)


def test_model_recovers_strength_order():
    df = make_league()
    model = DixonColes().fit(df)
    r = model.ratings().set_index("team").loc[TEAMS, "strength"]
    assert spearmanr(r, TRUE_STRENGTH).statistic > 0.85
    assert 0.1 < model.home_adv < 0.4


def test_outcome_probs_sum_to_one():
    model = DixonColes().fit(make_league())
    assert abs(sum(model.outcome_probs("T00", "T19")) - 1) < 1e-9
    hw, _, aw = model.outcome_probs("T00", "T19")
    assert hw > aw


def test_remaining_fixtures_count():
    season = make_league()[lambda d: d.season == "2627"]
    assert len(remaining_fixtures(season, TEAMS)) == 380 - 100


def test_simulation_probabilities_consistent():
    df = make_league()
    model = DixonColes().fit(df)
    res = simulate_season(model, df[df.season == "2627"], n_sims=2000)
    assert abs(res.p_title.sum() - 1) < 1e-9
    assert abs(res.p_top4.sum() - 4) < 1e-9
    assert abs(res.p_relegation.sum() - 3) < 1e-9
    assert res.iloc[0].team in {"T00", "T01", "T02", "T03"}
