"""Monte Carlo simulation of the rest of the season.

EPL is a double round-robin, so remaining fixtures = every ordered (home, away)
pair of this season's 20 clubs that has not been played yet. No fixture API needed
(dates don't matter for the final table).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .model import MAX_GOALS, DixonColes


def remaining_fixtures(season_matches: pd.DataFrame, teams: list[str]) -> list[tuple[str, str]]:
    played = set(zip(season_matches["home"], season_matches["away"]))
    return [(h, a) for h, a in itertools.permutations(teams, 2) if (h, a) not in played]


def current_table(season_matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    t = pd.DataFrame(0, index=teams, columns=["P", "Pts", "GF", "GA"])
    for r in season_matches.itertuples():
        hp, ap = (3, 0) if r.hg > r.ag else (1, 1) if r.hg == r.ag else (0, 3)
        t.loc[r.home, ["P", "Pts", "GF", "GA"]] += [1, hp, r.hg, r.ag]
        t.loc[r.away, ["P", "Pts", "GF", "GA"]] += [1, ap, r.ag, r.hg]
    t["GD"] = t["GF"] - t["GA"]
    return t.sort_values(["Pts", "GD", "GF"], ascending=False)


def simulate_season(model: DixonColes, season_matches: pd.DataFrame,
                    n_sims: int = 10_000, seed: int = 0) -> pd.DataFrame:
    teams = sorted(set(season_matches["home"]) | set(season_matches["away"]))
    base = current_table(season_matches, teams).loc[teams]
    fixtures = remaining_fixtures(season_matches, teams)
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    pts = np.tile(base["Pts"].to_numpy(float), (n_sims, 1))
    gf = np.tile(base["GF"].to_numpy(float), (n_sims, 1))
    ga = np.tile(base["GA"].to_numpy(float), (n_sims, 1))
    side = MAX_GOALS + 1

    for h, a in fixtures:
        cdf = np.cumsum(model.score_matrix(h, a).ravel())
        draw = np.minimum(np.searchsorted(cdf, rng.random(n_sims)), side * side - 1)
        hg, ag = draw // side, draw % side
        i, j = idx[h], idx[a]
        gf[:, i] += hg; ga[:, i] += ag; gf[:, j] += ag; ga[:, j] += hg
        pts[:, i] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, j] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))

    gd = gf - ga
    # Rank: points, then goal difference, then goals for; random jitter breaks exact ties.
    key = pts * 1e8 + (gd + 1000) * 1e4 + gf + rng.random(pts.shape) * 0.5
    pos = (-key).argsort(axis=1).argsort(axis=1) + 1          # 1 = champion

    out = pd.DataFrame({
        "team": teams,
        "pts_now": base["Pts"].to_numpy(int),
        "played": base["P"].to_numpy(int),
        "exp_pts": pts.mean(axis=0).round(1),
        "p_title": (pos == 1).mean(axis=0),
        "p_top4": (pos <= 4).mean(axis=0),
        "p_relegation": (pos >= n - 2).mean(axis=0),
        "exp_pos": pos.mean(axis=0).round(2),
    })
    return out.sort_values(["p_title", "exp_pts"], ascending=False, ignore_index=True)
