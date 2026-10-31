"""Dixon-Coles (1997) bivariate Poisson model with time-decay weights.

Dixon, M. J. & Coles, S. G. (1997). Modelling Association Football Scores and
Inefficiencies in the Football Betting Market. JRSS Series C, 46(2), 265-280.

home goals ~ Poisson(lambda = exp(home + att_h - def_a))
away goals ~ Poisson(mu     = exp(att_a - def_h))
with the low-score correction tau(x, y; rho) for 0-0, 1-0, 0-1, 1-1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


@dataclass
class DixonColes:
    xi: float = 0.0019          # time-decay per day (tunable; validate on held-out rounds)
    l2: float = 0.05            # ridge penalty, keeps promoted/new teams near average early on
    teams: list[str] | None = None
    params: np.ndarray | None = None

    # ---- fitting -------------------------------------------------------
    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColes":
        as_of = as_of or matches["date"].max()
        m = matches[matches["date"] <= as_of]
        self.teams = sorted(set(m["home"]) | set(m["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        h = m["home"].map(idx).to_numpy()
        a = m["away"].map(idx).to_numpy()
        x = m["hg"].to_numpy()
        y = m["ag"].to_numpy()
        w = np.exp(-self.xi * (as_of - m["date"]).dt.days.to_numpy())
        n = len(self.teams)

        def nll(p):
            att, dfn, home, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            att = att - att.mean()                     # identifiability
            lam = np.exp(home + att[h] - dfn[a])
            mu = np.exp(att[a] - dfn[h])
            tau = np.clip(_tau(x, y, lam, mu, rho), 1e-10, None)
            ll = (np.log(tau) + x * np.log(lam) - lam - gammaln(x + 1)
                  + y * np.log(mu) - mu - gammaln(y + 1))
            return -(w * ll).sum() + self.l2 * (att @ att + dfn @ dfn)

        p0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
        bounds = [(None, None)] * (2 * n) + [(None, None), (-0.3, 0.3)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            raise RuntimeError(f"Dixon-Coles fit failed: {res.message}")
        p = res.x.copy()
        p[:n] -= p[:n].mean()
        self.params = p
        return self

    # ---- prediction ----------------------------------------------------
    def ratings(self) -> pd.DataFrame:
        n = len(self.teams)
        return (pd.DataFrame({"team": self.teams, "attack": self.params[:n],
                              "defence": self.params[n:2 * n]})
                .assign(strength=lambda d: d.attack + d.defence)
                .sort_values("strength", ascending=False, ignore_index=True))

    @property
    def home_adv(self) -> float:
        return float(self.params[2 * len(self.teams)])

    @property
    def rho(self) -> float:
        return float(self.params[2 * len(self.teams) + 1])

    def rates(self, home: str, away: str) -> tuple[float, float]:
        n, i, j = len(self.teams), self.teams.index(home), self.teams.index(away)
        att, dfn = self.params[:n], self.params[n:2 * n]
        return (float(np.exp(self.home_adv + att[i] - dfn[j])),
                float(np.exp(att[j] - dfn[i])))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.rates(home, away)
        g = np.arange(MAX_GOALS + 1)
        ph = np.exp(g * np.log(lam) - lam - gammaln(g + 1))
        pa = np.exp(g * np.log(mu) - mu - gammaln(g + 1))
        mat = np.outer(ph, pa)
        X, Y = np.meshgrid(g, g, indexing="ij")
        mat *= _tau(X, Y, np.full_like(mat, lam), np.full_like(mat, mu), self.rho)
        return mat / mat.sum()

    def outcome_probs(self, home: str, away: str) -> tuple[float, float, float]:
        mat = self.score_matrix(home, away)
        return float(np.tril(mat, -1).sum()), float(np.trace(mat)), float(np.triu(mat, 1).sum())
