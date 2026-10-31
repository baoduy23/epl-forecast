"""Weekly job: ingest -> fit -> simulate -> write outputs.

python -m epl_forecast.run --seasons 2425 2526 2627 --current 2627
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ingest import load_history
from .model import DixonColes
from .simulate import simulate_season


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2425", "2526", "2627"])
    ap.add_argument("--current", default="2627")
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = ap.parse_args(argv)

    matches = load_history(args.seasons, args.data_dir)
    season = matches[matches["season"] == args.current]
    model = DixonColes().fit(matches)
    result = simulate_season(model, season, n_sims=args.sims)

    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    last_match = season["date"].max().strftime("%Y-%m-%d")
    result.insert(0, "run_date", run_at)
    result.insert(1, "data_through", last_match)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "latest.csv", index=False)
    model.ratings().to_csv(args.out_dir / "ratings.csv", index=False)
    hist = args.out_dir / "history.csv"
    prev = pd.read_csv(hist) if hist.exists() else pd.DataFrame()
    if not prev.empty:
        prev = prev[prev["data_through"] != last_match]       # idempotent re-runs
    pd.concat([prev, result], ignore_index=True).to_csv(hist, index=False)

    top = result.head(8)
    lines = [f"# EPL 2026-27 title odds (data through {last_match})", "",
             "| Team | Pts | Exp. pts | Title | Top 4 | Relegation |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.team} | {r.pts_now} | {r.exp_pts} | {r.p_title:.1%} | {r.p_top4:.1%} | "
              f"{r.p_relegation:.1%} |" for r in top.itertuples()]
    (args.out_dir / "latest.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
