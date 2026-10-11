"""Track a real or paper equity account against a benchmark.

    python scripts/track_portfolio.py --trades trades.csv --cashflows cashflows.csv \
        --prices prices.csv --benchmark SPY --out reports/my_account

trades.csv    : date,ticker,quantity,price,fees   (quantity > 0 buy, < 0 sell)
cashflows.csv : date,amount                       (deposits > 0, withdrawals < 0)
prices.csv    : date,<TICKER1>,<TICKER2>,...,<BENCHMARK>   daily closes (e.g. a broker or yfinance export)

Writes report.json (NAV, P&L, TWR, CAGR, vol, Sharpe, Sortino, max drawdown, beta, alpha,
tracking error, information ratio vs the benchmark), pnl_by_position.csv, valuation.csv and
equity / drawdown charts.

``--demo`` replays the frozen strategy's paper ledger from results/final/ instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela import plots  # noqa: E402
from vela.tracker.ledger import Ledger, performance_report, time_weighted_returns  # noqa: E402


def run(ledger: Ledger, prices: pd.DataFrame, benchmark: str | None, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bench = prices[benchmark] if benchmark and benchmark in prices else None
    val = ledger.valuation(prices.drop(columns=[benchmark], errors="ignore") if benchmark else prices)
    rep = performance_report(val, bench)
    val.to_csv(out / "valuation.csv")
    ledger.pnl_by_position(prices).to_csv(out / "pnl_by_position.csv")
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    r = time_weighted_returns(val)
    r = r.loc[rep["start"]:]
    series = {"Account (time-weighted)": r}
    colors = {"Account (time-weighted)": plots.ROLE["strategy"]}
    if bench is not None:
        series[benchmark] = bench.reindex(r.index).ffill().pct_change().fillna(0.0)
        colors[benchmark] = plots.ROLE["SPY"]
    plots.equity_curves(series, out / "equity.png", "Account vs benchmark", colors)
    plots.drawdowns(series, out / "drawdown.png", "Drawdowns", colors)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades")
    ap.add_argument("--cashflows")
    ap.add_argument("--prices")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--out", default="reports/account")
    ap.add_argument("--demo", action="store_true", help="track the holdout paper-trading ledger")
    args = ap.parse_args()

    if args.demo:
        root = Path(__file__).resolve().parents[1]
        from vela.research import ResearchData
        d = ResearchData.load("v1")
        folder = next((root / "results" / "final").glob("*_paper_ledger"))
        trades = folder / "trades.csv.gz" if (folder / "trades.csv.gz").exists() else folder / "trades.csv"
        ledger = Ledger.from_csv(trades, folder / "cashflows.csv")
        start = ledger.cashflows["date"].min()
        tick = sorted(ledger.trades["ticker"].unique())
        prices = pd.DataFrame(d.panel.close, index=d.dates, columns=d.panel.tickers)[tick].loc[start:]
        prices["SPY"] = d.panel.benchmarks["SPY"].loc[start:]
        out = root / "reports" / "paper_account"
    else:
        if not (args.trades and args.cashflows and args.prices):
            ap.error("--trades, --cashflows and --prices are required (or use --demo)")
        ledger = Ledger.from_csv(args.trades, args.cashflows)
        prices = pd.read_csv(args.prices, parse_dates=["date"], index_col="date").sort_index()
        out = Path(args.out)
    rep = run(ledger, prices, args.benchmark, out)
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
