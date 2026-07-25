"""
Parameter sweep with an out-of-sample split.

Run:  python sweep.py XAUUSD 60 0.30

Tests every combination of the entry/exit settings on the FIRST 60% of the
history (train), then re-runs the same settings on the untouched LAST 40%
(test) and reports both.

Why the split matters: with enough combinations, something always looks
great on a given stretch of history purely by chance. A setting that works
on data it was never tuned against is worth far more than the best-looking
number on the training half. If the two columns disagree wildly, the
"winner" is noise, not an edge.
"""

from __future__ import annotations

import itertools
import sys

from backtest import WARMUP, Params, load_market, simulate, stats
from config import load_config

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Grid. Keep it modest — more combinations means more chances to fool yourself.
THRESHOLDS = [4, 5, 6]
MIN_ADX = [15.0, 20.0, 25.0, 30.0]
SL_MULTS = [1.0, 1.5, 2.0, 2.5]
RR_RATIOS = [1.0, 1.5, 2.0, 3.0]

MIN_TRADES_PER_HALF = 25   # ignore combos too rare to judge


def main(symbol: str, days: int, spread: float | None) -> None:
    cfg = load_config()
    md = load_market(symbol, days, spread, cfg)

    n = len(md.close)
    split = WARMUP + int((n - WARMUP) * 0.60)
    combos = list(itertools.product(THRESHOLDS, MIN_ADX, SL_MULTS, RR_RATIOS))

    print(f"\nTrain bars {WARMUP}–{split} | Test bars {split}–{n}")
    print(f"Testing {len(combos)} combinations…\n")

    rows = []
    for k, (thr, adx, slm, rr) in enumerate(combos, 1):
        if k % 24 == 0:
            print(f"  …{k}/{len(combos)}")
        p = Params(signal_threshold=thr, min_adx=adx, require_momentum=True,
                   sl_atr_mult=slm, rr_ratio=rr)
        tr = stats(simulate(md, p, WARMUP, split))
        te = stats(simulate(md, p, split, n))
        if tr["trades"] < MIN_TRADES_PER_HALF or te["trades"] < MIN_TRADES_PER_HALF:
            continue
        rows.append((p, tr, te))

    if not rows:
        print("No combination produced enough trades in both halves.")
        print("Widen the grid, lengthen the history, or loosen the filters.")
        return

    # Rank by TRAIN, then look at what the untouched TEST half says.
    rows.sort(key=lambda r: r[1]["net"], reverse=True)

    print("\n" + "=" * 78)
    print("TOP 12 BY TRAIN RESULT — the TEST column is the one that matters")
    print("=" * 78)
    print(f"{'settings':<32}{'train net':>11}{'train pf':>10}"
          f"{'test net':>11}{'test pf':>9}{'test n':>7}")
    print("-" * 78)
    for p, tr, te in rows[:12]:
        print(f"{p.label():<32}{tr['net']:>+11.0f}{tr['pf']:>10.2f}"
              f"{te['net']:>+11.0f}{te['pf']:>9.2f}{te['trades']:>7}")

    # Robust = profitable on data it was never tuned against.
    robust = [r for r in rows if r[2]["net"] > 0 and r[2]["pf"] >= 1.3 and r[1]["net"] > 0]
    robust.sort(key=lambda r: r[2]["pf"], reverse=True)

    print("\n" + "=" * 78)
    print("HELD UP OUT-OF-SAMPLE (test profit factor >= 1.3 and profitable on both)")
    print("=" * 78)
    if not robust:
        print("None.")
        print()
        print("That is a real answer, not a bug: no setting in this grid shows an")
        print("edge that survives on unseen data. Tuning harder would only fit")
        print("noise. Better next steps are a different entry signal, a cheaper")
        print("instrument, or a slower timeframe where spread matters less.")
        return

    for p, tr, te in robust[:8]:
        print(f"{p.label():<32}{tr['net']:>+11.0f}{tr['pf']:>10.2f}"
              f"{te['net']:>+11.0f}{te['pf']:>9.2f}{te['trades']:>7}")

    best = robust[0][0]
    print("\nBest out-of-sample settings → put these in .env:")
    print(f"  SIGNAL_THRESHOLD={best.signal_threshold}")
    print(f"  MIN_ADX={best.min_adx:.0f}")
    print(f"  SL_ATR_MULTIPLIER={best.sl_atr_mult:g}")
    print(f"  RR_RATIO={best.rr_ratio:g}")
    print("\nEven these deserve a demo run before real money — an out-of-sample")
    print("win on one symbol over one period is evidence, not proof.")


if __name__ == "__main__":
    if mt5 is None:
        print("MetaTrader5 package not available (Windows only).")
        sys.exit(1)
    _cfg = load_config()
    _symbol = sys.argv[1] if len(sys.argv) > 1 else _cfg.symbols[0]
    _days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    _spread = float(sys.argv[3]) if len(sys.argv) > 3 else None
    main(_symbol, _days, _spread)
