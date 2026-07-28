"""
Regression guard for ict.py. Needs only numpy — no MT5, so it runs anywhere.

Run:  python test_ict.py

It exists because this project has already shipped two backtesters that
manufactured profit out of nothing (intrabar lookahead in the trailing stop,
and under-fetched higher timeframes). The two tests that matter here:

  * PREFIX EQUALITY — signals computed over bars[0:k] must be byte-identical to
    signals computed over the whole series and then truncated at k. If any
    future bar leaks into a decision, this fails.
  * NULL HYPOTHESIS — on a driftless random walk the model must LOSE roughly
    the spread. A random walk has no edge to find, so a positive expectancy
    there is proof of a bug, not a discovery.
"""

import sys
import types

import numpy as np

# Stub the Windows-only imports so the pure logic can be exercised anywhere.
for _name, _attrs in (("config", {"AppConfig": object, "load_config": lambda: None}),
                      ("mt5_connector", {"MT5Connector": object})):
    _m = types.ModuleType(_name)
    _m.__dict__.update(_attrs)
    sys.modules.setdefault(_name, _m)

from dataclasses import replace  # noqa: E402

from ict import IctParams, Market, _atr, compute_signals, simulate, stats  # noqa: E402


def _mk(o, h, l, c, spread=0.30):
    t = np.datetime64("2025-01-01") + np.arange(len(c)) * np.timedelta64(15, "m")
    return Market("TEST", 15, spread, 10.0, 0.1, t, o, h, l, c, _atr(h, l, c))


def trending(n=4000, seed=0):
    """Random walk with drift regimes, so sweeps and reversals actually occur."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.35, n // 50 + 1), 50)[:n]
    c = 2000 + np.cumsum(rng.normal(0, 1.0, n) + drift)
    o = np.empty(n)
    o[0] = c[0]
    o[1:] = c[:-1] + rng.normal(0, 0.3, n - 1)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.8, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.8, n))
    return _mk(o, h, l, c)


def driftless(n=6000, seed=0):
    """Zero drift. There is no edge here to find."""
    rng = np.random.default_rng(seed)
    c = 2000 + np.cumsum(rng.normal(0, 1.0, n))
    o = np.empty(n)
    o[0] = c[0]
    o[1:] = c[:-1]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.8, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.8, n))
    return _mk(o, h, l, c)


def main() -> int:
    m, p = trending(), IctParams()
    fails: list[str] = []
    side, sl_a, kind_a, pool_a, rng_a = compute_signals(m, p)
    print(f"signals on full series : {int((side != 0).sum())}")

    # 1. No lookahead. A prefix must decide exactly what the full series decided.
    for cut in (1500, 2500, 3300):
        pre = _mk(m.o[:cut], m.h[:cut], m.l[:cut], m.c[:cut])
        s2, sl2, _, _, _ = compute_signals(pre, p)
        if not np.array_equal(side[:cut], s2):
            bad = np.where(side[:cut] != s2)[0][:10]
            fails.append(f"LOOKAHEAD: prefix {cut} differs at bars {bad}")
        elif not np.allclose(np.nan_to_num(sl_a[:cut]), np.nan_to_num(sl2)):
            fails.append(f"LOOKAHEAD: stop levels differ at prefix {cut}")

    trades = simulate(m, p)

    # 2. Fills land on the NEXT bar's open, never the signal bar's close.
    if trades:
        first = int(np.where(side != 0)[0][0])
        if not np.isclose(trades[0].entry, m.o[first + 1]):
            fails.append(f"FILL: entry {trades[0].entry} != next open "
                         f"{m.o[first + 1]} (signal close {m.c[first]})")

    # 3. R accounting agrees with the dollar figure and carries the spread.
    for t in trades:
        risk = abs(t.entry - t.sl)
        want = (abs(t.tp - t.entry) / risk if t.reason == "TP" else -1.0)
        want -= m.spread / risk
        if not np.isclose(t.r, want, atol=1e-6):
            fails.append(f"R MISMATCH: {t.reason} got {t.r:+.4f} want {want:+.4f}")
            break
        if (t.r > 0) != (t.profit_usd > 0):
            fails.append(f"SIGN: r={t.r:+.3f} usd={t.profit_usd:+.2f}")
            break
        if t.bars_held < 1:
            fails.append("HOLD: trade closed on its own entry bar")
            break

    # 4. The target sits at rr_mult x the REAL risk taken from the fill.
    for t in trades[:50]:
        risk = abs(t.entry - t.sl)
        want = t.entry + risk * p.rr_mult if t.is_buy else t.entry - risk * p.rr_mult
        if not np.isclose(t.tp, want):
            fails.append(f"TP GEOMETRY: {t.tp} != {want}")
            break

    # 5. The sweep's cache must not change any result.
    if stats(simulate(m, p, None, 0, 2400)) != stats(
            simulate(m, p, (side, sl_a, kind_a, pool_a, rng_a), 0, 2400)):
        fails.append("CACHE: precomputed signals gave a different result")

    # 6. ...and its key is only valid if rr_mult cannot move a signal.
    a = compute_signals(m, replace(p, rr_mult=1.5))[0]
    b = compute_signals(m, replace(p, tp_mode="pool", rr_mult=3.0))[0]
    if not np.array_equal(a, b):
        fails.append("CACHE KEY: rr_mult/tp_mode changed which bars signal")

    # 7. Structural targets must sit at the level they claim, and never be
    #    taken when the chart offers less room than min_rr.
    for tpm, level in (("pool", pool_a), ("range", rng_a)):
        q = replace(p, tp_mode=tpm, min_rr=1.0)
        for t in simulate(m, q)[:50]:
            risk = abs(t.entry - t.sl)
            cap = t.entry + (risk if t.is_buy else -risk) * q.max_struct_rr
            want = min(level[t.bar], cap) if t.is_buy else max(level[t.bar], cap)
            if not np.isclose(t.tp, want):
                fails.append(f"{tpm.upper()} TP: {t.tp} != {want} "
                             f"(level {level[t.bar]}, cap {cap})")
                break
            if abs(t.tp - t.entry) / risk > q.max_struct_rr + 1e-9:
                fails.append(f"{tpm.upper()} MAX_RR: target beyond the cap")
                break
            if not np.isclose(t.entry, m.o[t.bar + 1]):
                fails.append(f"{tpm.upper()} FILL: entry != next open")
                break
            if abs(t.tp - t.entry) / abs(t.entry - t.sl) < q.min_rr - 1e-9:
                fails.append(f"{tpm.upper()} MIN_RR: took a trade below min_rr")
                break

    # 7b. No stop may sit closer than the risk floor, and the floor must never
    #     override the ceiling.
    for mr in (0.5, 1.0):
        q = replace(p, min_risk_atr=mr)
        sd, sl2, _, _, _ = compute_signals(m, q)
        idx = np.where(sd != 0)[0]
        dist = np.abs(m.c[idx] - sl2[idx]) / m.atr[idx]
        lo = min(mr, q.max_risk_atr)
        if len(idx) and dist.min() < lo - 1e-9:
            fails.append(f"RISK FLOOR: stop at {dist.min():.3f} ATR, floor {lo}")
        if len(idx) and dist.max() > q.max_risk_atr + 1e-9:
            fails.append(f"RISK CEILING: stop at {dist.max():.3f} ATR, cap {q.max_risk_atr}")

    # 7c. Both reversal filters may only ever REMOVE trades, never add one,
    #     and neither may touch a continuation.
    base_rev = replace(p, trade_continuation=False)
    n0 = len(simulate(m, base_rev))
    for tag, q in (("trend", replace(base_rev, trend_ema=200)),
                   ("poke", replace(base_rev, sweep_max_atr=0.5))):
        if len(simulate(m, q)) > n0:
            fails.append(f"{tag.upper()} FILTER: added reversal trades ({n0} -> {len(simulate(m, q))})")
    cont = replace(p, trade_reversal=False)
    if len(simulate(m, cont)) != len(simulate(m, replace(cont, trend_ema=200))):
        fails.append("TREND FILTER: changed continuation trades, should not")

    # 8b. A tighter R cap can only shrink targets, never grow them.
    wide = {t.bar: t.tp for t in simulate(m, replace(p, tp_mode="pool", max_struct_rr=20))}
    for t in simulate(m, replace(p, tp_mode="pool", max_struct_rr=3)):
        if t.bar in wide:
            far = wide[t.bar]
            if (t.tp > far + 1e-9) if t.is_buy else (t.tp < far - 1e-9):
                fails.append("MAX_RR MONOTONICITY: tighter cap gave a further target")
                break

    # 8. A higher min_rr can only ever remove trades, never add them.
    counts = [len(simulate(m, replace(p, tp_mode="pool", min_rr=r)))
              for r in (0.5, 1.0, 2.0, 3.0)]
    if counts != sorted(counts, reverse=True):
        fails.append(f"MIN_RR MONOTONICITY: trade counts {counts} not decreasing")

    s = stats(trades)
    print(f"trending sample        : {s['n']} trades, win {s['wr']:.1f}%, "
          f"avg {s['avg_r']:+.3f}R, pf {s['pf']:.2f}")
    for tpm in ("pool", "range"):
        k = stats(simulate(m, replace(p, tp_mode=tpm)))
        print(f"  tp={tpm:<18}: {k['n']} trades, win {k['wr']:.1f}%, "
              f"avg {k['avg_r']:+.3f}R, pf {k['pf']:.2f}")

    # 9. Null hypothesis, every target mode. No drift means no edge, so a
    #    positive expectancy on noise is a leak no matter how the TP is chosen.
    for tpm in ("atr", "pool", "range"):
        q = replace(p, tp_mode=tpm)
        acc: list = []
        for seed in range(12):
            acc += simulate(driftless(seed=seed), q)
        ns = stats(acc)
        print(f"noise tp={tpm:<14}: {ns['n']} trades, win {ns['wr']:.1f}%, "
              f"avg {ns['avg_r']:+.3f}R, pf {ns['pf']:.2f}")
        if ns["avg_r"] > 0:
            fails.append(f"NULL HYPOTHESIS ({tpm}): {ns['avg_r']:+.3f}R on pure "
                         "noise — the backtester is inventing profit")

    print()
    if fails:
        print("FAILED:\n  " + "\n  ".join(fails))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
