# MT5 Telegram Trading Bot

Automated trading bot: MetaTrader 5 + Telegram alerts. Runs on the user's
Windows PC (MetaTrader5 package is Windows-only — it is a COM bridge to a
running MT5 terminal, so the bot process must be on the same machine).

Branch: `claude/mt5-telegram-trading-bot-rxcae1`

## Layout

| File | Role |
|---|---|
| `config.py` | Loads `.env` into typed dataclasses. Single source of settings. |
| `mt5_connector.py` | Only module importing MetaTrader5. Orders, positions, OHLCV. |
| `analyzer.py` | Pure TA: EMA/RSI/MACD/ATR/ADX, S/R, signal voting. No I/O. |
| `risk_manager.py` | Lot sizing, SL/TP calculation. |
| `telegram_bot.py` | Async command handlers + outbound alerts. |
| `bot.py` | Orchestrator: APScheduler scan cycle + position monitor. |
| `backtest.py` | Replays current `.env` settings over history. **Use before any live change.** Also the engine (`load_market`/`simulate`) used by the sweep. |
| `sweep.py` | Grid search with a 60/40 train/test split. Judge settings by the test column. |
| `ict.py` | Backtest for the liquidity-sweep/CISD/IFVG model (the Pine indicator), with its own `--sweep`. Selectable targets (`atr`/`pool`/`range`) and an ADR impact filter. Not wired into the live bot. |
| `test_ict.py` | Regression guard for `ict.py`. numpy only, runs anywhere. |
| `tradingview/*.pine` | Companion TradingView indicator (separate from the bot). |

## Modes (mutually exclusive, set in `.env`)

- `BALANCED_MODE` — H1/M15/**M5** trigger. The recommended one.
- `SCALP_MODE` — M15/M5/**M1**. High spread cost.
- `TURBO_MODE` — M1 on the **forming** bar, seconds-scale. Bleeds money on spread; demo experiments only.
- Default (all false) — D1/H4/H1/**M15** swing.

Timeframe slots in `analyzer.analyze()` are generic; each mode feeds
different timeframes into the same `d1/h4/h1/m15` parameters.

## Measured findings (do not re-guess these)

All M5 (`BALANCED_MODE`), 30–44 days, spread charged per trade.

| Run | Result |
|---|---|
| BTCUSD, thr 4/7, ADX≥18, SL 1.0×ATR | 976 trades, 33.5% win, PF 0.82, **−$1,282** |
| BTCUSD, thr 5/7, ADX≥25, SL 1.5×ATR | 426 trades, 51.6% win, PF 0.95, **−$161** |
| XAUUSD, thr 5/7, ADX≥25, SL 1.5×ATR, no trail | 270 trades, 37.4% win, PF **1.08**, +$2,547 |

- **Spread dominates.** In the first run it was 91% of the loss (976 × $1.20).
  Trade count is the main cost lever.
- **XAUUSD PF 1.08 is not tradeable.** Breakeven win rate is 35.6% vs 37.4%
  actual, and max drawdown ($2,159) nearly equals net profit. Real slippage
  and variable spread would erase it.
- Spreads: **BTCUSD ~$12**, XAUUSD ~$0.25–0.30 (weekend ticks read ~$1.00 —
  always pass an explicit spread when the market is shut).
- Gold M5 median ATR **$4.67**; at 0.2 lot a 1.5×ATR stop is **$140**.

### The ICT model measured: no edge (XAUUSD M15, 0.2 lot, spread $0.30)

Ran the per-timeframe preset over 273 days, and a 540-combination sweep with a
60/40 split over 365 days.

| Run | Result |
|---|---|
| Preset `piv10 arm12 run2 tp:pool rev+cont adr<1.4` | 300 trades, 25.3% win, PF 0.94, **−$3,429**, max DD **−$8,540** |
| ...of which REVERSAL | 183 trades, 24.6% win, avg −0.16R, **−$6,106** |
| ...of which CONTINUATION | 117 trades, 26.5% win, avg +0.04R, **+$2,677** |

**8 of 540 combinations "held up out-of-sample" — that is what chance produces.**
The combinations overlap heavily so the effective number of independent tests is
far below 540, but 8 survivors is still within the range noise generates. They
also disagree with each other (`piv5` and `piv12`, `rev` and `cont` and
`rev+cont` all appear), and a genuine edge shows a coherent neighbourhood of
nearby settings rather than scattered singletons. **Do not trade this model on
the strength of that list.**

Two patterns are consistent enough to be worth a focused follow-up, with a small
grid so the multiple-comparisons problem shrinks:

1. **`arm6` appears in all 12 top-by-train rows and all 8 survivors** (grid was
   6/12/20). A short arming window means the CISD must follow the sweep
   promptly, which is mechanically sensible — a stale sweep is not a trap.
2. **Continuation beats reversal**, in the baseline split above and in 10 of the
   12 top-by-train rows. The classic ICT premise is the reversal; here it is the
   one losing money.

**The structural targets did not survive.** `tp:pool` and `tp:range` fill the
top-by-train table and then vanish from the out-of-sample list, which is
textbook overfitting — they fit history better and generalised worse. The plain
`tp:atr` × 3 is what survived. The reasoning for structural targets was sound
and the data still rejected it; the 60/40 split is what caught it.

Spread was **53%** of the baseline loss (300 × $6.00 = $1,800 of $3,429), which
is the same story as the BTCUSD M5 run.

### Two backtester bugs that produced fake profits — do not reintroduce

1. **Intrabar lookahead in the trailing stop.** Raising the peak from a bar's
   favourable extreme and then triggering the exit against the same bar's
   adverse extreme assumes an unknowable intrabar order. It reported 84% win
   rate, 5-minute holds and **+$25,316** on XAUUSD; the true figure with the
   trail disabled was +$2,547. The floor must come from the peak as of the
   previous bar's close.
2. **O(n²) recomputation.** Calling `analyze()` per bar recomputed every
   indicator over the whole series. Indicators are backward-looking, so
   compute once up front.
3. **Under-fetched higher timeframes.** `df_mid`/`df_macro` were hardcoded to
   5,000 bars while the trigger series scaled with the requested period. At M15
   that capped the tradeable window at ~52 days, so `backtest.py XAUUSD 150`
   returned byte-identical results to `XAUUSD 60` while reporting "224.6 days".
   Every timeframe must be sized from the requested period, and the report must
   measure the window from the first bar with usable context
   (`MarketData.first_valid`), not the whole fetch.

### Two tests that catch a lying backtester

`test_ict.py` encodes the general defence against the bugs above. Both apply to
`backtest.py` just as much as to `ict.py`:

1. **Prefix equality.** Signals computed over `bars[0:k]` must be identical to
   signals computed over the whole series and truncated at `k`. Any future bar
   leaking into a decision breaks this. It is the cheapest possible lookahead
   detector — it would have caught the trailing-stop bug immediately.
2. **Null hypothesis.** On a driftless random walk the model must *lose* about
   the spread. There is no edge in noise, so a positive expectancy there is
   proof of a bug, never a discovery. `ict.py` scores −0.34R on noise; the
   29.1% win rate against a theoretical 33.3% for a 2R target is the
   stop-before-target rule biasing pessimistic, which is the correct direction
   to be wrong in.

Fills must always be the **next bar's open** (`backtest.py:315`). The signal is
decided on a bar's close, which is a price you cannot transact at.

### Live bot ran a different strategy than the backtest

`analyze_symbol()` fetched exactly 200 M15 bars while signals read `iloc[-2]`.
A 200-period SMA needs 201 bars to be defined there, so `sma200` was NaN,
`above_sma200` resolved to `None`, and the condition could satisfy neither BUY
nor SELL — the live bot silently scored 5-of-**6** while the backtest scored
5-of-7. Bar counts are now derived from `sma_200 + 150` (min 400), and
`analyze()` logs a warning if `sma200` is ever NaN on the signal bar.

### R is the wrong yardstick when lots are fixed

Every mode here trades a **fixed** lot, so a trade with a wide stop risks more
dollars than one with a tight stop. Average R weights those equally; the account
does not. A structural target can therefore show avg **−0.11R** and profit factor
**1.30** at the same time — both true, measuring different things. `stats()`
reports `per` (dollars per trade) alongside `avg_r`, and `ict.py --sweep`
requires a setting to be positive in **both** before calling it robust. Only
switch to R as the primary measure if lot size ever becomes risk-scaled.

### Why the fast timeframes keep losing — two independent reasons

Nothing on M1/M5 has ever survived out-of-sample here, and the swing sweep
(D1/H4/H1→M15) is the only family that has. Two separate mechanisms, both
working the same direction:

1. **Spread is fixed in dollars; the move is not.** Gold spread ~$0.30 against
   a 1.5×ATR stop is roughly **18% of risk on M1, 10% on M5, 2.5% on M15, 1.2%
   on H1**. Faster timeframes also take more trades, so the toll is paid more
   often. Both factors compound.
2. **Signal-to-noise improves as √T.** Directional movement grows with time,
   random movement with the square root of time, so the informative fraction of
   a bar rises as √(timeframe). M15 is ~3.9× cleaner than M1 for free.

The counterweight: **XAUUSD minimum lot is 0.1 and cannot go lower**, so dollar
risk rises with stop distance — about $17 on M1, $121 on M15, $243 on H1. H1 has
the best cost ratio but risks a lot per trade at a lot size that cannot be cut.
**M15 is the floor and probably the sweet spot** for this account.

### On-chart R without spread is not a result

The Pine results table scored R with no cost until the spread input was added.
On XAUUSD Loose that read **M15 −0.02R, M5 −0.03R, M1 −0.22R per trade** — the
first two look like breakeven and are not. Spread as a share of R scales with
how tight the stop is, so on M15 (risk ~$2–4) it is roughly 0.1R per trade and
on M1 (risk ~$0.8) roughly 0.35R. That gap is the whole difference between the
chart reading "nearly flat" and `ict.py` reporting −$3,429.

Derived from the same three tables (avg winner from `netR = w·avgwin − l`):

| Chart | n | win | avg winner | breakeven win | gap | Stopped, then TP |
|---|---|---|---|---|---|---|
| M15 Loose | 258 | 26.7% | 2.65R | 27.4% | −0.6pp | 19.0% |
| M5 Loose | 129 | 27.9% | 2.47R | 28.8% | −0.9pp | 23.7% |
| M1 Loose | 78 | 23.1% | 2.38R | 29.6% | **−6.5pp** | 30.0% |

Always read `avg` as `Net R ÷ Trades` rather than trusting a glance at the cell.

### Measuring whether a stop is too tight

"The read was right but I got stopped" is testable, not a feeling. After every
stop-out the indicator keeps watching the original target for `watchBars` and
counts how often price gets there anyway — the **"Stopped, then TP"** row. Under
~25% the stops are roughly right; over ~40% they are sitting in the noise and
`stopRoom` should go up. It is the before/after number for any stop change,
because widening a stop always raises the win rate on its own and that alone
proves nothing.

### ...and a structural stop needs a floor, not just a ceiling

The mirror image, found on the same M5 gold chart. A **continuation** entry is
taken the moment price closes beyond the swept level, so entry sits a hair from
its own anchor: a SELL at 4082.59 against a swept low near 4084.0 gave **$1.65
of risk**. `max_risk_atr` is a ceiling — it only ever tightens the stop — so
nothing stopped it landing inside one candle of noise. `min_risk_atr`
(default 0.5) is the floor, clamped by the ceiling so the two cannot fight.

Reversals do not have this problem: the anchor is the sweep wick, which is by
definition some distance from the entry.

### A structural target needs a ceiling, not just a floor

Pools are consumed once swept, so the nearest *unswept* pool can be a whole
day's range away. On M1 gold that produced a **SELL with a $1.65 stop and a $31
target — 18.8R**. Risk was capped at `max_risk_atr` while reward was left
unbounded, so the stop sat inside one bar of noise and the target needed hours
to days. They resolve on different clocks and the stop always wins first.
`max_struct_rr` (default 5) clamps it. `min_rr` alone is not enough — it only
guards the near side.

Dollar-based `TRAIL_ACTIVATE_USD` does not transfer across symbols: at $12 it
arms after a $0.60 move on XAUUSD 0.2 lot (inside bar noise) but needs a $120
move on BTCUSD 0.1 lot. Make it ATR-relative if it is revived.

Earlier live lesson: setting both `MAX_LOSS_USD` and an ATR stop makes the
tighter one always win, silently breaking the intended risk:reward. Use one
exit plan, not competing ones.

## Broker specifics (Axi demo)

- Positions come back **lowercase** (`btcusd`) from `positions_get` while config
  uses uppercase — always compare with `.upper()`.
- Minimum lots: XAUUSD 0.1, BTCUSD 0.3.
- `trade_stops_level` enforces a minimum SL/TP distance; very tight dollar stops
  get rejected or widened, which is why the loss cap is enforced in the monitor
  rather than only as an order-level SL.
- Orders fail with retcode **10027** when the terminal's **Algo Trading** button
  is off. Not a code bug.

## Conventions

- `.env` holds live credentials and is gitignored — never commit it, never put
  credentials in `.env.example`, `setup.bat`, or docs.
- All MT5 calls are synchronous; `bot.py` wraps them in `asyncio.to_thread`
  serialized by `_mt5_lock`.
- Signals use the last **completed** bar (`iloc[-2]`); only turbo reads the
  forming bar.
- Indicators are all backward-looking, so `backtest.py` computes them once over
  the series — mathematically identical to per-bar recomputation, far faster.

## CURRENT FOCUS: the TradingView strategy only

The bot is **paused**. Do not change `bot.py`, `config.py`, `analyzer.py`,
`risk_manager.py`, `telegram_bot.py` or the `.env` unless the user says the
exact words **"lets go back to the bot"**. Work happens in
`tradingview/liq_ifvg_cisd.pine`. `ict.py` stays available for measuring the
Pine model — that is strategy work, not bot work.

The user trades the indicator by hand and does **not** follow its TP/SL
exactly, so on-chart results and account results measure different things. The
account reflects their management; the results table reflects the levels.

## Working with this user

They run everything on Windows via Git Bash and are not a developer — give
exact click-by-click steps, one command at a time, and expect to paste whole
file contents rather than describe edits.

**Pine scripts: always send the COMPLETE file, every time.** They paste it into
the TradingView editor wholesale and cannot hunt through it for "find this line,
change that one". Never send a partial diff for `tradingview/*.pine`.
They trade M1/M5/M15/M30, so those four are the timeframes that matter. Settings changes need a **full bot
restart** to take effect (`.env` is read once at startup); this has been the
cause of several "it didn't work" reports.
