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

### Live bot ran a different strategy than the backtest

`analyze_symbol()` fetched exactly 200 M15 bars while signals read `iloc[-2]`.
A 200-period SMA needs 201 bars to be defined there, so `sma200` was NaN,
`above_sma200` resolved to `None`, and the condition could satisfy neither BUY
nor SELL — the live bot silently scored 5-of-**6** while the backtest scored
5-of-7. Bar counts are now derived from `sma_200 + 150` (min 400), and
`analyze()` logs a warning if `sma200` is ever NaN on the signal bar.

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

## Working with this user

They run everything on Windows via Git Bash and are not a developer — give
exact click-by-click steps, one command at a time, and expect to paste whole
file contents rather than describe edits. Settings changes need a **full bot
restart** to take effect (`.env` is read once at startup); this has been the
cause of several "it didn't work" reports.
