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
| `backtest.py` | Replays current `.env` settings over history. **Use before any live change.** |
| `tradingview/*.pine` | Companion TradingView indicator (separate from the bot). |

## Modes (mutually exclusive, set in `.env`)

- `BALANCED_MODE` — H1/M15/**M5** trigger. The recommended one.
- `SCALP_MODE` — M15/M5/**M1**. High spread cost.
- `TURBO_MODE` — M1 on the **forming** bar, seconds-scale. Bleeds money on spread; demo experiments only.
- Default (all false) — D1/H4/H1/**M15** swing.

Timeframe slots in `analyzer.analyze()` are generic; each mode feeds
different timeframes into the same `d1/h4/h1/m15` parameters.

## Measured findings (do not re-guess these)

Backtest, BTCUSD 30d, M5, threshold 4/7, ADX≥18, SL 1.0×ATR, RR 1:2:

- 976 trades, 33.5% win rate, profit factor 0.82, **net −$1,282**
- **Spread was 91% of the loss** (976 × $1.20). Pre-spread the edge is ~zero.
- Trade count is the dominant cost driver. Fewer, higher-quality trades beats
  more trades at this timeframe.
- Trailing stop fired 6/976 times — dollar-based `TRAIL_ACTIVATE_USD` does not
  transfer across symbols/lot sizes. Consider making it ATR-relative.
- Spreads observed: **BTCUSD ~$12**, XAUUSD ~$0.25. Bitcoin is a poor scalping
  instrument despite trading 24/7.

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
