"""
Weekly COT (Commitment of Traders) report for gold.

Pulls the CFTC's public Legacy futures-only data, computes net positioning for
large speculators and commercials, and flags how stretched it is against its own
history. Runs on your PC (this needs plain internet access, not MT5).

Run:  python cot_report.py                 print gold to the console
      python cot_report.py --telegram      also send it to your Telegram
      python cot_report.py --market SILVER
      python cot_report.py --weeks 260     history window for the percentile

Why it matters: COT shows who is holding what. Large speculators are trend
followers and are usually most crowded right before a reversal; commercials are
producers hedging and tend to lean the other way. Extremes are a context signal,
not a trade trigger — the CFTC data is published Friday but reflects the
PRECEDING TUESDAY, so it is always at least three days stale.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass

CFTC_ENDPOINT = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# Substring matched against the CFTC's market names
MARKETS = {
    "GOLD": "GOLD - COMMODITY EXCHANGE INC.",
    "SILVER": "SILVER - COMMODITY EXCHANGE INC.",
    "COPPER": "COPPER- #1 - COMMODITY EXCHANGE INC.",
    "PLATINUM": "PLATINUM - NEW YORK MERCANTILE EXCHANGE",
    "PALLADIUM": "PALLADIUM - NEW YORK MERCANTILE EXCHANGE",
}


@dataclass
class Week:
    date: str
    open_interest: int
    spec_long: int
    spec_short: int
    comm_long: int
    comm_short: int

    @property
    def spec_net(self) -> int:
        return self.spec_long - self.spec_short

    @property
    def comm_net(self) -> int:
        return self.comm_long - self.comm_short

    @property
    def spec_long_pct(self) -> float:
        total = self.spec_long + self.spec_short
        return (self.spec_long / total * 100) if total else 0.0


def _num(row: dict, *names: str) -> int:
    """CFTC field names vary and contain typos; try each in turn."""
    for n in names:
        if n in row and row[n] not in (None, ""):
            try:
                return int(float(row[n]))
            except (TypeError, ValueError):
                continue
    return 0


def fetch(market: str, weeks: int) -> list[Week]:
    query = {
        "$where": f"market_and_exchange_names like '%{market}%'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(weeks),
    }
    url = f"{CFTC_ENDPOINT}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"User-Agent": "cot-report/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        rows = json.loads(resp.read().decode())

    out: list[Week] = []
    for r in rows:
        out.append(Week(
            date=str(r.get("report_date_as_yyyy_mm_dd", ""))[:10],
            open_interest=_num(r, "open_interest_all"),
            spec_long=_num(r, "noncomm_positions_long_all"),
            spec_short=_num(r, "noncomm_positions_short_all"),
            comm_long=_num(r, "comm_positions_long_all"),
            comm_short=_num(r, "comm_positions_short_all"),
        ))
    return out


def percentile(value: float, series: list[float]) -> float:
    if not series:
        return 50.0
    below = sum(1 for v in series if v < value)
    return below / len(series) * 100


def build_report(market_key: str, weeks: list[Week]) -> str:
    if not weeks:
        return f"No COT data returned for {market_key}."

    now = weeks[0]
    prev = weeks[1] if len(weeks) > 1 else now
    hist = [float(w.spec_net) for w in weeks]

    spec_chg = now.spec_net - prev.spec_net
    comm_chg = now.comm_net - prev.comm_net
    oi_chg = now.open_interest - prev.open_interest
    pct = percentile(float(now.spec_net), hist)

    if pct >= 90:
        read = "specs VERY crowded long — stretched, reversal risk"
    elif pct >= 75:
        read = "specs heavily long"
    elif pct <= 10:
        read = "specs VERY crowded short — stretched, squeeze risk"
    elif pct <= 25:
        read = "specs heavily short"
    else:
        read = "positioning is mid-range — no extreme"

    lines = [
        f"COT — {market_key}   (week of {now.date})",
        "=" * 52,
        f"Open interest      {now.open_interest:>12,}  ({oi_chg:+,})",
        "",
        "Large speculators (trend followers)",
        f"  long             {now.spec_long:>12,}",
        f"  short            {now.spec_short:>12,}",
        f"  NET              {now.spec_net:>12,}  ({spec_chg:+,} on the week)",
        f"  long share       {now.spec_long_pct:>11.1f}%",
        f"  vs {len(weeks)} weeks     {pct:>11.0f}th percentile",
        "",
        "Commercials (producers / hedgers)",
        f"  long             {now.comm_long:>12,}",
        f"  short            {now.comm_short:>12,}",
        f"  NET              {now.comm_net:>12,}  ({comm_chg:+,} on the week)",
        "",
        f"Read: {read}",
        "",
        "Data is as of the Tuesday before publication, so it is at least 3 days",
        "stale. Use it as background context, never as an entry trigger.",
    ]
    return "\n".join(lines)


def send_telegram(text: str) -> None:
    from config import load_config
    cfg = load_config()
    for chat_id in cfg.telegram.allowed_chat_ids:
        payload = urllib.parse.urlencode({
            "chat_id": str(chat_id),
            "text": f"```\n{text}\n```",
            "parse_mode": "Markdown",
        }).encode()
        url = f"https://api.telegram.org/bot{cfg.telegram.bot_token}/sendMessage"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=payload), timeout=30) as r:
                r.read()
            print(f"Sent to chat {chat_id}")
        except Exception as e:
            print(f"Telegram send failed for {chat_id}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly CFTC COT report")
    ap.add_argument("--market", default="GOLD", help="GOLD, SILVER, COPPER, PLATINUM, PALLADIUM")
    ap.add_argument("--weeks", type=int, default=156, help="history for the percentile (default 3y)")
    ap.add_argument("--telegram", action="store_true", help="also send to Telegram")
    args = ap.parse_args()

    key = args.market.upper()
    needle = MARKETS.get(key, key)

    try:
        weeks = fetch(needle, args.weeks)
    except Exception as e:
        print(f"Could not fetch COT data: {e}")
        print("Check internet access, or that the market name matches the CFTC's.")
        return 1

    report = build_report(key, weeks)
    print(report)
    if args.telegram:
        send_telegram(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
