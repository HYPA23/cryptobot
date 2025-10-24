import os
import time
import csv
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

load_dotenv()

JUP_PRICE_URL = "https://lite-api.jup.ag/price/v3"

# -------------------------
# TOKENS & GROUPS
# -------------------------
TOKENS = {
    # majors
    "So11111111111111111111111111111111111111112": {"label": "SOL",  "group": "majors"},
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": {"label": "mSOL", "group": "majors"},
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN": {"label": "JUP",  "group": "majors"},
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": {"label": "PYTH", "group": "majors"},
    # high volatility
    "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm": {"label": "WIF",  "group": "high_vol"},
    "4k3Dyjzvzp8eMZWUXbBCjEvwSkkkPgkXyJzJ6kJ6KCQJ": {"label": "RAY", "group": "majors"},

}

# Group configs (day-trade bias)
CONFIGS = {
    "majors":   {"ema_fast": 12, "ema_slow": 36, "cooldown": 420},  # 5 min
    "high_vol": {"ema_fast": 10, "ema_slow": 30, "cooldown": 600},  # 8 min
}

SAMPLE_SECONDS = 5

# --- Anti-noise (existing) ---
WARMUP_SAMPLES_FACTOR = 1.0
CONFIRMATION_SAMPLES = 3

# --- New: filters & risk ---
NOISE_BAND_BPS = 15          # require price to be > slowEMA by this many bps for BUY; below for SELL (0.15%)
REQUIRE_TREND_SLOPE = True   # BUY only if slowEMA rising vs last tick; SELL only if falling
TP_PCT = 0.01               # 0.7% take-profit
SL_PCT = 0.01               # 1% stop-loss
USE_TRAILING_STOP = True
TRAIL_PCT = 0.006            # 0.5% trailing from best favorable price

# --- Paper trading & logging ---
PAPER_UNITS = 1.0
LOG_PATH = "trades_log.csv"

# --- Costs (per side) ---
FEE_BPS_PER_SIDE = 20        # 0.20%
SLIPPAGE_BPS_PER_SIDE = 10   # 0.10%
COST_RATE_PER_SIDE = (FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10000.0

# --- Open P/L heartbeat ---
OPENPL_PUSH_EVERY_SEC = 300
INCLUDE_FLAT_TOKENS = False

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def utc_now(): return datetime.now(timezone.utc)
def fmt_ts(dt): return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
def fmt_pct(x): return ("+" if x >= 0 else "") + f"{x*100:.2f}%"
def fmt_usd(x):
    if x is None: return "n/a"
    return f"${x:,.4f}" if abs(x) < 100 else f"${x:,.2f}"
def fmt_dur(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp_utc","token","action","price_usd","units",
                "entry_eff","exit_eff","pnl_pct","pnl_usd",
                "hold_seconds","fee_bps","slip_bps","note"
            ])

def log_row(token, action, price, units, entry_eff=None, exit_eff=None,
            pnl_pct=None, pnl_usd=None, hold_s=None, note=""):
    ensure_log_header()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            fmt_ts(utc_now()), token, action, f"{price:.8f}", units,
            ("" if entry_eff is None else f"{entry_eff:.8f}"),
            ("" if exit_eff is None else f"{exit_eff:.8f}"),
            ("" if pnl_pct is None else f"{pnl_pct:.6f}"),
            ("" if pnl_usd is None else f"{pnl_usd:.6f}"),
            ("" if hold_s is None else int(hold_s)),
            FEE_BPS_PER_SIDE, SLIPPAGE_BPS_PER_SIDE, note
        ])

def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except requests.RequestException as e:
        print("Telegram error:", e)

def get_prices(mints):
    if not mints: return {}
    try:
        r = requests.get(JUP_PRICE_URL, params={"ids": ",".join(mints)}, timeout=10)
        r.raise_for_status(); data = r.json()
    except Exception as e:
        print("Price fetch error:", e); return {}
    out = {}
    for mint in mints:
        try:
            if mint in data and isinstance(data[mint], dict) and "usdPrice" in data[mint]:
                out[mint] = float(data[mint]["usdPrice"])
            elif "data" in data and mint in data["data"] and "price" in data["data"][mint]:
                out[mint] = float(data["data"][mint]["price"])
        except Exception: pass
    return out

class EmaCross:
    def __init__(self, fast, slow):
        self.alpha_fast = 2 / (fast + 1)
        self.alpha_slow = 2 / (slow + 1)
        self.ema_fast = None
        self.ema_slow = None
        self.prev_ema_slow = None
        self.last_sign = 0  # -1 fast<slow, +1 fast>slow

    def update(self, price: float):
        """Returns (sign +1/-1, crossed +1/-1/0)."""
        if self.ema_fast is None:
            self.ema_fast = price; self.ema_slow = price; self.prev_ema_slow = price
            return 0, 0
        self.ema_fast = (price - self.ema_fast) * self.alpha_fast + self.ema_fast
        # keep previous slow for slope check
        self.prev_ema_slow = self.ema_slow
        self.ema_slow = (price - self.ema_slow) * self.alpha_slow + self.ema_slow
        spread = self.ema_fast - self.ema_slow
        sign = 1 if spread > 0 else (-1 if spread < 0 else self.last_sign)
        crossed = 0
        if self.last_sign != 0 and sign != self.last_sign:
            crossed = 1 if sign > 0 else -1
        self.last_sign = sign
        return sign, crossed

# Per-token state
engines, cooldowns, last_alert_time = {}, {}, {}
warmup_needed, samples_seen, pending_cross = {}, {}, {}
positions = {}  # mint -> {"in": bool, "entry_eff": float, "raw_entry": float, "t_entry": dt, "trail_anchor": float}
cum_pnl_pct, cum_pnl_usd = {}, {}

for mint, meta in TOKENS.items():
    cfg = CONFIGS[meta["group"]]
    engines[mint] = EmaCross(cfg["ema_fast"], cfg["ema_slow"])
    cooldowns[mint] = cfg["cooldown"]
    last_alert_time[mint] = 0
    warmup_needed[mint] = int(cfg["ema_slow"] * WARMUP_SAMPLES_FACTOR)
    samples_seen[mint] = 0
    pending_cross[mint] = {"dir": 0, "count": 0}
    positions[mint] = {"in": False, "entry_eff": None, "raw_entry": None, "t_entry": None, "trail_anchor": None}
    cum_pnl_pct[mint] = 0.0
    cum_pnl_usd[mint] = 0.0

watch_list = ", ".join(f'{info["label"]}({info["group"]})' for info in TOKENS.values())
send_telegram(
    "🤖 Day-trade mode online with filters & risk.\n"
    f"Watching {len(TOKENS)} tokens: {watch_list}\n"
    f"Warmup≈{WARMUP_SAMPLES_FACTOR}×slow | Confirm={CONFIRMATION_SAMPLES} | Band={NOISE_BAND_BPS}bps | "
    f"TP={TP_PCT*100:.2f}% SL={SL_PCT*100:.2f}% Trail={TRAIL_PCT*100:.2f}%"
)
ensure_log_header()
last_openpl_push = 0

def maybe_push_openpl_summary(prices: dict):
    global last_openpl_push
    now = time.time()
    if now - last_openpl_push < OPENPL_PUSH_EVERY_SEC:
        return
    lines = []
    for mint, st in positions.items():
        if not st["in"] and not INCLUDE_FLAT_TOKENS: continue
        label = TOKENS[mint]["label"]; px = prices.get(mint)
        if st["in"] and px is not None:
            exit_eff = px * (1 - COST_RATE_PER_SIDE)
            entry_eff = st["entry_eff"]
            unreal_pct = (exit_eff - entry_eff) / entry_eff
            unreal_usd = unreal_pct * PAPER_UNITS * entry_eff
            hold_s = (utc_now() - st["t_entry"]).total_seconds() if st["t_entry"] else 0
            lines.append(f"{label}: U/PnL {fmt_pct(unreal_pct)} ({fmt_usd(unreal_usd)}) | Now {fmt_usd(px)} | In @ {fmt_usd(entry_eff)} | Held {fmt_dur(hold_s)}")
        else:
            lines.append(f"{label}: flat")
    if lines:
        send_telegram("📈 Open positions:\n" + "\n".join(lines))
        last_openpl_push = now

def pass_noise_band(price, ema_slow, side_dir):
    # side_dir: +1 for BUY, -1 for SELL
    band = ema_slow * (NOISE_BAND_BPS / 10000.0)
    return (price >= ema_slow + band) if side_dir == 1 else (price <= ema_slow - band)

def pass_trend_gate(engine, side_dir):
    if not REQUIRE_TREND_SLOPE:
        return True
    # BUY only if slow EMA rising, SELL only if falling
    if engine.prev_ema_slow is None: return False
    rising = engine.ema_slow > engine.prev_ema_slow
    return rising if side_dir == 1 else (not rising)

while True:
    try:
        mints = list(TOKENS.keys())
        prices = get_prices(mints)
        if not prices:
            time.sleep(2); continue

        now = time.time(); now_dt = utc_now()

        for mint, price in prices.items():
            label = TOKENS[mint]["label"]
            sign, crossed = engines[mint].update(price)
            samples_seen[mint] += 1

            ef = engines[mint].ema_fast; es = engines[mint].ema_slow
            ef_s = f"{ef:.4f}" if ef is not None else "n/a"
            es_s = f"{es:.4f}" if es is not None else "n/a"
            print(f"{label}: {fmt_usd(price)} | EMA≈{ef_s}/{es_s} | seen={samples_seen[mint]} | pos={'ON' if positions[mint]['in'] else 'OFF'}")

            # Warm-up gate
            if samples_seen[mint] < warmup_needed[mint]:
                pending_cross[mint] = {"dir": 0, "count": 0}
                continue

            # Confirmation
            if crossed != 0:
                pending_cross[mint] = {"dir": crossed, "count": 1}
            else:
                if pending_cross[mint]["dir"] != 0 and sign == pending_cross[mint]["dir"]:
                    pending_cross[mint]["count"] += 1
                else:
                    if pending_cross[mint]["dir"] != 0:
                        pending_cross[mint] = {"dir": 0, "count": 0}

            # Risk exits (TP/SL/Trail) if in position
            st = positions[mint]
            if st["in"]:
                entry_eff = st["entry_eff"]
                # effective exit if we sold now (cost on exit)
                exit_eff_now = price * (1 - COST_RATE_PER_SIDE)
                pnl_pct_now = (exit_eff_now - entry_eff) / entry_eff

                # trailing anchor update (max favorable price after entry)
                if USE_TRAILING_STOP:
                    if st["trail_anchor"] is None:
                        st["trail_anchor"] = price
                    # for a long, anchor is the highest price seen since entry
                    if price > st["trail_anchor"]:
                        st["trail_anchor"] = price
                    # if price falls X% from anchor, trigger trail stop
                    if st["trail_anchor"] > 0 and (price <= st["trail_anchor"] * (1 - TRAIL_PCT)):
                        # trail stop fires
                        exit_eff = exit_eff_now
                        hold_s = (utc_now() - st["t_entry"]).total_seconds() if st["t_entry"] else 0
                        pnl_usd = pnl_pct_now * PAPER_UNITS * entry_eff
                        cum_pnl_pct[mint] += pnl_pct_now; cum_pnl_usd[mint] += pnl_usd
                        send_telegram(
                            f"{label}: TRAIL STOP ⛔\n"
                            f"Exit eff: {fmt_usd(exit_eff)} | Entry eff: {fmt_usd(entry_eff)}\n"
                            f"P/L: {fmt_pct(pnl_pct_now)} ({fmt_usd(pnl_usd)}) over {fmt_dur(hold_s)}\n"
                            f"Cumulative: {fmt_pct(cum_pnl_pct[mint])} ({fmt_usd(cum_pnl_usd[mint])})"
                        )
                        log_row(label, "TRAIL_STOP", price, PAPER_UNITS, entry_eff=entry_eff, exit_eff=exit_eff,
                                pnl_pct=pnl_pct_now, pnl_usd=pnl_usd, hold_s=hold_s, note="trail_stop")
                        positions[mint] = {"in": False, "entry_eff": None, "raw_entry": None, "t_entry": None, "trail_anchor": None}
                        # after risk exit, skip further actions this tick
                        continue

                # hard TP/SL (based on effective prices)
                if pnl_pct_now >= TP_PCT or pnl_pct_now <= -SL_PCT:
                    exit_eff = exit_eff_now
                    hold_s = (utc_now() - st["t_entry"]).total_seconds() if st["t_entry"] else 0
                    pnl_usd = pnl_pct_now * PAPER_UNITS * entry_eff
                    cum_pnl_pct[mint] += pnl_pct_now; cum_pnl_usd[mint] += pnl_usd
                    reason = "TAKE_PROFIT ✅" if pnl_pct_now >= TP_PCT else "STOP_LOSS ❌"
                    send_telegram(
                        f"{label}: {reason}\n"
                        f"Exit eff: {fmt_usd(exit_eff)} | Entry eff: {fmt_usd(entry_eff)}\n"
                        f"P/L: {fmt_pct(pnl_pct_now)} ({fmt_usd(pnl_usd)}) over {fmt_dur(hold_s)}\n"
                        f"Cumulative: {fmt_pct(cum_pnl_pct[mint])} ({fmt_usd(cum_pnl_usd[mint])})"
                    )
                    log_row(label, reason, price, PAPER_UNITS, entry_eff=entry_eff, exit_eff=exit_eff,
                            pnl_pct=pnl_pct_now, pnl_usd=pnl_usd, hold_s=hold_s, note=reason.lower())
                    positions[mint] = {"in": False, "entry_eff": None, "raw_entry": None, "t_entry": None, "trail_anchor": None}
                    continue

            # If confirmed cross and cooldown ok, consider entry/exit
            pc = pending_cross[mint]
            if pc["dir"] != 0 and pc["count"] >= CONFIRMATION_SAMPLES and (now - last_alert_time[mint] >= cooldowns[mint]):
                side = pc["dir"]  # +1 buy, -1 sell

                # entry gating (trend & band) for BUY only (we're long-only here)
                if side == 1 and not positions[mint]["in"]:
                    passes_band = pass_noise_band(price, engines[mint].ema_slow, +1)
                    passes_trend = pass_trend_gate(engines[mint], +1)
                    if passes_band and passes_trend:
                        entry_eff = price * (1 + COST_RATE_PER_SIDE)
                        positions[mint] = {"in": True, "entry_eff": entry_eff, "raw_entry": price, "t_entry": now_dt, "trail_anchor": price}
                        send_telegram(
                            f"{label}: BUY ⬆️  (confirmed + filters passed)\n"
                            f"Entry eff: {fmt_usd(entry_eff)} (raw {fmt_usd(price)})\n"
                            f"Band={NOISE_BAND_BPS}bps | Trend={'ON' if REQUIRE_TREND_SLOPE else 'OFF'} | "
                            f"TP={TP_PCT*100:.2f}% SL={SL_PCT*100:.2f}% Trail={TRAIL_PCT*100:.2f}%"
                        )
                        log_row(label, "BUY", price, PAPER_UNITS, entry_eff=entry_eff, note="confirmed_buy_filters_passed")
                    else:
                        log_row(label, "BUY_BLOCKED", price, PAPER_UNITS, note=f"band={passes_band},trend={passes_trend}")
                elif side == -1:
                    # exit on confirmed sell if in position (even if TP/SL didn't trigger yet)
                    if positions[mint]["in"]:
                        entry_eff = positions[mint]["entry_eff"]
                        exit_eff = price * (1 - COST_RATE_PER_SIDE)
                        hold_s = (utc_now() - positions[mint]["t_entry"]).total_seconds() if positions[mint]["t_entry"] else 0
                        pnl_pct = (exit_eff - entry_eff) / entry_eff
                        pnl_usd = pnl_pct * PAPER_UNITS * entry_eff
                        cum_pnl_pct[mint] += pnl_pct; cum_pnl_usd[mint] += pnl_usd
                        send_telegram(
                            f"{label}: SELL ⬇️ (confirmed cross)\n"
                            f"Exit eff: {fmt_usd(exit_eff)} | Entry eff: {fmt_usd(entry_eff)}\n"
                            f"P/L: {fmt_pct(pnl_pct)} ({fmt_usd(pnl_usd)}) over {fmt_dur(hold_s)}\n"
                            f"Cumulative: {fmt_pct(cum_pnl_pct[mint])} ({fmt_usd(cum_pnl_usd[mint])})"
                        )
                        log_row(label, "SELL", price, PAPER_UNITS, entry_eff=entry_eff, exit_eff=exit_eff,
                                pnl_pct=pnl_pct, pnl_usd=pnl_usd, hold_s=hold_s, note="confirmed_sell")
                        positions[mint] = {"in": False, "entry_eff": None, "raw_entry": None, "t_entry": None, "trail_anchor": None}

                last_alert_time[mint] = now
                pending_cross[mint] = {"dir": 0, "count": 0}

        maybe_push_openpl_summary(prices)
        time.sleep(SAMPLE_SECONDS)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)






