import os
import time
import csv
from collections import deque
from math import sqrt
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

load_dotenv()

JUP_PRICE_URL = "https://lite-api.jup.ag/price/v3"

# ==========================================================
# TOKENS & GROUPS
# ==========================================================
TOKENS = {
    # majors
    "So11111111111111111111111111111111111111112": {"label": "SOL",  "group": "majors"},
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": {"label": "mSOL", "group": "majors"},
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN": {"label": "JUP",  "group": "majors"},
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": {"label": "PYTH", "group": "majors"},
    "4k3Dyjzvzp8eMZWUXbBCjEvwSkkkPgkXyJzJ6kJ6KCQJ": {"label": "RAY",  "group": "majors"},
    # high volatility
    "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm": {"label": "WIF",  "group": "high_vol"},
}

# ==========================================================
# CONFIGS (day-trade bias)
# ==========================================================
CONFIGS = {
    "majors":   {"ema_fast": 12, "ema_slow": 36, "cooldown": 420},  # 7 min
    "high_vol": {"ema_fast": 10, "ema_slow": 30, "cooldown": 600},  # 10 min
}

SAMPLE_SECONDS = 5
WARMUP_SAMPLES_FACTOR = 1.0
CONFIRMATION_SAMPLES = 3

# ---------------- Base filters & risk ----------------
NOISE_BAND_BPS = 15                 # require price > slowEMA + band (BUY)
REQUIRE_TREND_SLOPE = True          # BUY only if slowEMA rising

# Raw-based TP/SL (distance from raw entry)
RAW_TP_PCT = 0.010                  # +1.0% from raw entry
RAW_SL_PCT = 0.010                  # -1.0% from raw entry
MIN_HOLD_SEC = 20                   # small grace period before TP/SL allowed

USE_TRAILING_STOP = True
TRAIL_PCT = 0.006                   # 0.6% trailing stop

# ---------------- New confirmation indicators ----------------
# RSI filter
USE_RSI_FILTER = True
RSI_LEN = 14
RSI_MAX_FOR_BUY = 70                # skip buys if RSI above this (overbought)
# ADX trend-strength filter (proxy using tick data)
USE_ADX_FILTER = True
ADX_LEN = 14
ADX_MIN_FOR_TREND = 22              # require ADX >= this for entry
# Bollinger Bands breakout filter
USE_BB_BREAKOUT = True
BB_PERIOD = 20
BB_STD = 2.0                        # price must be >= upper band to buy
# “Volume spike” proxy using abs returns vs moving average
USE_VOL_SPIKE = True
VOLWIN = 20
SPIKE_MULT = 1.2                    # abs_ret >= mean_abs_ret * SPIKE_MULT

# ---------------- Paper trading & logging ----------------
PAPER_UNITS = 1.0
LOG_PATH = "trades_log.csv"

# ---------------- Costs (per side) ----------------
FEE_BPS_PER_SIDE = 20               # 0.20%
SLIPPAGE_BPS_PER_SIDE = 10          # 0.10%
COST_RATE_PER_SIDE = (FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10000.0

# ---------------- Heartbeat ----------------
OPENPL_PUSH_EVERY_SEC = 300
INCLUDE_FLAT_TOKENS = False

# ---------------- Telegram ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================================================
# Helpers
# ==========================================================
def utc_now(): return datetime.now(timezone.utc)
def fmt_ts(dt): return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
def fmt_pct(x): return ("+" if x >= 0 else "") + f"{x*100:.2f}%"
def fmt_usd(x): return "n/a" if x is None else (f"${x:,.4f}" if abs(x) < 100 else f"${x:,.2f}")
def fmt_dur(sec):
    sec = int(sec); h, rem = divmod(sec, 3600); m, s = divmod(rem, 60)
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
            "" if entry_eff is None else f"{entry_eff:.8f}",
            "" if exit_eff is None else f"{exit_eff:.8f}",
            "" if pnl_pct is None else f"{pnl_pct:.6f}",
            "" if pnl_usd is None else f"{pnl_usd:.6f}",
            "" if hold_s is None else int(hold_s),
            FEE_BPS_PER_SIDE, SLIPPAGE_BPS_PER_SIDE, note
        ])

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=10
        )
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
    for m in mints:
        try:
            if m in data and "usdPrice" in data[m]:
                out[m] = float(data[m]["usdPrice"])
            elif "data" in data and m in data["data"] and "price" in data["data"][m]:
                out[m] = float(data["data"][m]["price"])
        except Exception:
            pass
    return out

# ==========================================================
# EMA Engine
# ==========================================================
class EmaCross:
    def __init__(self, fast, slow):
        self.a_f = 2/(fast+1); self.a_s = 2/(slow+1)
        self.ema_f = self.ema_s = self.prev_s = None
        self.last_sign = 0
    def update(self, price):
        if self.ema_f is None:
            self.ema_f = self.ema_s = self.prev_s = price
            return 0, 0
        self.ema_f = (price - self.ema_f)*self.a_f + self.ema_f
        self.prev_s = self.ema_s
        self.ema_s = (price - self.ema_s)*self.a_s + self.ema_s
        spread = self.ema_f - self.ema_s
        sign = 1 if spread > 0 else (-1 if spread < 0 else self.last_sign)
        crossed = 1 if (self.last_sign == -1 and sign == 1) else (-1 if (self.last_sign == 1 and sign == -1) else 0)
        self.last_sign = sign
        return sign, crossed

# ==========================================================
# Momentum/Volatility Indicators (RSI, ADX proxy, Bollinger, Spike)
# ==========================================================
class Indicators:
    def __init__(self, rsi_len=14, adx_len=14, bb_len=20, bb_std=2.0, volwin=20):
        self.rsi_len = rsi_len
        self.adx_len = adx_len
        self.bb_len = bb_len
        self.bb_std = bb_std
        self.volwin = volwin

        self.prev = None
        # RSI state (EMA/Wilder-like)
        self.avg_gain = None
        self.avg_loss = None
        self.rsi = None

        # ADX proxy state (using tick deltas: TR=|Δp|, DM+=max(Δp,0), DM-=max(-Δp,0))
        self.tr_ema = 0.0
        self.dm_pos_ema = 0.0
        self.dm_neg_ema = 0.0
        self.dx_ema = None
        self.adx = None
        self.alpha_wilder = 1.0 / adx_len

        # Bollinger state
        self.window = deque(maxlen=bb_len)

        # Volume spike proxy (abs returns)
        self.abs_ret_win = deque(maxlen=volwin)

    def update(self, price: float):
        # Always push price to BB window
        self.window.append(price)

        # Calculate delta
        if self.prev is None:
            self.prev = price
            return {
                "rsi": None, "adx": None,
                "bb_mid": None, "bb_up": None, "bb_lo": None,
                "vol_spike": False
            }

        delta = price - self.prev
        abs_delta = abs(delta)

        # ---------------- RSI (Wilder-like smoothing) ----------------
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        if self.avg_gain is None:
            # seed with first delta (simple)
            self.avg_gain = gain
            self.avg_loss = loss
        else:
            a = 1.0 / self.rsi_len
            self.avg_gain = (1 - a)*self.avg_gain + a*gain
            self.avg_loss = (1 - a)*self.avg_loss + a*loss

        if self.avg_loss and self.avg_loss > 1e-12:
            rs = self.avg_gain / self.avg_loss
            self.rsi = 100 - (100 / (1 + rs))
        else:
            self.rsi = 100.0  # no losses yet ~ overbought

        # ---------------- ADX proxy (tick-based) ----------------
        # True Range proxy: |Δp|
        tr = abs_delta
        dm_pos = gain
        dm_neg = loss
        # Wilder smoothing (EMA with alpha = 1/period)
        self.tr_ema = (1 - self.alpha_wilder) * self.tr_ema + self.alpha_wilder * tr
        self.dm_pos_ema = (1 - self.alpha_wilder) * self.dm_pos_ema + self.alpha_wilder * dm_pos
        self.dm_neg_ema = (1 - self.alpha_wilder) * self.dm_neg_ema + self.alpha_wilder * dm_neg

        di_pos = 100.0 * (self.dm_pos_ema / self.tr_ema) if self.tr_ema > 0 else 0.0
        di_neg = 100.0 * (self.dm_neg_ema / self.tr_ema) if self.tr_ema > 0 else 0.0
        denom = (di_pos + di_neg)
        dx = (100.0 * abs(di_pos - di_neg) / denom) if denom > 0 else 0.0

        if self.dx_ema is None:
            self.dx_ema = dx
        else:
            self.dx_ema = (1 - self.alpha_wilder) * self.dx_ema + self.alpha_wilder * dx
        self.adx = self.dx_ema

        # ---------------- Bollinger Bands ----------------
        bb_mid = bb_up = bb_lo = None
        if len(self.window) >= self.bb_len:
            n = len(self.window)
            mean = sum(self.window) / n
            var = sum((x - mean) ** 2 for x in self.window) / n
            std = sqrt(max(var, 0.0))
            bb_mid = mean
            bb_up = mean + self.bb_std * std
            bb_lo = mean - self.bb_std * std

        # ---------------- Volatility Spike (abs returns) ----------------
        vol_spike = False
        # Safe percent return (guard div by zero)
        if self.prev != 0:
            abs_ret = abs((price - self.prev) / self.prev)
            self.abs_ret_win.append(abs_ret)
            if len(self.abs_ret_win) >= max(5, self.volwin // 2):
                avg_abs = sum(self.abs_ret_win) / len(self.abs_ret_win)
                if avg_abs > 0 and abs_ret >= avg_abs * SPIKE_MULT:
                    vol_spike = True

        self.prev = price

        return {
            "rsi": self.rsi, "adx": self.adx,
            "bb_mid": bb_mid, "bb_up": bb_up, "bb_lo": bb_lo,
            "vol_spike": vol_spike
        }

# ==========================================================
# State initialization
# ==========================================================
engines, cooldowns, last_alert = {}, {}, {}
warmup, samples, pending = {}, {}, {}
positions, cum_pct, cum_usd = {}, {}, {}
inds = {}

for m, meta in TOKENS.items():
    c = CONFIGS[meta["group"]]
    engines[m] = EmaCross(c["ema_fast"], c["ema_slow"])
    cooldowns[m] = c["cooldown"]
    last_alert[m] = 0
    warmup[m] = int(c["ema_slow"] * WARMUP_SAMPLES_FACTOR)
    samples[m] = 0
    pending[m] = {"dir": 0, "count": 0}
    positions[m] = {"in": False, "entry_eff": None, "raw_entry": None, "t_entry": None, "trail_anchor": None}
    cum_pct[m] = 0.0; cum_usd[m] = 0.0
    inds[m] = Indicators(RSI_LEN, ADX_LEN, BB_PERIOD, BB_STD, VOLWIN)

# ==========================================================
# Startup message
# ==========================================================
watch_list = ", ".join(f"{v['label']}({v['group']})" for v in TOKENS.values())
send_telegram(
    "🤖 Day-trade mode online\n"
    f"Watching {len(TOKENS)} tokens: {watch_list}\n"
    f"Warmup≈{WARMUP_SAMPLES_FACTOR}×slow | Confirm={CONFIRMATION_SAMPLES} | Band={NOISE_BAND_BPS}bps\n"
    f"RAW TP/SL={RAW_TP_PCT*100:.1f}%/{RAW_SL_PCT*100:.1f}% | Trail={TRAIL_PCT*100:.1f}% | MinHold={MIN_HOLD_SEC}s\n"
    f"Filters: RSI≤{RSI_MAX_FOR_BUY}({RSI_LEN}), ADX≥{ADX_MIN_FOR_TREND}({ADX_LEN}), "
    f"BB breakout({BB_PERIOD},{BB_STD}σ), VolSpike×{SPIKE_MULT}\n"
    f"Costs {FEE_BPS_PER_SIDE+SLIPPAGE_BPS_PER_SIDE} bps/side"
)
ensure_log_header()
last_openpl = 0

# ==========================================================
# Utility functions
# ==========================================================
def pass_noise_band(p, s, d):  # d=+1 buy, -1 sell
    band = s * (NOISE_BAND_BPS/10000)
    return p >= s + band if d==1 else p <= s - band

def pass_trend_gate(e, d):
    if not REQUIRE_TREND_SLOPE or e.prev_s is None: return True
    rising = e.ema_s > e.prev_s
    return rising if d==1 else not rising

def maybe_push_openpl(prices):
    global last_openpl
    now = time.time()
    if now - last_openpl < OPENPL_PUSH_EVERY_SEC: return
    lines=[]
    for m, st in positions.items():
        if not st["in"] and not INCLUDE_FLAT_TOKENS: continue
        lbl = TOKENS[m]["label"]; px = prices.get(m)
        if st["in"] and px:
            exit_eff = px*(1-COST_RATE_PER_SIDE)
            entry = st["entry_eff"]
            upct = (exit_eff-entry)/entry
            uusd = upct*PAPER_UNITS*entry
            hold=(utc_now()-st["t_entry"]).total_seconds() if st["t_entry"] else 0
            lines.append(f"{lbl}: U/PnL {fmt_pct(upct)} ({fmt_usd(uusd)}) | Now {fmt_usd(px)} | In @ {fmt_usd(entry)} | Held {fmt_dur(hold)}")
        else: lines.append(f"{lbl}: flat")
    if lines:
        send_telegram("📈 Open positions:\n" + "\n".join(lines))
        last_openpl = now

# ==========================================================
# Main loop
# ==========================================================
while True:
    try:
        mints = list(TOKENS.keys())
        prices = get_prices(mints)
        if not prices: time.sleep(2); continue
        now = time.time(); now_dt = utc_now()

        for m, px in prices.items():
            lbl = TOKENS[m]["label"]
            sgn, crossed = engines[m].update(px)
            samples[m] += 1

            # Update indicators for this token
            ind = inds[m].update(px)
            rsi = ind["rsi"]; adx = ind["adx"]
            bb_up = ind["bb_up"]; bb_lo = ind["bb_lo"]; vol_spike = ind["vol_spike"]

            ef, es = engines[m].ema_f, engines[m].ema_s
            ef_s = f"{ef:.4f}" if ef is not None else "n/a"
            es_s = f"{es:.4f}" if es is not None else "n/a"
            print(f"{lbl}: {fmt_usd(px)} | EMA≈{ef_s}/{es_s} | RSI={None if rsi is None else round(rsi,1)} | ADX={None if adx is None else round(adx,1)} | pos={'ON' if positions[m]['in'] else 'OFF'}")

            # Warm-up gate
            if samples[m] < warmup[m]:
                pending[m] = {"dir": 0, "count": 0}
                continue

            # Confirmation logic
            if crossed != 0:
                pending[m] = {"dir": crossed, "count": 1}
            elif pending[m]["dir"] != 0 and sgn == pending[m]["dir"]:
                pending[m]["count"] += 1
            else:
                if pending[m]["dir"] != 0:
                    pending[m] = {"dir": 0, "count": 0}

            st = positions[m]

            # --------------- Risk exits ---------------
            if st["in"]:
                entry_eff = st["entry_eff"]
                exit_eff_now = px * (1 - COST_RATE_PER_SIDE)
                pnl_now = (exit_eff_now - entry_eff) / entry_eff

                # trailing stop (raw anchor)
                if USE_TRAILING_STOP:
                    if st["trail_anchor"] is None: st["trail_anchor"] = px
                    if px > st["trail_anchor"]: st["trail_anchor"] = px
                    if st["trail_anchor"] > 0 and px <= st["trail_anchor"] * (1 - TRAIL_PCT):
                        hold = (utc_now() - st["t_entry"]).total_seconds()
                        pnl_usd = pnl_now * PAPER_UNITS * entry_eff
                        cum_pct[m] += pnl_now; cum_usd[m] += pnl_usd
                        send_telegram(f"{lbl}: TRAIL STOP ⛔\nP/L {fmt_pct(pnl_now)} ({fmt_usd(pnl_usd)}) over {fmt_dur(hold)}")
                        log_row(lbl, "TRAIL_STOP", px, PAPER_UNITS, entry_eff, exit_eff_now, pnl_now, pnl_usd, hold, "trail")
                        positions[m] = {"in": False, "entry_eff": None, "raw_entry": None, "t_entry": None, "trail_anchor": None}
                        continue

                # raw-based TP/SL after min hold
                if (utc_now() - st["t_entry"]).total_seconds() >= MIN_HOLD_SEC:
                    raw_entry = st["raw_entry"]
                    if raw_entry:
                        raw_move = (px - raw_entry) / raw_entry
                        if raw_move >= RAW_TP_PCT or raw_move <= -RAW_SL_PCT:
                            exit_eff = exit_eff_now
                            pnl_pct = (exit_eff - entry_eff) / entry_eff
                            hold = (utc_now() - st["t_entry"]).total_seconds()
                            pnl_usd = pnl_pct * PAPER_UNITS * entry_eff
                            cum_pct[m] += pnl_pct; cum_usd[m] += pnl_usd
                            reason = "TAKE_PROFIT ✅" if raw_move >= RAW_TP_PCT else "STOP_LOSS ❌"
                            send_telegram(
                                f"{lbl}: {reason}\n"
                                f"Raw move {fmt_pct(raw_move)} | Eff P/L {fmt_pct(pnl_pct)} ({fmt_usd(pnl_usd)}) over {fmt_dur(hold)}"
                            )
                            log_row(lbl, reason, px, PAPER_UNITS, entry_eff, exit_eff, pnl_pct, pnl_usd, hold, "raw_trigger")
                            positions[m] = {"in": False, "entry_eff": None, "raw_entry": None, "t_entry": None, "trail_anchor": None}
                            continue

            # --------------- Entries ---------------
            pc = pending[m]
            if pc["dir"] != 0 and pc["count"] >= CONFIRMATION_SAMPLES and (now - last_alert[m] >= cooldowns[m]):
                side = pc["dir"]
                if side == 1 and not positions[m]["in"]:
                    # Existing gates
                    gate_band = pass_noise_band(px, engines[m].ema_s, +1)
                    gate_trend = pass_trend_gate(engines[m], +1)

                    # New indicator gates
                    gate_rsi = (True if not USE_RSI_FILTER or rsi is None else (rsi <= RSI_MAX_FOR_BUY))
                    gate_adx = (True if not USE_ADX_FILTER or adx is None else (adx >= ADX_MIN_FOR_TREND))
                    gate_bb  = (True if not USE_BB_BREAKOUT or bb_up is None else (px >= bb_up))
                    gate_spk = (True if not USE_VOL_SPIKE else bool(vol_spike))

                    if gate_band and gate_trend and gate_rsi and gate_adx and gate_bb and gate_spk:
                        entry_eff = px * (1 + COST_RATE_PER_SIDE)
                        positions[m] = {"in": True, "entry_eff": entry_eff, "raw_entry": px, "t_entry": now_dt, "trail_anchor": px}
                        send_telegram(
                            f"{lbl}: BUY ⬆️ @ {fmt_usd(px)} (eff {fmt_usd(entry_eff)})\n"
                            f"TP/SL {RAW_TP_PCT*100:.1f}%/{RAW_SL_PCT*100:.1f}% | Trail {TRAIL_PCT*100:.1f}%\n"
                            f"RSI={None if rsi is None else round(rsi,1)} | ADX={None if adx is None else round(adx,1)} | "
                            f"BB_up={'n/a' if bb_up is None else fmt_usd(bb_up)} | Spike={'Y' if gate_spk else 'N'}"
                        )
                        send_telegram(f"{lbl}: Raw TP @ {fmt_usd(px*(1+RAW_TP_PCT))} | Raw SL @ {fmt_usd(px*(1-RAW_SL_PCT))}")
                        log_row(lbl, "BUY", px, PAPER_UNITS, entry_eff, None, None, None, None, "buy_filters_ok")
                    else:
                        # Log which gates failed
                        note = f"band={gate_band},trend={gate_trend},rsi={gate_rsi},adx={gate_adx},bb={gate_bb},spike={gate_spk}"
                        log_row(lbl, "BUY_BLOCKED", px, PAPER_UNITS, None, None, None, None, None, note)

                elif side == -1 and positions[m]["in"]:
                    entry_eff = positions[m]["entry_eff"]
                    exit_eff = px * (1 - COST_RATE_PER_SIDE)
                    pnl = (exit_eff - entry_eff) / entry_eff
                    pnl_usd = pnl * PAPER_UNITS * entry_eff
                    hold = (utc_now() - positions[m]["t_entry"]).total_seconds()
                    cum_pct[m] += pnl; cum_usd[m] += pnl_usd
                    send_telegram(f"{lbl}: SELL ⬇️ confirmed | P/L {fmt_pct(pnl)} ({fmt_usd(pnl_usd)}) over {fmt_dur(hold)}")
                    log_row(lbl, "SELL", px, PAPER_UNITS, entry_eff, exit_eff, pnl, pnl_usd, hold, "confirmed_sell")
                    positions[m] = {"in": False, "entry_eff": None, "raw_entry": None, "t_entry": None, "trail_anchor": None}

                last_alert[m] = now
                pending[m] = {"dir": 0, "count": 0}

        maybe_push_openpl(prices)
        time.sleep(SAMPLE_SECONDS)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)





