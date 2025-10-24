import os
import time
import csv
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
# CONFIGS
# ==========================================================
CONFIGS = {
    "majors":   {"ema_fast": 12, "ema_slow": 36, "cooldown": 420},  # 7 min
    "high_vol": {"ema_fast": 10, "ema_slow": 30, "cooldown": 600},  # 10 min
}

SAMPLE_SECONDS = 5
WARMUP_SAMPLES_FACTOR = 1.0
CONFIRMATION_SAMPLES = 3

# ---------------- Filters & risk ----------------
NOISE_BAND_BPS = 15
REQUIRE_TREND_SLOPE = True
RAW_TP_PCT = 0.010     # +1.0% raw take-profit
RAW_SL_PCT = 0.010     # −1.0% raw stop-loss
MIN_HOLD_SEC = 20      # minimum seconds before TP/SL can fire
USE_TRAILING_STOP = True
TRAIL_PCT = 0.006      # 0.6 % trailing stop from best favorable price

# ---------------- Paper trading & logging ----------------
PAPER_UNITS = 1.0
LOG_PATH = "trades_log.csv"

# ---------------- Costs (per side) ----------------
FEE_BPS_PER_SIDE = 20        # 0.20 %
SLIPPAGE_BPS_PER_SIDE = 10   # 0.10 %
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
# State initialization
# ==========================================================
engines, cooldowns, last_alert = {}, {}, {}
warmup, samples, pending = {}, {}, {}
positions, cum_pct, cum_usd = {}, {}, {}

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

# ==========================================================
# Startup message
# ==========================================================
watch_list = ", ".join(f"{v['label']}({v['group']})" for v in TOKENS.values())
send_telegram(
    "🤖 Day-trade mode online\n"
    f"Watching {len(TOKENS)} tokens: {watch_list}\n"
    f"Warmup≈{WARMUP_SAMPLES_FACTOR}×slow | Confirm={CONFIRMATION_SAMPLES} | "
    f"Band={NOISE_BAND_BPS}bps | RAW TP/SL={RAW_TP_PCT*100:.1f}%/{RAW_SL_PCT*100:.1f}% | "
    f"Trail={TRAIL_PCT*100:.1f}% | Costs {FEE_BPS_PER_SIDE+SLIPPAGE_BPS_PER_SIDE} bps/side"
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
            ef, es = engines[m].ema_f, engines[m].ema_s
            print(f"{lbl}: {fmt_usd(px)} | EMA≈{ef:.4f}/{es:.4f} | pos={'ON' if positions[m]['in'] else 'OFF'}")

            if samples[m] < warmup[m]:
                pending[m]={"dir":0,"count":0}; continue

            # Confirmation logic
            if crossed!=0:
                pending[m]={"dir":crossed,"count":1}
            elif pending[m]["dir"]!=0 and sgn==pending[m]["dir"]:
                pending[m]["count"]+=1
            else:
                if pending[m]["dir"]!=0: pending[m]={"dir":0,"count":0}

            st=positions[m]
            # --------------- Risk exits ---------------
            if st["in"]:
                entry_eff=st["entry_eff"]
                exit_eff_now=px*(1-COST_RATE_PER_SIDE)
                pnl_now=(exit_eff_now-entry_eff)/entry_eff

                # trailing stop
                if USE_TRAILING_STOP:
                    if st["trail_anchor"] is None: st["trail_anchor"]=px
                    if px>st["trail_anchor"]: st["trail_anchor"]=px
                    if st["trail_anchor"]>0 and px<=st["trail_anchor"]*(1-TRAIL_PCT):
                        hold=(utc_now()-st["t_entry"]).total_seconds()
                        pnl_usd=pnl_now*PAPER_UNITS*entry_eff
                        cum_pct[m]+=pnl_now; cum_usd[m]+=pnl_usd
                        send_telegram(f"{lbl}: TRAIL STOP ⛔\nP/L {fmt_pct(pnl_now)} ({fmt_usd(pnl_usd)}) over {fmt_dur(hold)}")
                        log_row(lbl,"TRAIL_STOP",px,PAPER_UNITS,entry_eff,exit_eff_now,pnl_now,pnl_usd,hold,"trail")
                        positions[m]={"in":False,"entry_eff":None,"raw_entry":None,"t_entry":None,"trail_anchor":None}
                        continue

                # raw-based TP/SL
                if (utc_now()-st["t_entry"]).total_seconds() >= MIN_HOLD_SEC:
                    raw_entry=st["raw_entry"]
                    raw_move=(px-raw_entry)/raw_entry
                    if raw_move>=RAW_TP_PCT or raw_move<=-RAW_SL_PCT:
                        exit_eff=exit_eff_now
                        pnl_pct=(exit_eff-entry_eff)/entry_eff
                        hold=(utc_now()-st["t_entry"]).total_seconds()
                        pnl_usd=pnl_pct*PAPER_UNITS*entry_eff
                        cum_pct[m]+=pnl_pct; cum_usd[m]+=pnl_usd
                        reason="TAKE_PROFIT ✅" if raw_move>=RAW_TP_PCT else "STOP_LOSS ❌"
                        send_telegram(
                            f"{lbl}: {reason}\n"
                            f"Raw move {fmt_pct(raw_move)} | Eff P/L {fmt_pct(pnl_pct)} ({fmt_usd(pnl_usd)}) over {fmt_dur(hold)}"
                        )
                        log_row(lbl,reason,px,PAPER_UNITS,entry_eff,exit_eff,pnl_pct,pnl_usd,hold,"raw_trigger")
                        positions[m]={"in":False,"entry_eff":None,"raw_entry":None,"t_entry":None,"trail_anchor":None}
                        continue

            # --------------- Entries ---------------
            pc=pending[m]
            if pc["dir"]!=0 and pc["count"]>=CONFIRMATION_SAMPLES and (now-last_alert[m]>=cooldowns[m]):
                side=pc["dir"]
                if side==1 and not positions[m]["in"]:
                    if pass_noise_band(px,engines[m].ema_s,+1) and pass_trend_gate(engines[m],+1):
                        entry_eff=px*(1+COST_RATE_PER_SIDE)
                        positions[m]={"in":True,"entry_eff":entry_eff,"raw_entry":px,"t_entry":now_dt,"trail_anchor":px}
                        send_telegram(f"{lbl}: BUY ⬆️ @ {fmt_usd(px)} (eff {fmt_usd(entry_eff)})\nTP/SL {RAW_TP_PCT*100:.1f}%/{RAW_SL_PCT*100:.1f}% Trail {TRAIL_PCT*100:.1f}%")
                        send_telegram(f"{lbl}: Raw TP @ {fmt_usd(px*(1+RAW_TP_PCT))} | Raw SL @ {fmt_usd(px*(1-RAW_SL_PCT))}")
                        log_row(lbl,"BUY",px,PAPER_UNITS,entry_eff,None,None,None,None,"buy")
                    else:
                        log_row(lbl,"BUY_BLOCKED",px,PAPER_UNITS,None,None,None,None,None,"filter")
                elif side==-1 and positions[m]["in"]:
                    entry_eff=positions[m]["entry_eff"]
                    exit_eff=px*(1-COST_RATE_PER_SIDE)
                    pnl=(exit_eff-entry_eff)/entry_eff
                    pnl_usd=pnl*PAPER_UNITS*entry_eff
                    hold=(utc_now()-positions[m]["t_entry"]).total_seconds()
                    cum_pct[m]+=pnl; cum_usd[m]+=pnl_usd
                    send_telegram(f"{lbl}: SELL ⬇️ confirmed | P/L {fmt_pct(pnl)} ({fmt_usd(pnl_usd)}) over {fmt_dur(hold)}")
                    log_row(lbl,"SELL",px,PAPER_UNITS,entry_eff,exit_eff,pnl,pnl_usd,hold,"confirmed_sell")
                    positions[m]={"in":False,"entry_eff":None,"raw_entry":None,"t_entry":None,"trail_anchor":None}
                last_alert[m]=now
                pending[m]={"dir":0,"count":0}

        maybe_push_openpl(prices)
        time.sleep(SAMPLE_SECONDS)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)





