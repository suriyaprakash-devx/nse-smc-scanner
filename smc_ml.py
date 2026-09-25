"""Pandas translation of TradingView's Smart Money Concept ML indicator.

Input data must have a DatetimeIndex (or a ``time`` column) and OHLC columns
named ``open``, ``high``, ``low`` and ``close``.  ``run_smc_ml`` returns the
per-bar data plus data frames for breaks, signals, zones and liquidity pools.

This is an indicator/back-test translation: TradingView labels, lines, boxes,
tables and alert() calls are represented as data, not drawn automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from math import erf, exp, log, sqrt
from typing import Literal, Optional, Dict, Any, List

import numpy as np
import pandas as pd

CAL_CAP, CAL_LAMBDA, CAL_MIN, FT_WIN, DRAW_WIN = 500, 50.0, 30, 150, 1000


@dataclass
class Settings:
    swing_length: int = 10
    internal_length: int = 3
    fvg_min_atr: float = 0.3
    equal_tolerance_atr: float = 0.15
    retest_window: int = 20
    entry: Literal["Break Close", "Level Retest", "Adaptive"] = "Break Close"
    adaptive_threshold: float = 0.80
    signal_breaks: Literal["CHoCH + BOS", "CHoCH only", "BOS only"] = "CHoCH + BOS"
    stop_buffer_atr: float = 0.20
    max_risk_atr: float = 6.0
    tp1_r: float = 1.0
    tp2_r: float = 2.0
    stop_to_entry_after_tp1: bool = True
    trade_timeout: int = 150
    order_blocks_per_side: int = 2


def _clamp_p(p: float) -> float:
    return max(0.005, min(0.995, p))


def _logit(p: float) -> float:
    p = _clamp_p(p)
    return log(p / (1 - p))


def _touch(distance: float, sd: float, bars: int) -> float:
    """Reflection-principle chance a zero-drift walk touches a level."""
    if not np.isfinite(distance) or not np.isfinite(sd) or sd <= 0:
        return np.nan
    phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    return _clamp_p(2 * (1 - phi(distance / (sd * sqrt(bars)))))


@dataclass
class Calibration:
    fix_intercept: bool = False
    cut1: float = .70
    cut2: float = .90
    x: list[float] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    a: float = 0.0
    b: float = 1.0
    n: int = 0
    hits: int = 0
    sum_p: float = 0.0
    brier_model: float = 0.0
    brier_formula: float = 0.0
    brier_base: float = 0.0

    def probability(self, formula_p: float) -> float:
        if not np.isfinite(formula_p):
            return np.nan
        z = max(-700, min(700, self.a + self.b * _logit(formula_p)))
        return 1 / (1 + exp(-z))

    def score(self, model_p: float, formula_p: float, outcome: int) -> None:
        """Score a previously displayed probability, then MAP-refit the model."""
        base = self.hits / self.n if self.n else .5
        self.brier_base += (base - outcome) ** 2
        self.brier_model += (model_p - outcome) ** 2
        self.brier_formula += (formula_p - outcome) ** 2
        self.sum_p += model_p
        self.n += 1
        self.hits += outcome
        self.x.append(_logit(formula_p)); self.y.append(outcome)
        if len(self.x) > CAL_CAP:
            self.x.pop(0); self.y.pop(0)
        self.fit()

    def fit(self) -> None:
        """Four Newton iterations of Pine's prior-centred logistic MAP fit."""
        if not self.x:
            return
        a, b = self.a, self.b
        for _ in range(4):
            ga, gb = CAL_LAMBDA * a, CAL_LAMBDA * (b - 1)
            haa, hab, hbb = CAL_LAMBDA, 0., CAL_LAMBDA
            for x, y in zip(self.x, self.y):
                z = max(-700, min(700, a + b * x)); p = 1 / (1 + exp(-z))
                e, w = p - y, p * (1 - p)
                ga += e; gb += e * x; haa += w; hab += w * x; hbb += w * x * x
            if self.fix_intercept:
                b -= gb / hbb
            else:
                det = haa * hbb - hab * hab
                if det > 0:
                    a -= (hbb * ga - hab * gb) / det
                    b -= (haa * gb - hab * ga) / det
        self.a, self.b = a, b


@dataclass
class Structure:
    length: int
    hi: float = np.nan; lo: float = np.nan
    hi_i: int = -1; lo_i: int = -1
    hi_open: bool = False; lo_open: bool = False
    prev_hi: float = np.nan; prev_lo: float = np.nan
    prev_hi_i: int = -1; prev_lo_i: int = -1
    leg_lo: float = np.nan; leg_lo_i: int = -1
    leg_hi: float = np.nan; leg_hi_i: int = -1
    trend: int = 0


@dataclass
class Pool:
    level: float; kind: str; index: int


@dataclass
class Zone:
    direction: int; top: float; bottom: float; index: int; kind: str


@dataclass
class PendingBreak:
    direction: int; level: float; entry: float; protected: float; risk: float
    index: int; choch: bool; fvg: bool; swept: bool; formula_p: float; model_p: float
    retest_done: bool = False; ft_done: bool = False


@dataclass
class PendingLiquidity:
    up: float; down: float; nearer_is_up: bool; index: int; formula_p: float; model_p: float


@dataclass
class Trade:
    direction: int; entry: float; initial_stop: float; stop: float; tp1: float; tp2: float
    index: int; method: str; tp1_hit: bool = False; open: bool = True
    exit_index: int | None = None; exit_price: float | None = None; result: str | None = None


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame.close.shift()
    tr = pd.concat([frame.high - frame.low, (frame.high - previous).abs(),
                    (frame.low - previous).abs()], axis=1).max(axis=1)
    # Pine ta.atr is Wilder's RMA.
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _pivot(values: np.ndarray, i: int, length: int, high: bool) -> float:
    """Value of a pivot confirmed on bar i, centred at i - length."""
    c = i - length
    if c < length or i >= len(values): return np.nan
    window = values[c - length:c + length + 1]
    value = values[c]
    return value if (value == (np.max(window) if high else np.min(window))) else np.nan


def _add_pool(pools: list[Pool], pool: Pool, side: int) -> None:
    pools.append(pool)
    pools.sort(key=lambda p: p.level, reverse=(side < 0))  # nearest-first by side
    del pools[20:]


def _sweep(pools: list[Pool], high: float, low: float, side: int) -> bool:
    taken = [p for p in pools if high > p.level] if side == 1 else [p for p in pools if low < p.level]
    pools[:] = [p for p in pools if p not in taken]
    return any(p.kind == "equal" for p in taken)


def _step_structure(s: Structure, h: np.ndarray, l: np.ndarray, c: np.ndarray, i: int,
                    pivot_h: float, pivot_l: float) -> dict:
    """Direct equivalent of f_step; pivot values are only known at confirmation."""
    event = {"direction": 0, "choch": False, "level": np.nan, "protected": np.nan,
             "protected_i": -1, "swept": False, "new_hi": False, "new_lo": False}
    if np.isfinite(s.leg_lo) and l[i] < s.leg_lo: s.leg_lo, s.leg_lo_i = l[i], i
    if np.isfinite(s.leg_hi) and h[i] > s.leg_hi: s.leg_hi, s.leg_hi_i = h[i], i
    if s.hi_open and c[i] > s.hi:
        event.update(direction=1, choch=s.trend == -1, level=s.hi,
                     protected=s.leg_lo if np.isfinite(s.leg_lo) else l[i],
                     protected_i=s.leg_lo_i if s.leg_lo_i >= 0 else i)
        # Latest previous low standing before protected low.
        ref = s.lo if s.lo_i < event["protected_i"] else s.prev_lo
        event["swept"] = np.isfinite(ref) and event["protected"] < ref
        s.trend, s.hi_open = 1, False
    elif s.lo_open and c[i] < s.lo:
        event.update(direction=-1, choch=s.trend == 1, level=s.lo,
                     protected=s.leg_hi if np.isfinite(s.leg_hi) else h[i],
                     protected_i=s.leg_hi_i if s.leg_hi_i >= 0 else i)
        ref = s.hi if s.hi_i < event["protected_i"] else s.prev_hi
        event["swept"] = np.isfinite(ref) and event["protected"] > ref
        s.trend, s.lo_open = -1, False
    if np.isfinite(pivot_h):
        s.prev_hi, s.prev_hi_i = s.hi, s.hi_i; s.hi, s.hi_i, s.hi_open = pivot_h, i-s.length, True
        look = l[i-s.length+1:i+1]; off = int(np.argmin(look)); s.leg_lo, s.leg_lo_i = look[off], i-s.length+1+off
        event["new_hi"] = True
    if np.isfinite(pivot_l):
        s.prev_lo, s.prev_lo_i = s.lo, s.lo_i; s.lo, s.lo_i, s.lo_open = pivot_l, i-s.length, True
        look = h[i-s.length+1:i+1]; off = int(np.argmax(look)); s.leg_hi, s.leg_hi_i = look[off], i-s.length+1+off
        event["new_lo"] = True
    return event


def run_smc_ml(data: pd.DataFrame, settings: Settings = Settings()) -> dict[str, pd.DataFrame]:
    """Run the translated indicator over chronological OHLC data.

    Output `bars` includes trend and live probabilities. `breaks`, `signals`,
    `zones`, `pools`, and `trades` are suitable inputs for matplotlib/plotly.
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(data.columns): raise ValueError(f"Missing OHLC columns: {required-set(data.columns)}")
    df = data.copy().reset_index(names="time") if data.index.name or not isinstance(data.index, pd.RangeIndex) else data.copy()
    df.columns = [str(x).lower() for x in df.columns]
    for col in required: df[col] = pd.to_numeric(df[col], errors="raise")
    o,h,l,c = (df[x].to_numpy(float) for x in ("open","high","low","close")); n=len(df)
    atr = _atr(df).to_numpy(); sd = pd.Series(c).diff().rolling(100, min_periods=100).std(ddof=0).to_numpy()
    sw, internal = Structure(settings.swing_length), Structure(settings.internal_length)
    cr, cd = Calibration(cut1=.70,cut2=.90), Calibration(True,.65,.80)
    up: list[Pool]=[]; down: list[Pool]=[]; zones: list[Zone]=[]; pending: list[PendingBreak]=[]; lpending: list[PendingLiquidity]=[]
    breaks: list[dict]=[]; signals: list[dict]=[]; completed: list[Trade]=[]; trades: list[Trade]=[]; out=[]
    retest_arm: dict | None = None
    last_fvg_bull = last_fvg_bear = -10**9
    tp1r,tp2r = sorted((settings.tp1_r,settings.tp2_r))

    def close_trade(t: Trade, i: int, price: float, result: str) -> None:
        t.open=False; t.exit_index=i; t.exit_price=price; t.result=result; completed.append(t)

    for i in range(n):
        atr_ok=np.isfinite(atr[i]) and atr[i]>0
        # exits precede every new break/signal, precisely as in the Pine code
        if trades and trades[-1].open and i > trades[-1].index:
            t=trades[-1]; long=t.direction==1
            hit_sl=(l[i]<=t.stop) if long else (h[i]>=t.stop)
            hit_t2=(h[i]>=t.tp2) if long else (l[i]<=t.tp2)
            hit_t1=not t.tp1_hit and ((h[i]>=t.tp1) if long else (l[i]<=t.tp1))
            if hit_sl:
                px=min(o[i],t.stop) if long else max(o[i],t.stop)
                close_trade(t,i,px,"TP1, then stop at entry" if t.tp1_hit and t.stop==t.entry else "TP1, then SL" if t.tp1_hit else "SL")
            elif hit_t2:
                t.tp1_hit=True; close_trade(t,i,max(o[i],t.tp2) if long else min(o[i],t.tp2),"TP2")
            else:
                if hit_t1:
                    t.tp1_hit=True
                    if settings.stop_to_entry_after_tp1: t.stop=t.entry
                if i-t.index >= settings.trade_timeout: close_trade(t,i,c[i],"timeout")
        # Resolve already-recorded outcomes, starting on their next bar.
        for b in pending[:]:
            if i<=b.index: continue
            long=b.direction==1
            touched = (l[i] <= b.level) if long else (h[i] >= b.level)
            if not b.retest_done and (touched or i-b.index >= settings.retest_window):
                b.retest_done=True
                cr.score(b.model_p,b.formula_p,int(touched))
                match_b = next((breaks[j] for j in range(len(breaks)-1,-1,-1) if breaks[j]["index"]==b.index), None)
                if match_b is not None:
                    match_b["retest"]=int(touched)
            if not b.ft_done:
                fail=(c[i]<b.protected) if long else (c[i]>b.protected); win=(c[i]>=b.entry+b.risk) if long else (c[i]<=b.entry-b.risk)
                if fail or win or i-b.index>=FT_WIN: b.ft_done=True
            if b.retest_done and b.ft_done: pending.remove(b)
        for w in lpending[:]:
            if i<=w.index: continue
            hu,hd=h[i]>w.up,l[i]<w.down
            if hu or hd or i-w.index>=DRAW_WIN:
                if hu != hd: cd.score(w.model_p,w.formula_p,int(hu==w.nearer_is_up))
                lpending.remove(w)
        # FVG status uses bar i and the two prior bars.
        bull_fvg=bear_fvg=False
        if i>=2 and atr_ok:
            bull_fvg=l[i]>h[i-2] and c[i-1]>h[i-2] and l[i]-h[i-2]>=settings.fvg_min_atr*atr[i]
            bear_fvg=h[i]<l[i-2] and c[i-1]<l[i-2] and l[i-2]-h[i]>=settings.fvg_min_atr*atr[i]
            if bull_fvg: last_fvg_bull=i
            if bear_fvg: last_fvg_bear=i
            if bull_fvg: zones.append(Zone(1, l[i], h[i-2], i-2, "fvg"))
            if bear_fvg: zones.append(Zone(-1, l[i-2], h[i], i-2, "fvg"))
        zones[:]=[z for z in zones if not ((z.kind=="order_block" and ((z.direction==1 and c[i]<z.bottom) or (z.direction==-1 and c[i]>z.top))) or (z.kind=="fvg" and ((z.direction==1 and l[i]<=z.bottom) or (z.direction==-1 and h[i]>=z.top))))]
        _sweep(up,h[i],l[i],1); _sweep(down,h[i],l[i],-1)
        phs=_pivot(h,i,settings.swing_length,True); pls=_pivot(l,i,settings.swing_length,False)
        phi=_pivot(h,i,settings.internal_length,True); pli=_pivot(l,i,settings.internal_length,False)
        se=_step_structure(sw,h,l,c,i,phs,pls); ie=_step_structure(internal,h,l,c,i,phi,pli)
        if atr_ok:
            if se["new_hi"]: _add_pool(up,Pool(sw.hi,"swing",sw.hi_i),1)
            if se["new_lo"]: _add_pool(down,Pool(sw.lo,"swing",sw.lo_i),-1)
            if ie["new_hi"] and np.isfinite(internal.prev_hi) and abs(internal.hi-internal.prev_hi)<=settings.equal_tolerance_atr*atr[i]: _add_pool(up,Pool(max(internal.hi,internal.prev_hi),"equal",internal.prev_hi_i),1)
            if ie["new_lo"] and np.isfinite(internal.prev_lo) and abs(internal.lo-internal.prev_lo)<=settings.equal_tolerance_atr*atr[i]: _add_pool(down,Pool(min(internal.lo,internal.prev_lo),"equal",internal.prev_lo_i),-1)
        signal=None
        if atr_ok and se["direction"]:
            d=se["direction"]; long=d==1; risk=abs(c[i]-se["protected"]); p0=_touch(abs(c[i]-se["level"]),sd[i],settings.retest_window); pm=cr.probability(p0)
            record={"index":i,"direction":d,"choch":se["choch"],"level":se["level"],"protected":se["protected"],"formula_retest_odds":p0,"retest_odds":pm,"retest":None}
            breaks.append(record)
            if risk>0: pending.append(PendingBreak(d,se["level"],c[i],se["protected"],risk,i,se["choch"],(last_fvg_bull if long else last_fvg_bear)>=se["protected_i"],se["swept"],p0,pm))
            if up and down and up[0].level>c[i]>down[0].level and np.isfinite(sd[i]) and sd[i]>0:
                du,dd=up[0].level-c[i],c[i]-down[0].level
                if du+dd<=sd[i]*sqrt(DRAW_WIN):
                    nearer_up=du<=dd; q0=_clamp_p((dd if nearer_up else du)/(du+dd)); lpending.append(PendingLiquidity(up[0].level,down[0].level,nearer_up,i,q0,cd.probability(q0)))
            # Last opposite candle at/after the protected extreme, within 3 bars.
            back = min(max(i-se["protected_i"], 0), 400)
            candidates = range(back, min(back + 3, i) + 1)
            ob = next((i-k for k in candidates if ((c[i-k] < o[i-k]) if long else (c[i-k] > o[i-k]))), se["protected_i"])
            if h[ob]-l[ob]<=3*atr[i]: zones.append(Zone(d,h[ob],l[ob],ob,"order_block"))
            zones[:]=[z for z in zones if z.kind!="order_block"] + [z for side in (1,-1) for z in [x for x in zones if x.kind=="order_block" and x.direction==side][-settings.order_blocks_per_side:]]
            qualifies=settings.signal_breaks=="CHoCH + BOS" or (settings.signal_breaks=="CHoCH only" and se["choch"]) or (settings.signal_breaks=="BOS only" and not se["choch"])
            wait=settings.entry=="Level Retest" or (settings.entry=="Adaptive" and np.isfinite(pm) and pm>=settings.adaptive_threshold)
            if qualifies:
                if wait: retest_arm={"d":d,"bar":i,"level":se["level"],"protected":se["protected"]}
                else: signal=(d,c[i],se["protected"]-settings.stop_buffer_atr*atr[i] if long else se["protected"]+settings.stop_buffer_atr*atr[i],"break close")
        if retest_arm and i>retest_arm["bar"] and signal is None and atr_ok:
            d=retest_arm["d"]; long=d==1
            if i-retest_arm["bar"]>settings.retest_window or ((c[i]<retest_arm["protected"]) if long else (c[i]>retest_arm["protected"])): retest_arm=None
            elif ((l[i]<=retest_arm["level"] and c[i]>retest_arm["level"]) if long else (h[i]>=retest_arm["level"] and c[i]<retest_arm["level"])):
                signal=(d,c[i],retest_arm["protected"]-settings.stop_buffer_atr*atr[i] if long else se["protected"]+settings.stop_buffer_atr*atr[i],"level retest"); retest_arm=None
        if signal:
            d,en,stop,method=signal
            if abs(en-stop)<=settings.max_risk_atr*atr[i] and en!=stop and (not trades or not trades[-1].open or trades[-1].direction!=d):
                if trades and trades[-1].open: close_trade(trades[-1],i,en,"reversed")
                r=abs(en-stop); t=Trade(d,en,stop,stop,en+d*tp1r*r,en+d*tp2r*r,i,method); trades.append(t); signals.append({"index":i,"direction":d,"entry":en,"stop":stop,"tp1":t.tp1,"tp2":t.tp2,"method":method})
        live_up=up[0].level if up and up[0].level>c[i] else np.nan; live_down=down[0].level if down and down[0].level<c[i] else np.nan
        p_up=np.nan
        if np.isfinite(live_up) and np.isfinite(live_down):
            du,dd=live_up-c[i],c[i]-live_down; nearer=du<=dd; q=cd.probability(_clamp_p((dd if nearer else du)/(du+dd))); p_up=q if nearer else 1-q
        out.append({"trend":sw.trend,"atr":atr[i],"retest_calibration_n":cr.n,"liquidity_calibration_n":cd.n,"pool_above":live_up,"pool_below":live_down,"odds_pool_above_first":p_up})
    out_df = pd.DataFrame(out)
    df_base = df.drop(columns=[col for col in out_df.columns if col in df.columns], errors="ignore")
    bars = pd.concat([df_base, out_df], axis=1)
    trade_rows=[]
    for t in completed:
        r=abs(t.entry-t.initial_stop); raw=t.direction*(t.exit_price-t.entry)/r
        trade_rows.append({**asdict(t),"r_multiple":.5*tp1r+.5*raw if t.tp1_hit else raw})
    return {"bars":bars,"breaks":pd.DataFrame(breaks),"signals":pd.DataFrame(signals),"trades":pd.DataFrame(trade_rows),"zones":pd.DataFrame([asdict(z) for z in zones]),"pools":pd.DataFrame([asdict(p) | {"side":"above"} for p in up]+[asdict(p) | {"side":"below"} for p in down])}


@dataclass
class SMCSignalSnapshot:
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry: float
    stop: float  # SL Exit
    tp1: float   # Target 1 Exit
    tp2: float   # Target 2 Exit
    risk_reward: float
    risk_amount: float
    risk_pct: float
    tp1_pct: float
    tp2_pct: float
    method: str
    signal_bar_index: int
    signal_time: str
    trend: str      # "BULLISH" (1) or "BEARISH" (-1)
    atr: float
    order_blocks_count: int
    fvgs_count: int
    liquidity_above: float
    liquidity_below: float
    odds_pool_above: float
    raw_result: dict = field(default_factory=dict)


def analyze_smc_stock(
    symbol: str,
    df: pd.DataFrame,
    current_price: Optional[float] = None,
    settings: Settings = Settings()
) -> SMCSignalSnapshot:
    """Convenience evaluator that executes run_smc_ml and extracts the definitive BUY or SELL
    recommendation with precise Entry, Stop Loss (Exit), and Take Profit (Exit) levels.
    """
    clean_sym = symbol.strip().upper()
    df_clean = df.copy()

    # Standardize column names
    col_map = {c: str(c).lower() for c in df_clean.columns}
    df_clean.rename(columns=col_map, inplace=True)

    # Ensure required columns
    for req in ["open", "high", "low", "close"]:
        if req not in df_clean.columns:
            raise ValueError(f"Missing required column '{req}' for {clean_sym}")

    res = run_smc_ml(df_clean, settings=settings)
    bars_df = res["bars"]
    signals_df = res["signals"]
    zones_df = res["zones"]
    breaks_df = res["breaks"]

    last_bar = bars_df.iloc[-1]
    
    def _scalar(val, default=0.0):
        if isinstance(val, (pd.Series, np.ndarray, list)):
            val = val.iloc[-1] if hasattr(val, "iloc") else val[-1]
        try:
            f = float(val)
            return f if np.isfinite(f) else default
        except Exception:
            return default

    curr_px = float(current_price if (current_price is not None and current_price > 0) else _scalar(last_bar["close"], 100.0))
    last_atr = _scalar(last_bar.get("atr"), curr_px * 0.01)
    if last_atr <= 0:
        last_atr = curr_px * 0.01

    trend_val = int(_scalar(last_bar.get("trend"), 0))
    trend_str = "BULLISH" if trend_val == 1 else ("BEARISH" if trend_val == -1 else "SIDEWAYS")

    # Time resolution
    time_col = None
    for cand in ["time", "timestamp", "datetime", "date"]:
        if cand in bars_df.columns:
            time_col = cand
            break

    # Determine BUY or SELL signal
    # If explicit signals were generated during the run, pick the most recent relevant signal.
    # Otherwise, derive the structural signal from current SMC trend and market structure.
    if not signals_df.empty:
        last_sig = signals_df.iloc[-1]
        sig_dir_num = int(last_sig["direction"])
        direction = "BUY" if sig_dir_num == 1 else "SELL"
        entry = float(last_sig["entry"])
        stop = float(last_sig["stop"])
        tp1 = float(last_sig["tp1"])
        tp2 = float(last_sig["tp2"])
        method = str(last_sig.get("method", "Break Close"))
        sig_idx = int(last_sig["index"])
        sig_time = str(bars_df.iloc[sig_idx][time_col]) if time_col else f"Bar {sig_idx}"
    else:
        # Fallback to structure trend: BUY if bullish, SELL if bearish or sideways
        sig_dir_num = 1 if trend_val >= 0 else -1
        direction = "BUY" if sig_dir_num == 1 else "SELL"
        entry = curr_px
        method = "Structure Flow"
        sig_idx = len(bars_df) - 1
        sig_time = str(last_bar[time_col]) if time_col else f"Bar {sig_idx}"

        # Standard SMC stop buffer calculation
        # If bullish: SL below latest swing low / protected level
        # If bearish: SL above latest swing high / protected level
        recent_low = float(df_clean["low"].tail(15).min())
        recent_high = float(df_clean["high"].tail(15).max())

        if direction == "BUY":
            stop = round(max(recent_low - settings.stop_buffer_atr * last_atr, curr_px * 0.95), 2)
            risk = max(abs(entry - stop), last_atr * 0.5)
            tp1 = round(entry + settings.tp1_r * risk, 2)
            tp2 = round(entry + settings.tp2_r * risk, 2)
        else:
            stop = round(min(recent_high + settings.stop_buffer_atr * last_atr, curr_px * 1.05), 2)
            risk = max(abs(entry - stop), last_atr * 0.5)
            tp1 = round(entry - settings.tp1_r * risk, 2)
            tp2 = round(entry - settings.tp2_r * risk, 2)

    risk_amount = abs(entry - stop)
    risk_pct = (risk_amount / entry * 100.0) if entry > 0 else 0.0
    tp1_pct = ((tp1 - entry) / entry * 100.0) if entry > 0 else 0.0
    tp2_pct = ((tp2 - entry) / entry * 100.0) if entry > 0 else 0.0
    risk_reward = round(abs(tp2 - entry) / risk_amount, 2) if risk_amount > 0 else 2.0

    # Count zones
    ob_count = len(zones_df[zones_df["kind"] == "order_block"]) if not zones_df.empty else 0
    fvg_count = len(zones_df[zones_df["kind"] == "fvg"]) if not zones_df.empty else 0

    return SMCSignalSnapshot(
        symbol=clean_sym,
        direction=direction,
        entry=round(entry, 2),
        stop=round(stop, 2),
        tp1=round(tp1, 2),
        tp2=round(tp2, 2),
        risk_reward=risk_reward,
        risk_amount=round(risk_amount, 2),
        risk_pct=round(risk_pct, 2),
        tp1_pct=round(tp1_pct, 2),
        tp2_pct=round(tp2_pct, 2),
        method=method,
        signal_bar_index=sig_idx,
        signal_time=sig_time,
        trend=trend_str,
        atr=round(last_atr, 2),
        order_blocks_count=ob_count,
        fvgs_count=fvg_count,
        liquidity_above=_scalar(last_bar.get("pool_above"), np.nan),
        liquidity_below=_scalar(last_bar.get("pool_below"), np.nan),
        odds_pool_above=_scalar(last_bar.get("odds_pool_above_first"), np.nan),
        raw_result=res
    )
