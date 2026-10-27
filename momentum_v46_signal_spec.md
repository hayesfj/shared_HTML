# Momentum v46 — BUY / SELL / EXIT / HOLD Signal Specification (for back-testing)

**Source:** `momentum_unified_v46.py` (Google Drive, modified 2026-10-27). Every rule below was read from that file's `Config`, `compute_indicator_frame`, `add_composites`, `squeeze_score_frame`, `score_frame`, `_apply_gates`, `apply_gates_history` and `scorecard`.
**Purpose:** a precise, implementable definition of every signal state so it can be re-implemented (or the original code driven) and back-tested.
**Convention:** *"As coded"* sections describe exactly what v46 does. *"Backtest recommendation"* sections are my suggestions and are **not** in the code.

---

## 0. TL;DR decision table

Evaluated **per ticker, per daily bar, on the closed bar**. Output states: `BUY`, `SELL`, `EXIT`, `HOLD` (default).

| State | Requires (all of) |
|---|---|
| **BUY** | (1) Buy score ≥ regime/ATR-adjusted threshold · (2) PPO histogram crossed **up** within last 3 bars **and** histogram is still > 0 · (3) bar confirmation (dollar-vol ≥ 1.25× 20d median **or** close in top 34% of bar range) · (4) liquidity floor · (5) data OK · (6) **not** blocked by post-score gates (SPY regime, relative strength, EMA200) unless exempt (Dormant→Waking + full squeeze setup) |
| **SELL** | (1) Sell score ≥ regime/ATR-adjusted threshold · (2) PPO histogram crossed **down** within last 3 bars **and** histogram is still < 0 · (3) bar confirmation (volume surge **or** close in bottom 34%) · (4) liquidity · (5) data OK · (6) **not** downgraded by the "stronger top" filter (SELL is turned into HOLD if close > EMA50 **and** blended momentum > 0) |
| **EXIT** | Both sides clear their thresholds with their bar/liquidity/data checks, **both** a fresh up-cross and a fresh down-cross sit inside the 3-bar window (a whipsaw), **and** \|Buy − Sell\| ≤ 10. Gates and the top filter do **not** apply to EXIT |
| **HOLD** | Everything else. The code is **stateless** — HOLD means "no alert today", not "keep the position". See §8 for sub-types and a proposed stateful definition |

Precedence: `EXIT` > `BUY` > `SELL` > `HOLD` (a BUY and SELL alert can never both be true on one bar because they need opposite histogram signs).

---

## 1. Pipeline order of operations (per ticker, per bar)

1. Load daily OHLCV (yfinance, auto-adjusted). Optional forming/live bar is appended (live use only — **not** for back-test).
2. Compute raw indicators for every bar (§2).
3. Compute causal percentile composites → **Trailing %ile**, **Emerging %ile** → **Regime** (§3).
4. Compute squeeze set-up score and flag (§7).
5. Compute **Buy score** and **Sell score** (§4, §5) — these use the Regime for weights.
6. Compute thresholds (regime table + ATR bump), alerts, `Recommendation` (BUY/SELL/EXIT/HOLD) and `Watch` / `Gate_Block`.
7. **Post-score gates** (§4.5): SPY regime, relative strength, EMA200 → blocked BUY becomes HOLD (`Watch="BUY"`).
8. **Stronger-top filter** (§5.4): weak SELL becomes HOLD.
9. Scorecard: forward returns after each *new* BUY/SELL (§10).

Since v46, steps 7–8 run over **every historical bar**, so history == what the dashboard shows. (Before v46 the history was ungated.)

---

## 2. Indicator definitions (exact)

| Name | Definition |
|---|---|
| `EMA(n)` | `close.ewm(span=n, adjust=False, min_periods=n)` |
| `RSI_14` | Wilder RSI, `alpha=1/14`, `min_periods=14`; 100 when avg-loss = 0 |
| `PPO` | `(EMA12 − EMA26) / EMA26 × 100` |
| `PPO_Signal` | `EMA9(PPO)` (min_periods 9) |
| `PPO_Delta` = `PPO_Hist` | `PPO − PPO_Signal` |
| Pos cross | `hist > 0` and prior `hist ≤ 0`. Neg cross: `hist < 0` and prior `hist ≥ 0` |
| `PPO_Bars_Since_Pos` / `_Neg` | bars since last pos / neg cross (0 on the cross bar; NaN before first cross → treated as 99) |
| `PPO_Bars_Since_Zero_Pos/Neg` | bars since `PPO` crossed above / below 0 |
| `PPO_Cross_Strength` | `abs(clip(hist / max(abs(PPO_Signal), 0.01), −1, 1))` |
| `PPO_Hist_Expand_Up` | `hist > 0 and hist > hist[t−1]`; `Expand_Dn`: `hist < 0 and hist < hist[t−1]` |
| `PPO_Above_Zero` | `PPO > 0` |
| Fresh bull / bear | `Bars_Since_Pos ≤ 3` / `Bars_Since_Neg ≤ 3` (`ppo_fresh_bars = 3`) |
| `EMA20/50/200`, `Below_EMA20/50/200` | `close < EMA` (**NaN EMA → False → treated as "above"**; see §11 warm-up trap) |
| `High_52W_Ratio` | `close / rolling_max(close, 252, min_periods=126)` |
| `ATR_Pct` | `Wilder ATR(14) / close × 100` |
| `Historical_Vol` | `std(1d pct return, 20) × √252 × 100` |
| `Signal_Reliability` (`rel`) | `1 − (1 − 0.45) × clip(HV/100, 0, 1)` → range 0.45…1.00 |
| `Mom_{20,50,100,252}d` | `pct_change(n) × weight` with weights 0.40 / 0.30 / 0.20 / 0.10 |
| `Momentum_Blended` | sum of the four weighted layers; `Momentum_Direction = sign(Momentum_Blended)` |
| `ADX_Trend` | Wilder ADX(14); `DMI_Value = +DI − −DI` |
| `OBV_slope` | OLS slope over 20 bars of `OBV / 20d-avg-volume` |
| `Trend_Persistence` | 20-bar rolling correlation of daily return with its 1-bar lag, clipped ±1 |
| `Dollar_Volume_20d` | rolling 20 median (min 15) of `close × volume` |
| `Vol_Ratio` | `Dollar_Volume / Dollar_Volume_20d` (that bar's dollar-vol vs the 20d median, **includes** that bar in the median) |
| `Close_Loc` | `(close − low)/(high − low)`, clipped 0–1 |
| `Ret_20d` | `close.pct_change(20) × 100` |
| `RS_20d` | `Ret_20d(ticker) − Ret_20d(SPY)` on the same date (percentage points) |
| `RSI_Zone` | `<30` Oversold · `30–<40` Weak · `40–65` Healthy · `>65–75` Overheated · `>75` Extreme |

**Informational only, never used in signals:** Swenlin PMO (`PMO`, `PMO_Signal`), theme scores, Rotation/Six Pack/Sector/Country/Currency tab displays. Do not back-test these as signals.

---

## 3. Regime (the dependency that drives everything else)

### 3.1 Percentile composites (causal)
Each component is converted to a percentile against **SPY's own component history over the 504 bars strictly before the scored date** (`pctile_lookback = 504`, `pctile_min_obs = 50`; fewer than 50 obs → NaN → treated as 50 in the regime step).

| Composite | Components (weight) |
|---|---|
| **Trailing %ile** `t` | `Mom_20d` 0.35 · `Mom_50d` 0.25 · `Mom_252d` 0.20 · `ADX_Trend` 0.10 · `High_52W_Ratio` 0.10 |
| **Emerging %ile** `e` | `PPO_Delta` 0.25 · `DMI_Value` 0.20 · `OBV_slope` 0.15 · `Momentum_Blended` 0.20 · `Trend_Persistence` 0.20 |
| `Momentum_Pctile` | `0.60·t + 0.40·e` |

Weights are re-normalised per row over components that exist (young tickers).
**Currency ETFs** (FXE, FXY, FXB, FXF, FXA, FXC, FXS, FXM, UUP) are percentiled against the *pooled history of that group* (lookback scaled by group size) instead of SPY.

### 3.2 Regime rule (first match wins, in this order)

| # | Regime | Condition (`hi=55`, `lo=45`, `gap=10`) | Meaning |
|---|---|---|---|
| 1 | **Strong** | `t ≥ 55` and `e ≥ 55` | trend + acceleration |
| 2 | **Weak** | `t < 45` and `e < 45` | no trend, no thrust |
| 3 | **Dormant→Waking** | `e > t + 10` | coiled / early |
| 4 | **Mature/Cooling** | `t > e + 10` | strong on paper, losing thrust |
| 5 | **Neutral** | none of the above | |

`Regime_Confidence` is informational (not used in signals).

### 3.3 Everything that changes by regime

| Parameter | Strong | Dormant→Waking | Neutral | Mature/Cooling | Weak |
|---|---|---|---|---|---|
| **Buy threshold** | 35 | 35 | 40 | 50 | 55 |
| **Sell threshold** | 50 | 50 | 40 | 35 | 35 |
| Buy PPO weight `wp` | 1.0 | 1.2 | 1.0 | 0.8 | 1.5 |
| Buy RSI weight `wr` | 0.6 | 0.7 | 0.8 | 0.9 | 0.4 |
| Buy EMA weight `we` | 0.7 | 0.8 | 0.9 | 1.1 | 1.2 |
| Sell PPO weight `sp` (floored at 0.85) | 0.6→**0.85** | 0.7→**0.85** | 0.8→**0.85** | 1.1 | 0.9 |
| Sell RSI weight `sr` | 1.0 | 0.9 | 0.9 | 1.2 | 1.0 |
| Sell EMA weight `se` | 0.8 | 0.7 | 0.9 | 1.0 | 1.1 |
| Exempt from BUY gates? | no | **yes**, if `Squeeze_Setup_Flag = 1` | no | no | no |
| Counts as "bad market" when it is **SPY's** regime? | no | no | no | **yes** | **yes** |
| Regime-specific EMA exception (below EMA200) | no | **yes** (needs squeeze room + release) | no | no | no |

Tuple order in code is `(PPO, RSI, EMA)`: BUY `Strong (1.0,0.6,0.7)`, `Dormant (1.2,0.7,0.8)`, `Neutral (1.0,0.8,0.9)`, `Mature (0.8,0.9,1.1)`, `Weak (1.5,0.4,1.2)`; SELL `Strong (0.6,1.0,0.8)`, `Dormant (0.7,0.9,0.7)`, `Neutral (0.8,0.9,0.9)`, `Mature (1.1,1.2,1.0)`, `Weak (0.9,1.0,1.1)`.

**Design intent visible in the numbers:** BUYs are easy in Strong/Dormant (35) and hard in Weak/Mature (55/50); SELLs are the mirror image. Backtest this explicitly (§10).

---

## 4. BUY — full criteria

### 4.1 Buy score (0–100; theoretical max ≈ 83)

`Buy = min(PPO + RSI + EMA + Persistence + Squeeze, 100)`

**a) PPO-cross component (0–35)**
```
fresh   = max(0, 1 − Bars_Since_Pos / 5)          # 1.0, .8, .6, .4 at bars 0..3; 0 at 5+
base    = 18 · fresh
exp     = 8 · Hist_Expand_Up · max(fresh, 0.4)
zero    = 9 if PPO crossed above 0 within 3 bars; else 4 if PPO > 0; else 0
str     = 1 + 0.5 · PPO_Cross_Strength
B_ppo   = min( floor( (base + exp + zero) · str · wp · rel ), 35 )
```

**b) RSI component (0–18)**
```
rsi_mult : Oversold 1.15 · Weak 1.15 · Healthy 1.00 · Overheated 0.70 · Extreme 0.00
tag      : Oversold 6 · Weak 4 · Healthy 2 · else 0
B_rsi    = floor( clip( (rsi_mult − 1)·max(B_ppo, 12) + tag·wr·rel , 0, 18 ) )
```

**c) EMA component (0–12)**
```
cnt    = (close>EMA20) + (close>EMA50) + (close>EMA200)
B_ema  = floor(12 · cnt/3 · we)
if PPO_Delta < 0:                       B_ema = floor(B_ema · 0.5)
if Below_EMA200 and NOT(regime == Dormant→Waking and Room>0 and Release>0):
                                         B_ema = floor(B_ema · 0.35)
if High_52W_Ratio > 0.97 and Release == 0:
                                         B_ema = floor(B_ema · 0.6)        # crowded highs
```
(applied in that order)

**d) Persistence (0–8)**: `B_per = floor(8 · max(Trend_Persistence,0) · rel)` only if `Momentum_Direction > 0` **or** `PPO_Delta > 0`, else 0.

**e) Squeeze add-on (0–10, cap)**: `min(10, 6·[Release] + 3·[PPO hist > 0] + 3·[Room])` where Release = squeeze fired within last 10 bars and not currently on; Room = `High_52W_Ratio < 0.90`.

**f) RSI-Extreme cap:** if `RSI > 75`, `Buy = min(Buy, BuyThreshold − 1)` → an Extreme-RSI name can never alert BUY.

### 4.2 Buy threshold
```
BuyThreshold = BUY_THR_BY_REGIME[regime]  (35/35/40/50/55)
             + 10 if ATR_Pct > 6.0
```

### 4.3 Buy trigger and bar checks (all required)
- `Buy ≥ BuyThreshold`
- **PPO trigger:** `Bars_Since_Pos ≤ 3` **and** `PPO_Delta > 0` (the v46 fix: a cross that has since rolled back under its signal line is not a trigger)
- **Bar confirmation:** `Vol_Ratio ≥ 1.25` **or** `Close_Loc ≥ 0.66`
- **Liquidity:** `Dollar_Volume_20d ≥ $5M` (≥ $1M for Currency and Country-ETF groups)
- **Data OK:** `Momentum_Data_OK == 1` (see §11 — not point-in-time as coded)

### 4.4 → 4.5 Post-score gates (BUY → HOLD, `Watch="BUY"`, reason in `Gate_Block`)

Applied only to rows that are BUY after §4.3, **excluding the benchmark ticker (SPY)**. Each gate can block independently; all reasons are concatenated.

| Gate | Blocks when | Skipped for |
|---|---|---|
| **SPY regime** | On that date SPY's *pre-gate* recommendation was `SELL`, **or** SPY's regime is `Weak` or `Mature/Cooling` | Currency ETFs, Country ETFs |
| **Relative strength** | `RS_20d < 0.0` (NaN treated as 0 → passes) | Currency ETFs |
| **Below EMA200** | `Below_EMA200 == 1` | — |

**Universal exemption:** `Regime == Dormant→Waking` **and** `Squeeze_Setup_Flag == 1` bypasses all three gates.

---

## 5. SELL — full criteria

### 5.1 Sell score (0–100; theoretical max ≈ 81)
`Sell = s_ppo + s_rsi + s_ema + s_persist + s_sqz`, then the cross floor, then cap 100.

**a) PPO-cross (0–35)**
```
fresh_n = max(0, 1 − Bars_Since_Neg / 5)
base    = 18 · fresh_n
exp     = 8 · Hist_Expand_Dn · max(fresh_n, 0.4)
zero    = 9 if PPO crossed below 0 within 3 bars; else 0 if PPO > 0; else 4
rel_x   = 1 − 0.35 · (1 − rel)                  # vol damping is gentler on sells
S_ppo   = min( floor( (base + exp + zero) · (1+0.5·strength) · max(sp,0.85) · rel_x ), 35 )
```

**b) RSI (0–18)**
```
rsi_mult : Extreme 1.15 · Overheated 1.15 · Healthy 1.00 · Weak 0.85 · Oversold 0.65
           override: Oversold AND close > EMA50 → 0.40
tag      : Extreme 8 · Overheated 5 · else 0
S_rsi    = floor( clip( (rsi_mult − 1)·max(S_ppo,12) + tag·sr·rel, 0, 18 ) )
```

**c) EMA (0–12):** `floor(12 · (Below20+Below50+Below200)/3 · se)`, halved (floor) if `PPO_Delta > 0`.
**d) Weakness/persistence (0–8):** `floor(8·max(Trend_Persistence,0)·rel)` only if `Momentum_Direction < 0` or `PPO_Delta < 0`.
**e) Failed-squeeze add-on (0 or 8):** `(Squeeze_On and Squeeze_Mom < 0)` **or** `(Release and PPO_Delta < 0 and Squeeze_Mom < 0)`.

### 5.2 Cross floor
If `Bars_Since_Neg ≤ 3` then `Sell = max(Sell, 30)` (`sell_cross_floor`). v46 lowered this from 35 → 30 so that a cross **alone** is below every regime sell threshold (lowest is 35) and needs ≥ 1 confirming factor.

### 5.3 Sell threshold
`SellThreshold = SELL_THR_BY_REGIME[regime] (50/50/40/35/35) + 10 if ATR_Pct > 6.0`

### 5.4 Sell trigger, bar checks, post filter
- `Sell ≥ SellThreshold`
- **PPO trigger:** `Bars_Since_Neg ≤ 3` **and** `PPO_Delta < 0`
- **Bar confirmation:** `Vol_Ratio ≥ 1.25` **or** `Close_Loc ≤ 0.34`
- Liquidity and data checks as for BUY
- **Stronger-top filter (post-score):** a SELL where `close > EMA50` **and** `Momentum_Direction > 0` is converted to **HOLD** (`Sell_Alert = 0`; `Watch` is *not* re-set).
  *Note:* there is **no** SPY/RS/EMA200 gate on the sell side; `require_negative_rs_for_sell = False`.

---

## 6. EXIT — full criteria (whipsaw / de-risk)

```
buy_base  = Buy  ≥ BuyThr  and data_ok and liq_ok and buy_bar_ok
sell_base = Sell ≥ SellThr and data_ok and liq_ok and sell_bar_ok
EXIT      = buy_base and sell_base
            and Bars_Since_Pos ≤ 3 and Bars_Since_Neg ≤ 3     # both a fresh up- and down-cross in window
            and |Buy − Sell| ≤ 10
```
Both `Buy_Alert` and `Sell_Alert` are set to 1. EXIT is **not** gated and **not** touched by the top filter. The scorecard does not measure EXIT — add it yourself.

---

## 7. Squeeze Setup (it feeds the Buy score and the gate exemption, so it must be reproduced)

**Raw squeeze state**
- `Squeeze_On`: Bollinger(20, 2σ, ddof=0) fully **inside** Keltner(EMA20 ± 1.5·EMA20(TR)).
- `Squeeze_Off`: Bollinger fully **outside** Keltner (strict).
- `Squeeze_Fired`: On on the previous bar **and** Off on this bar.
- `Squeeze_Recent_Fire`: a Fired within the last 10 bars **and** On == 0 now.
- `Squeeze_Prior_Length`: length of the current On-run if On, else length of the last completed On-run.
- `Squeeze_Mom`: last point of a 12-bar OLS line through `close − basis`, `basis = ((HH12+LL12)/2 + SMA12)/2`.

**Pillar points (raw max 85; `Squeeze_Setup_Strength = round(100·raw/85)`)**

| Pillar | Item | Points |
|---|---|---|
| Foundation (≤20) | `Prior_Length` ≥ 50 / ≥ 30 / ≥ 15 | 10 / 7 / 4 |
| | `Recent_Fire` | +10 |
| Momentum (≤30) | `PPO_Delta > 0` | 15 |
| | `DMI_Value > −5` | 15 |
| Condition (≤20) | `High_52W_Ratio < 0.90` ("room") | 10 |
| | `40 ≤ RSI ≤ 65` | 10 |
| Confirmation (≤15) | `OBV_slope > 0` | 8 |
| | `20 ≤ Momentum_Pctile ≤ 80` | 7 |

**`Squeeze_Setup_Flag = 1`** requires Foundation ≥ 14, Momentum ≥ 20, Condition ≥ 15, Confirmation ≥ 10. Working through the point values, that is equivalent to **all eight** being true: base ≥ 15 bars, recent fire, PPO hist > 0, DMI > −5, < 90% of 52-wk high, RSI 40–65, OBV slope > 0, momentum %ile 20–80.

---

## 8. HOLD — what it means and how to test it

### 8.1 As coded
`HOLD` is the default (`np.select(..., default="HOLD")`); there is **no position state** in the code. A name that was BUY yesterday and is HOLD today has simply produced no new alert.

### 8.2 HOLD sub-types (all derivable from the history columns — report forward returns for each)

| Sub-type | How to identify | What it tells you |
|---|---|---|
| Plain HOLD | `Recommendation=="HOLD"` and `Watch==""` and `Gate_Block==""` | no setup |
| Watch-BUY | `Watch=="BUY"` and `Gate_Block==""` | Buy score within 8 pts of threshold, no alert (`near_buy = ~buy_alert & Buy ≥ BuyThr−8 & data_ok`) |
| Watch-SELL | `Watch=="SELL"` | same for sells |
| Watch-BOTH | `Watch=="BOTH"` | both near |
| **Gate-blocked BUY** | `Watch=="BUY"` and `Gate_Block` contains `SPY regime` / `Relative strength` / `Below EMA200` | The gates' value-add: test whether blocked BUYs underperform allowed ones |
| Trigger/bar-blocked | `Gate_Block` in `No fresh cross`, `Volume bar`, `Liquidity`, `Short data` (possibly joined with ` + `) | score cleared but a hard requirement failed |
| Top-filtered SELL | previously SELL-qualifying, now HOLD (not flagged by a column — reconstruct by running with the filter off) | value of the stronger-top filter |

### 8.3 Backtest recommendation — stateful HOLD
To evaluate holding behaviour as a trading rule (not only as an event study), define a position state machine:

```
FLAT  --BUY-->   LONG
LONG  --SELL-->  FLAT     (exit at next open)
LONG  --EXIT-->  FLAT     (de-risk)
LONG  --HOLD-->  LONG     (keep)
FLAT  --HOLD/SELL/EXIT--> FLAT
optional: time stop N bars, ATR trailing stop, regime-change exit (see §10)
```
SELL/EXIT are *exit-from-long* signals in this long-only design; a short-side test is optional and would be a separate hypothesis.

---

## 9. Reference pseudocode (vectorised, columns as in the history DataFrame)

```python
# --- thresholds
buy_thr  = regime.map(BUY_THR_BY_REGIME)  + 10*(ATR_Pct > 6)
sell_thr = regime.map(SELL_THR_BY_REGIME) + 10*(ATR_Pct > 6)
buy      = where(RSI_Zone == "Extreme", minimum(buy, buy_thr-1), buy)

# --- hard requirements
liq_ok     = Dollar_Volume_20d >= floor_by_group(Ticker)          # 5M, or 1M for FX/country ETFs
buy_bar    = (Vol_Ratio >= 1.25) | (Close_Loc >= 0.66)
sell_bar   = (Vol_Ratio >= 1.25) | (Close_Loc <= 0.34)
fresh_bull = PPO_Bars_Since_Pos <= 3
fresh_bear = PPO_Bars_Since_Neg <= 3
buy_trig   = fresh_bull & (PPO_Delta > 0)
sell_trig  = fresh_bear & (PPO_Delta < 0)

buy_base   = (buy  >= buy_thr)  & data_ok & liq_ok & buy_bar
sell_base  = (sell >= sell_thr) & data_ok & liq_ok & sell_bar
buy_a, sell_a = buy_base & buy_trig, sell_base & sell_trig
exit_      = buy_base & sell_base & fresh_bull & fresh_bear & (abs(buy-sell) <= 10)
rec        = select([exit_, buy_a, sell_a], ["EXIT","BUY","SELL"], "HOLD")

# --- gates (history version, v46)
spy_bad    = per-date: (SPY.rec_pre_gate == "SELL") | SPY.Regime.isin(["Weak","Mature/Cooling"])
early      = (Regime == "Dormant→Waking") & (Squeeze_Setup_Flag == 1)
is_buy     = (rec == "BUY") & (Ticker != "SPY")
blocked    = is_buy & ~early & (
               (spy_bad & ~is_fx & ~is_country)
             | ((RS_20d.fillna(0) < 0) & ~is_fx)
             | (Below_EMA200 == 1))
rec[blocked] = "HOLD"; Watch[blocked] = "BUY"

# --- stronger-top filter
weak_sell  = (rec == "SELL") & (Below_EMA50 == 0) & (Momentum_Direction > 0)
rec[weak_sell] = "HOLD"
```
Note `spy_bad` is computed from SPY's recommendation **before** any gate or top-filter changes it.

---

## 10. Backtest protocol (recommendations)

### 10.1 Event definition
- **Match the dashboard:** a *new* BUY is `rec=="BUY"` and previous bar's `rec != "BUY"` (same for SELL). Only bars with `Momentum_Data_OK == 1`.
- Also report a **de-clustered** variant (e.g. ignore repeat events on the same ticker within 10 / 20 bars) and a **cross-sectional-clustered** variant (many names fire the same day in the same market regime).
- Add EXIT as its own event class.

### 10.2 Entry / exit prices (the dashboard scorecard uses the signal bar's own close)
| Variant | Entry | Comment |
|---|---|---|
| A (dashboard parity) | close of signal bar *t* | reproduces `scorecard()`; slightly optimistic |
| **B (primary, realistic)** | **open of bar t+1** | signal is only known after the close |
| C | close of bar t+1 | conservative |

Horizons: 5 / 10 / 20 bars (as coded) plus 40 / 60 for trend-follow. Include 5–10 bps per side (more for names near the $5M liquidity floor) and report gross vs net.

### 10.3 Metrics
Per event class and per slice: **n, hit rate, mean, median, excess vs SPY, hit-vs-SPY, MAE/MFE, return/risk (mean ÷ std), worst decile**. Use date-clustered standard errors / bootstrap by date. Compare against the unconditional base rate (all `Momentum_Data_OK` bars) *and* a matched-random baseline (same ticker, same regime, random dates).

### 10.4 Slices that map to the logic above
1. **By regime** (5 regimes) — the whole threshold table in §3.3 stands or falls on this.
2. **By score bucket**: Buy 35–44 / 45–54 / 55–64 / 65+ — is confidence monotonic?
3. **By component** : events with vs without squeeze add-on; by `RSI_Zone`; by `Bars_Since_Pos` 0/1/2/3; by `PPO_Above_Zero`; ATR bump on/off; `Vol_Ratio` ≥ 1.25 vs close-location-only confirmation.
4. **Gate ablation** (run each separately): no gates; SPY gate only; RS gate only; EMA200 gate only; all gates; all gates + exemption. Report the **blocked BUYs'** forward returns vs the passed BUYs'.
5. **Top-filter ablation** for SELL (on/off), and SELL measured as "avoided decline" (negative forward return = success, as the dashboard colours it).
6. **Group split:** core stocks vs ETFs vs Country vs Currency (different anchors/gates/liquidity).
7. **Market state:** SPY regime on the signal date (this is also what the SPY gate keys on).
8. **Baselines for value-added:** raw PPO-cross-only entry (no score), PPO-cross + EMA200 filter, SPY>EMA200 buy-and-hold, equal-weight universe.

### 10.5 Portfolio simulation (optional second stage)
Use the state machine in §8.3, equal-weight (or ATR-risk-parity), max N positions, ranking by Buy score when more BUYs than slots. Report CAGR, Sharpe/Sortino, max drawdown, exposure, turnover, average hold. Test exit variants: SELL/EXIT only; + 20-bar time stop; + 2×ATR trailing stop; + regime flips to Weak.

### 10.6 Out-of-sample discipline
The model has been revised through 46 versions, presumably while looking at the same history, so **every in-sample number is optimistic**. Freeze v46, then (a) hold out the most recent period(s) never examined, (b) walk-forward any threshold/weight tweak (fit on window A, test on window B), (c) report parameter sensitivity (±5 on thresholds, ±1 on `ppo_fresh_bars`) rather than a single point estimate.

---

## 11. Fidelity caveats and look-ahead traps (as coded)

1. **Data history trimmed.** Indicators are computed on `period="2y"` and only the last `keep_bars = 320` rows are scored. For a real back-test, compute on the full history (≥ 4–5 years) and score from roughly bar 260 onward, with SPY holding ≥ 504 + 50 bars before your first test date.
2. **EMA200 warm-up trap.** `Below_EMA200 = (close < EMA200)` is **False** when EMA200 is NaN (first 199 bars), i.e. the name looks "above EMA200" and passes that gate/earns the EMA points. `CORE_FIELDS` does not include `EMA200`. Require `EMA200.notna()` in the back-test.
3. **`Momentum_Data_OK` is not point-in-time.** It is `N_Bars ≥ 300` using the ticker's **total** bar count (a constant per ticker) and finite core fields. Early historical bars of a long-history ticker are flagged OK. Enforce a per-bar warm-up yourself (≥ 252 bars for `Mom_252d`; 126 for `High_52W_Ratio`).
4. **Same-bar close entry** in the shipped scorecard (see §10.2).
5. **Causal percentiles — good, with one fallback.** Percentiles use only SPY bars strictly before the date. If SPY data is unavailable, the code silently falls back to a **pooled, full-sample** distribution (look-ahead). Confirm the dashboard note reads "causally anchored to SPY".
6. **Survivorship / selection bias.** The universe is today's watchlist plus theme lists plus curated carve-outs (Tear's Short List, Frank's Short List, Goldman's Picks), mostly names that already worked. Expect long-side results to be flattered. Test the non-curated universe separately and consider a point-in-time universe.
7. **Price floor not applied in history.** `min_price = $5` is enforced only in `finalize_panel` (`Is_Tradeable`), not in the history/scorecard. Apply `close ≥ 5` yourself.
8. **Live bar.** The live/forming bar can drive displayed scores, but alerts are pinned to the last confirmed close (`alerts_eod_only`). Use closed bars only in back-tests (`--eod-only`).
9. **Liquidity floor is also computed at each bar** from that bar's rolling median, so it is point-in-time — good.
10. **Rounding/flooring.** Components use `floor` (not round) — reproduce exactly or scores will drift by ±1 and flip marginal alerts.
11. **Stronger-top filter does not clear `Watch`**; a top-filtered SELL shows `Watch=""`.
12. **SELL has no market/RS gate**, BUY has three — the asymmetry is intentional but worth testing (§10.4).
13. **Costs and slippage are not modelled** anywhere in v46.
14. **Percent units.** `RS_20d`, `Ret_20d`, `ATR_Pct`, `Close_vs_EMA*` are in percent, not fractions.

---

## 12. Worked example (Neutral regime, long setup)

Inputs: regime Neutral (`wp 1.0, wr 0.8, we 0.9`; Buy thr 40), `Bars_Since_Pos = 0`, `PPO_Hist_Expand_Up = 1`, `PPO > 0` with no zero-cross in last 3 bars, strength 0.5, `rel = 0.8`, RSI 55 (Healthy), close above all three EMAs, `High_52W_Ratio` 0.95, `Trend_Persistence` 0.2, no squeeze, ATR 3%.

| Step | Calc | Result |
|---|---|---|
| PPO | `(18·1 + 8·1·1 + 4) · 1.25 · 1.0 · 0.8` = 30.0 → floor | **30** |
| RSI | `(1−1)·30 + 2·0.8·0.8` = 1.28 → floor | **1** |
| EMA | `12·3/3·0.9` = 10.8 → floor; no halving; 52-wk ratio < 0.97 | **10** |
| Persistence | `8·0.2·0.8` = 1.28 → floor | **1** |
| Squeeze | none | **0** |
| **Buy** | 30 + 1 + 10 + 1 | **42 ≥ 40 ✔** |

Then: trigger (`Bars_Since_Pos 0 ≤ 3`, `PPO_Delta > 0`) ✔; bar check needs `Vol_Ratio ≥ 1.25` or `Close_Loc ≥ 0.66`; liquidity ≥ $5M ✔; gates: SPY not Weak/Mature, `RS_20d ≥ 0`, above EMA200. If all pass → **BUY**. If SPY were in a `Weak` regime and the name is not exempt → converted to **HOLD** with `Watch="BUY"`, `Gate_Block="SPY regime"`.

---

## 13. Config defaults (v46) used above

| Group | Parameter | Value |
|---|---|---|
| Data | `min_bars` / `min_load_bars` / `keep_bars` | 300 / 60 / 320 |
| | `period` / `bench_period` / `benchmark` | 2y / 5y / SPY |
| Liquidity | `min_dollar_vol` / `min_price` / group floors (FX, country) | $5M / $5.00 / $1M |
| PPO | fast / slow / signal; `ppo_fresh_bars`; persist window | 12 / 26 / 9; 3; 5 |
| EMAs | fast / mid / slow | 20 / 50 / 200 |
| RSI | healthy band | 40–65 (zones 30 / 40 / 65 / 75) |
| Squeeze | BB len/σ; KC len/scalar; mom len; recent window; min compression; DMI floor | 20/2.0; 20/1.5; 12; 10; 15; −5 |
| Regime | `regime_hi` / `regime_lo` / `regime_gap` | 55 / 45 / 10 |
| Percentile | lookback / min obs | 504 / 50 |
| Alerts | global buy/sell threshold (fallback) | 40 / 40 |
| | `watch_margin`; `conflict_margin` | 8; 10 |
| | `require_volume_confirm`; `vol_confirm_mult` | True; 1.25 |
| | `require_ppo_cross_for_alert`; `alerts_eod_only` | True; True |
| ATR | `atr_soft_cap`; `atr_extra_points` | 6.0%; +10 |
| Add-on caps | `sqz_buy_add_cap`; `sqz_sell_add_cap` | 10; 8 |
| Sell | `sell_cross_floor`; `sell_vol_damping_on_cross`; `sell_regime_weight_floor` | 30; 0.35; 0.85 |
| Reliability | `min_reliability` | 0.45 |
| Gates | `spy_regime_gate`; `rs_gate`; `rs_lookback`; `rs_min_excess_for_buy` | True; True; 20; 0.0 |
| | `require_above_ema200_for_buy`; `stronger_top_detection`; `require_negative_rs_for_sell` | True; True; False |
| Scorecard | `fwd_horizons` | 5, 10, 20 |

---

## 14. Columns to log from the history frame for the back-test

`Date, Ticker, Last_Close, Open, High, Low, Volume, Dollar_Volume_20d, Vol_Ratio, Close_Loc, Regime, Momentum_Trailing_Pctile, Momentum_Emerging_Pctile, Momentum_Pctile, Buy_Confidence, Sell_Confidence, Buy_Component_{PPO_Cross,RSI,EMA,Persistence,Squeeze}, Sell_Component_{PPO_Cross,RSI,EMA,Weakness,Squeeze}, Sell_Floor_Applied, Buy_Threshold, Sell_Threshold, Buy_Alert, Sell_Alert, Recommendation, Watch, Gate_Block, RSI_14, RSI_Zone, PPO, PPO_Signal, PPO_Delta, PPO_Bars_Since_Pos, PPO_Bars_Since_Neg, Below_EMA20/50/200, EMA_Bull_Stack, High_52W_Ratio, ATR_Pct, Historical_Vol, Signal_Reliability, RS_20d, Squeeze_Setup_Flag, Squeeze_Setup_Strength, Momentum_Direction, N_Bars, Momentum_Data_OK`.
