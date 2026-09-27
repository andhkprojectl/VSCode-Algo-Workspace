"""
scorer_ranker.py
Enhance an existing 1-min ES day-trading strategy with a Scorer/Ranker.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Protocol


# ------------------------------------------------------------------ #
# 1. Data containers
# ------------------------------------------------------------------ #
class Side(Enum):
    LONG  = 1
    SHORT = -1


@dataclass
class Signal:
    """Raw signal coming out of YOUR existing strategy."""
    timestamp: pd.Timestamp
    side: Side
    price: float
    stop: float
    target: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Candidate:
    signal: Signal
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    candidate: Candidate
    score: float
    feature_scores: Dict[str, float]
    size_multiplier: float


# ------------------------------------------------------------------ #
# 2. Strategy protocol  (implement this for your existing strategy)
# ------------------------------------------------------------------ #
class StrategyLike(Protocol):
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]: ...


# ------------------------------------------------------------------ #
# 3. Feature extractor  – snapshot market context at signal time
# ------------------------------------------------------------------ #
class FeatureExtractor:
    """
    All features are normalised to [0, 1] so the scorer is just a
    weighted sum. Anything you compute here must be causal (only past
    data) – the precompute step uses .shift() / rolling() correctly.
    """

    def __init__(self, df_1min: pd.DataFrame):
        self.df = df_1min.copy()
        self._precompute()

    def _precompute(self):
        df = self.df
        c = df["close"]

        # Trend (EMAs + higher-TF proxy via 5x slower EMA slope)
        df["ema_fast"] = c.ewm(span=9,  adjust=False).mean()
        df["ema_mid"]  = c.ewm(span=21, adjust=False).mean()
        df["ema_slow"] = c.ewm(span=50, adjust=False).mean()
        df["ema_slow_slope"] = df["ema_slow"].diff(5)

        # ATR & vol regime
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - c.shift()).abs(),
            (df["low"]  - c.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"]     = tr.ewm(span=14, adjust=False).mean()
        df["atr_pct"] = df["atr"] / c
        df["atr_pct_q"] = df["atr_pct"].rolling(390, min_periods=30).rank(pct=True)

        # RSI(14)
        delta = c.diff()
        gain  =  delta.clip(lower=0).ewm(alpha=1/14).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

        # Relative volume
        df["vol_ma"] = df["volume"].rolling(20).mean()
        df["rvol"]   = df["volume"] / df["vol_ma"].replace(0, np.nan)

        # Intraday range position
        df["date"]      = df.index.date
        df["day_high"]  = df.groupby("date")["high"].cummax()
        df["day_low"]   = df.groupby("date")["low"].cummin()
        rng = (df["day_high"] - df["day_low"]).replace(0, np.nan)
        df["range_pos"] = (c - df["day_low"]) / rng

        # Minutes since 09:30 ET
        df["minutes_from_open"] = (df.index.hour - 9) * 60 + df.index.minute - 30

        self.df = df

    # ---------------------------------------------------------------- #
    def extract(self, signal: Signal) -> Dict[str, float]:
        df = self.df
        if signal.timestamp in df.index:
            row = df.loc[signal.timestamp]
        else:
            pos = df.index.searchsorted(signal.timestamp)
            pos = min(pos, len(df) - 1)
            row = df.iloc[pos]

        side = signal.side
        feat: Dict[str, float] = {}

        # 1. Higher-TF trend alignment  (1 = with trend, 0.5 = flat, 0 = counter)
        align = np.sign(row["ema_slow_slope"]) * side.value
        feat["trend_align"] = float((align + 1) / 2)

        # 2. EMA stack alignment
        long_stack  = row["ema_fast"] > row["ema_mid"] > row["ema_slow"]
        short_stack = row["ema_fast"] < row["ema_mid"] < row["ema_slow"]
        feat["ema_stack"] = 1.0 if (long_stack if side == Side.LONG else short_stack) else 0.0

        # 3. Momentum via RSI, mapped to [0,1] in direction of trade
        rsi = row["rsi"]
        if side == Side.LONG:
            feat["momentum"] = float(np.clip((rsi - 40) / 40, 0, 1))
        else:
            feat["momentum"] = float(np.clip((60 - rsi) / 40, 0, 1))

        # 4. Volatility regime – prefer moderate-to-high ATR percentile
        q = row["atr_pct_q"]
        # Bell-ish: peak around 0.7, low at extremes
        feat["vol_regime"] = float(np.clip(1 - abs(q - 0.7) * 1.8, 0, 1))

        # 5. Volume confirmation
        feat["rvol"] = float(np.clip(row["rvol"] / 2.0, 0, 1))

        # 6. Range position – breakouts want to be near recent extreme
        rp = row["range_pos"]
        if pd.isna(rp): rp = 0.5
        feat["range_pos"] = float(rp if side == Side.LONG else 1 - rp)

        # 7. Time-of-day quality (open & close > midday > lunch lull)
        mfo = row["minutes_from_open"]
        if   mfo < 0   or mfo > 390: ts = 0.0
        elif mfo <= 60:              ts = 1.0      # 09:30-10:30
        elif mfo <= 120:             ts = 0.8
        elif mfo <= 180:             ts = 0.6
        elif mfo <= 300:             ts = 0.25     # 12:30-14:30 lunch
        else:                        ts = 0.85     # close
        feat["time_score"] = ts

        # 8. Reward/Risk quality of the original signal
        rr = abs(signal.target - signal.price) / max(abs(signal.price - signal.stop), 1e-9)
        feat["rr_quality"] = float(np.clip(rr / 3.0, 0, 1))  # 3R -> 1.0

        return feat


# ------------------------------------------------------------------ #
# 4. Scorer
# ------------------------------------------------------------------ #
class Scorer:
    """Weighted linear combination of features -> score in [0,1]."""

    def __init__(self, weights: Dict[str, float], min_score: float = 0.5):
        total = sum(weights.values()) or 1.0
        self.weights   = {k: v / total for k, v in weights.items()}
        self.min_score = min_score

    def score(self, candidate: Candidate) -> ScoredCandidate:
        feat_scores: Dict[str, float] = {}
        weighted = 0.0
        for name, w in self.weights.items():
            s = float(np.clip(candidate.features.get(name, 0.0), 0, 1))
            feat_scores[name] = s
            weighted += s * w
        # Size scales gently with conviction (0.25x .. 1.5x of base)
        size_mult = float(np.clip(0.25 + 1.25 * (weighted - 0.5) / 0.5, 0.25, 1.5)) \
                    if weighted >= 0.5 else 0.0
        return ScoredCandidate(
            candidate=candidate,
            score=weighted,
            feature_scores=feat_scores,
            size_multiplier=size_mult,
        )


# ------------------------------------------------------------------ #
# 5. Ranker  – applies portfolio-level constraints
# ------------------------------------------------------------------ #
class Ranker:
    def __init__(
        self,
        max_trades_per_day: int = 3,
        min_score: float = 0.55,
        cooldown_minutes: int = 10,
        max_same_direction: int = 2,
        allow_flip_same_bar: bool = False,
    ):
        self.max_trades_per_day   = max_trades_per_day
        self.min_score            = min_score
        self.cooldown             = pd.Timedelta(minutes=cooldown_minutes)
        self.max_same_direction   = max_same_direction
        self.allow_flip_same_bar  = allow_flip_same_bar

    def rank(
        self,
        scored: List[ScoredCandidate],
        last_trade_time: Optional[pd.Timestamp] = None,
        trades_today: int = 0,
        same_dir_today: Optional[Dict[Side, int]] = None,
    ) -> List[ScoredCandidate]:
        same_dir_today = same_dir_today or {Side.LONG: 0, Side.SHORT: 0}
        ranked = sorted(scored, key=lambda x: x.score, reverse=True)

        approved: List[ScoredCandidate] = []
        for sc in ranked:
            if trades_today >= self.max_trades_per_day:    break
            if sc.score < self.min_score:                  continue
            sig_ts = sc.candidate.signal.timestamp
            if last_trade_time is not None and (sig_ts - last_trade_time) < self.cooldown:
                continue
            if same_dir_today.get(sc.candidate.signal.side, 0) >= self.max_same_direction:
                continue
            approved.append(sc)
            trades_today += 1
            same_dir_today[sc.candidate.signal.side] += 1
            last_trade_time = sig_ts
        return approved


# ------------------------------------------------------------------ #
# 6. Enhancer  – wraps your existing strategy
# ------------------------------------------------------------------ #
class StrategyEnhancer:
    def __init__(
        self,
        strategy: StrategyLike,
        feature_extractor: FeatureExtractor,
        scorer: Scorer,
        ranker: Ranker,
    ):
        self.strategy = strategy
        self.fe       = feature_extractor
        self.scorer   = scorer
        self.ranker   = ranker

    def process_bar(self, df_so_far: pd.DataFrame) -> List[ScoredCandidate]:
        raw = self.strategy.generate_signals(df_so_far)
        if not raw: return []
        candidates = [Candidate(s, self.fe.extract(s)) for s in raw]
        scored     = [self.scorer.score(c) for c in candidates]
        return scored  # caller (or rank()) decides approval

    def backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Walk-forward, strictly causal."""
        rows = []
        last_trade_time = None
        trades_today = 0
        cur_day = None
        same_dir = {Side.LONG: 0, Side.SHORT: 0}

        # use the feature-enriched df
        fdf = self.fe.df

        for i in range(60, len(fdf)):
            ts  = fdf.index[i]
            day = ts.date()
            if day != cur_day:
                cur_day, trades_today = day, 0
                same_dir = {Side.LONG: 0, Side.SHORT: 0}
                last_trade_time = None

            scored = self.process_bar(fdf.iloc[: i + 1])
            if not scored: continue
            approved = self.ranker.rank(
                scored, last_trade_time, trades_today, same_dir
            )
            for sc in approved:
                sig = sc.candidate.signal
                rows.append({
                    "timestamp": sig.timestamp,
                    "side":      sig.side.name,
                    "entry":     sig.price,
                    "stop":      sig.stop,
                    "target":    sig.target,
                    "score":     round(sc.score, 4),
                    "size_mult": round(sc.size_multiplier, 3),
                    **{f"f_{k}": round(v, 3) for k, v in sc.feature_scores.items()},
                })
                last_trade_time = sig.timestamp
                trades_today += 1
                same_dir[sig.side] += 1

        return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# 7. Example: plugging in YOUR existing strategy
# ------------------------------------------------------------------ #
class MyExistingESStrategy:
    """
    Replace the body of generate_signals() with the logic from your
    current buy/sell and short/cover strategy.  It must return a list
    of Signal objects.  Here we use an EMA-cross + ATR-stop example.
    """
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        if len(df) < 60: return []
        last, prev = df.iloc[-1], df.iloc[-2]
        signals: List[Signal] = []

        long_cross  = prev["ema_fast"] <= prev["ema_mid"] and last["ema_fast"] >  last["ema_mid"]
        short_cross = prev["ema_fast"] >= prev["ema_mid"] and last["ema_fast"] <  last["ema_mid"]

        atr = last["atr"]
        if long_cross:
            signals.append(Signal(
                timestamp=df.index[-1], side=Side.LONG,
                price=last["close"],
                stop=last["close"]  - 1.5 * atr,
                target=last["close"] + 3.0 * atr,
            ))
        elif short_cross:
            signals.append(Signal(
                timestamp=df.index[-1], side=Side.SHORT,
                price=last["close"],
                stop=last["close"]  + 1.5 * atr,
                target=last["close"] - 3.0 * atr,
            ))
        return signals


# ------------------------------------------------------------------ #
# 8. Wire-up & run
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # df = pd.read_parquet("es_1min.parquet")  # columns: open,high,low,close,volume
    # ---- synthetic demo data so the script runs standalone ----
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-02 09:30", periods=390*5, freq="1min", tz="US/Eastern")
    px  = 5000 + np.cumsum(rng.normal(0, 1.5, len(idx)))
    df  = pd.DataFrame({
        "open": px, "high": px + rng.uniform(0.1, 2, len(idx)),
        "low":  px - rng.uniform(0.1, 2, len(idx)),
        "close": px + rng.normal(0, 0.5, len(idx)),
        "volume": rng.integers(500, 5000, len(idx)),
    }, index=idx)

    fe = FeatureExtractor(df)
    scorer = Scorer(weights={
        "trend_align":  0.18,
        "ema_stack":    0.12,
        "momentum":     0.13,
        "vol_regime":   0.10,
        "rvol":         0.12,
        "range_pos":    0.08,
        "time_score":   0.12,
        "rr_quality":   0.15,
    }, min_score=0.55)

    ranker = Ranker(
        max_trades_per_day=3,
        min_score=0.55,
        cooldown_minutes=10,
        max_same_direction=2,
    )

    enhancer = StrategyEnhancer(
        strategy=MyExistingESStrategy(),
        feature_extractor=fe,
        scorer=scorer,
        ranker=ranker,
    )

    trades = enhancer.backtest(fe.df)
    print(trades.head(10))
    print(f"\nTotal trades: {len(trades)} | Long: {(trades.side=='LONG').sum()} "
          f"| Short: {(trades.side=='SHORT').sum()}")
    print("Avg score:", trades["score"].mean() if len(trades) else 0)