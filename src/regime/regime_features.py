"""
국면 클러스터링용 피처 계산.
quant_xgboost/feature_engineering.py의 변동성/모멘텀 로직 중
클러스터링에 쓰는 4개만 이식. 라벨(label/future_return)은 계산하지 않음 --
이 레포는 지도학습이 아니므로 애초에 필요 없음.

quant_unsupervised/src/features/regime_features.py를 그대로 이식.

⚠️ 이 4개 컬럼명(REGIME_FEATURE_COLS)은 quant_position_sizing의
src/data/feature_engineering_triple_barrier.py가 만드는 FEATURE_COLS_BASE와
정의가 동일하다 (둘 다 quant_xgboost/feature_engineering.py에서 파생됨).
그래서 backtest_sizing_ablation.py는 이 값을 따로 계산하지 않고 이미 갖고 있는
triple-barrier 데이터프레임에서 이 4개 컬럼만 뽑아 쓴다.
"""

import numpy as np
import pandas as pd

REGIME_FEATURE_COLS = ["hist_vol_20d", "return_20d", "bb_width", "macd_hist"]


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    df["return_20d"] = close.pct_change(20)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - signal_line

    daily_ret = close.pct_change()
    df["hist_vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["bb_width"] = (upper - lower) / ma20

    return df


def build_regime_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """가격 데이터에 국면 피처를 붙이고 NaN을 제거해서 반환."""
    df = add_regime_features(df)
    keep_cols = REGIME_FEATURE_COLS
    return df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()