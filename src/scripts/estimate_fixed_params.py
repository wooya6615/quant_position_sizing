"""
사전 고정 파라미터(SIGMA_TARGET, BASE_UNIT_SIZE=half-Kelly) 1회성 계산 스크립트.

⚠️ 이 스크립트는 "딱 한 번" 실행해서 나온 값을 volatility_sizing.py의
SIGMA_TARGET, combine_sizing.py의 BASE_UNIT_SIZE 상수에 손으로 채워넣는 용도다.
fold마다, 또는 실험 반복마다 재실행해서 값을 갱신하면 그 자체가 in-sample
overfitting이므로 절대 하지 않는다 (prereg 문서 2-2/2-4절).

전제:
    quant_xgboost의 feature_engineering_triple_barrier.py를 이 레포의
    src/ 밑에 복사해 와야 함 (기존 프로젝트의 "자기완결적 레포" 컨벤션 --
    quant_ranking_kr이 base_features.py를 복사해 온 것과 동일한 방식).
    데이터/모델 설정은 quant_xgboost의 production 모델
    (build_production_model_064350_base.py)과 완전히 동일해야 함:
        PT_SL=(2,1), NUM_DAYS=30, THRESHOLD=0.60, BASE 13개 피처

산출:
    1. data/064350_walkforward_trades_with_proba.csv
       -- walk-forward로 재현한 거래 로그 (진입일, 예측확률, 순수익률, 보유기간)
          confidence_sizing.py가 이 파일을 입력으로 쓴다.
    2. 콘솔에 SIGMA_TARGET, BASE_UNIT_SIZE(half-Kelly) 최종값 출력
       -- 이 값을 volatility_sizing.py / combine_sizing.py 상수에 직접 채워넣을 것.

사용법 (레포 루트에서):
    python -m src.scripts.estimate_fixed_params   (레포 루트에서 실행)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.feature_engineering_triple_barrier import build_triple_barrier_dataset, FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKER_KRX = "064350"
TICKER = "064350.KS"

# production 모델(build_production_model_064350_base.py)과 완전히 동일 -- 재튜닝 금지
PT_SL = (2, 1)
NUM_DAYS = 30
THRESHOLD = 0.60
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS
ROUND_TRIP_COST = 0.002
SEED_FOR_TRADE_LOG = 42  # 확률/거래 로그 생성은 대표 seed 1개로 -- 5-seed 비교는 ablation 단계에서

MODEL_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=SEED_FOR_TRADE_LOG,
)


def config_label() -> str:
    return f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"


def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{config_label()}_base.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    else:
        print(f"{path.name} 없음 -- 새로 생성 중...")
        df = build_triple_barrier_dataset(ticker=TICKER, pt_sl=PT_SL, num_days=NUM_DAYS)
        df.to_csv(path)
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def walk_forward_splits(n_rows, train_size, test_size, step, embargo):
    """quant_xgboost의 backtest_num_days30_064350.py와 동일한 분할 로직."""
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = range(start, start + train_size)
        test_start = start + train_size + embargo
        test_idx = range(test_start, test_start + test_size)
        splits.append((train_idx, test_idx))
        start += step
    return splits


def generate_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """threshold=0.60 이상인 진입만 남긴 거래 로그. pred_proba 컬럼 포함."""
    X = df[FEATURE_COLS_BASE]
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        if y_train.nunique() < 2:
            continue

        model = xgb.XGBClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        test_positions = list(test_idx)
        proba = model.predict_proba(X.iloc[test_positions])[:, 1]

        i = 0
        while i < len(test_positions):
            if proba[i] >= THRESHOLD:
                row_idx = test_positions[i]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = int(df["holding_rows_tb"].iloc[row_idx])
                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "entry_date": df.index[row_idx],
                        "exit_date": df.index[exit_row],
                        "pred_proba": proba[i],
                        "hist_vol_20d": df["hist_vol_20d"].iloc[row_idx],
                        "net_return": net_return,
                        "holding_days": holding,
                    })
                i += max(holding, 1)
            else:
                i += 1

    return pd.DataFrame(trades)


def compute_sigma_target(df: pd.DataFrame) -> float:
    """전체 표본기간 hist_vol_20d의 중앙값 -- 진입 시점이 아니라 전체 시계열 기준
    (진입 편향 없는 종목 고유의 '평상시 변동성 수준'을 잡기 위함)."""
    return float(df["hist_vol_20d"].dropna().median())


def compute_half_kelly(trades: pd.DataFrame) -> float:
    """
    f* = (bp - q) / b   (b=평균이익/평균손실 odds, p=승률, q=1-p)
    half-Kelly = 0.5 * f*
    """
    wins = trades.loc[trades["net_return"] > 0, "net_return"]
    losses = trades.loc[trades["net_return"] <= 0, "net_return"]

    p = len(wins) / len(trades)
    q = 1 - p
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    b = avg_win / avg_loss

    kelly_fraction = (b * p - q) / b
    half_kelly = 0.5 * kelly_fraction

    print(f"승률 p={p:.3f}, 평균이익={avg_win:.4f}, 평균손실={avg_loss:.4f}, odds b={b:.3f}")
    print(f"Kelly f*={kelly_fraction:.4f}, half-Kelly={half_kelly:.4f}")

    if half_kelly <= 0:
        print("⚠️ half-Kelly가 0 이하 -- 이 신호의 승률/손익비로는 Kelly 배팅 자체가 "
              "권장되지 않는다는 뜻. BASE_UNIT_SIZE를 0보다 큰 임의값으로 강제하지 말고, "
              "이 결과 자체를 사이징 실험의 전제조건 위반으로 보고할 것.")

    return half_kelly


def main():
    df = load_dataset()
    trades = generate_trade_log(df)

    out_path = DATA_DIR / f"{TICKER_KRX}_walkforward_trades_with_proba.csv"
    trades.to_csv(out_path, index=False)
    print(f"거래 로그 저장: {out_path.name} ({len(trades)}건)\n")

    sigma_target = compute_sigma_target(df)
    print(f"\nSIGMA_TARGET (hist_vol_20d 중앙값) = {sigma_target:.4f}")
    print("-> volatility_sizing.py의 SIGMA_TARGET 상수에 이 값을 채워넣을 것\n")

    half_kelly = compute_half_kelly(trades)
    print(f"\nBASE_UNIT_SIZE (half-Kelly) = {half_kelly:.4f}")
    print("-> combine_sizing.py의 BASE_UNIT_SIZE 상수에 이 값을 채워넣을 것")


if __name__ == "__main__":
    main()