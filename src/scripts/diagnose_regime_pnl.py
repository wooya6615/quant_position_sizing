"""
[진단 전용, 실험 아님] regime_penalty가 깎는 "이례적 국면"(고변동성+급등 모멘텀)이
실제로는 위험한 국면이 아니라 오히려 수익이 좋은 국면이었는지 확인한다.

배경: backtest_sizing_ablation.py 결과에서 MDD는 크게 개선됐지만 Sharpe는 5/5 seed
전부 악화됐음 (PROJECT_SUMMARY.md 참고). 동일 배율이었다면 Sharpe는 불변이어야
하므로, 사이징이 거래를 잘못된 방향으로 차등 가중했다는 뜻 -- 유력한 후보가
regime_penalty. 이 스크립트는 그 가설 하나만 확인한다.

⚠️ 이건 사이징 공식을 바꾸는 게 아니라 "왜 실패했는지" 들여다보는 진단이라 새
pre-registration이나 새 브랜치가 필요 없음. 여기서 나온 결과로 penalty 방향을
뒤집는 v2를 만들기로 하면, 그때는 별도 pre-registration이 필요함 (이 스크립트가
그 결정을 자동으로 내리지 않음).

전제: estimate_fixed_params.py가 이미 생성한 거래 로그와 fit_regime_model_064350.py가
이미 생성한 국면 모델을 그대로 읽기만 함 -- 둘 다 재생성/재학습하지 않음.

사용법 (레포 루트에서):
    python -m src.scripts.diagnose_regime_pnl
"""

from pathlib import Path

import joblib
import pandas as pd

from src.data.feature_engineering_triple_barrier import build_triple_barrier_dataset, FEATURE_COLS_BASE
from src.regime.regime_features import REGIME_FEATURE_COLS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

TRADE_LOG_PATH = DATA_DIR / "064350_walkforward_trades_with_proba.csv"
REGIME_MODEL_PATH = MODELS_DIR / "064350_regime_model.joblib"

TICKER_KRX = "064350"
TICKER = "064350.KS"
PT_SL = (2, 1)
NUM_DAYS = 30


def config_label() -> str:
    return f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"


def load_triple_barrier_df() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{config_label()}_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def main():
    if not TRADE_LOG_PATH.exists():
        raise FileNotFoundError(
            f"{TRADE_LOG_PATH}가 없음 -- 먼저 python -m src.scripts.estimate_fixed_params 실행할 것"
        )
    if not REGIME_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{REGIME_MODEL_PATH}가 없음 -- 먼저 python -m src.scripts.fit_regime_model_064350 실행할 것"
        )

    trades = pd.read_csv(TRADE_LOG_PATH, parse_dates=["entry_date", "exit_date"])
    df = load_triple_barrier_df()
    artifact = joblib.load(REGIME_MODEL_PATH)
    scaler, kmeans, anomalous_cluster = artifact["scaler"], artifact["kmeans"], artifact["anomalous_cluster"]

    # 각 거래의 진입일(entry_date) 시점 국면 피처를 df에서 찾아 클러스터 라벨을 붙임
    regime_features = df.loc[trades["entry_date"], REGIME_FEATURE_COLS].reset_index(drop=True)
    missing = regime_features.isna().any(axis=1)
    if missing.any():
        print(f"경고: {missing.sum()}건은 국면 피처에 NaN이 있어 진단에서 제외함")
        trades = trades.loc[~missing].reset_index(drop=True)
        regime_features = regime_features.loc[~missing].reset_index(drop=True)

    X_scaled = scaler.transform(regime_features)
    trades["regime"] = kmeans.predict(X_scaled)
    trades["is_anomalous"] = trades["regime"] == anomalous_cluster

    print(f"=== {TICKER_KRX} 거래 {len(trades)}건, 진입 시점 국면별 실제 손익 ===\n")

    summary = trades.groupby("is_anomalous")["net_return"].agg(
        n_trades="count", mean_return="mean", win_rate=lambda s: (s > 0).mean(),
    )
    summary.index = summary.index.map({True: f"이례적 국면(cluster={anomalous_cluster})", False: "정상 국면"})
    print(summary.to_string())

    mean_anomalous = trades.loc[trades["is_anomalous"], "net_return"].mean()
    mean_normal = trades.loc[~trades["is_anomalous"], "net_return"].mean()
    diff = mean_anomalous - mean_normal

    print(f"\n이례적 국면 평균수익 - 정상 국면 평균수익 = {diff:+.4f}")
    if diff > 0:
        print(
            "-> 이례적 국면에서 오히려 평균수익이 더 높음. regime_penalty가 이 국면의 "
            "사이즈를 깎은 게 방향이 잘못됐을 가능성이 있음 -- 방향을 뒤집는 v2를 "
            "고려할 수 있으나, 이는 새로운 가설이므로 별도 pre-registration이 필요함."
        )
    else:
        print(
            "-> 이례적 국면 평균수익이 더 낮거나 비슷함. 이 가설(penalty 방향 문제)은 "
            "기각됨 -- Sharpe 악화의 원인이 다른 데 있거나, 이 사이징 라인 자체를 "
            "접고 다음 방향(페어트레이딩 등)으로 넘어가는 게 합리적."
        )


if __name__ == "__main__":
    main()