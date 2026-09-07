"""
064350(현대로템)에 국면 클러스터링을 딱 한 번 적용해서 저장하는 스크립트.

⚠️ 배경: quant_unsupervised는 모델을 저장하지 않고 실행할 때마다 그 자리에서
fit_regime_model()을 새로 호출하며, 지금까지 118990/052690에만 돌려봤을 뿐
064350에는 적용된 적이 없다. 즉 "quant_unsupervised에서 이미 학습된 모델을
재사용한다"는 pre-registration의 전제를 실제로 충족하려면, 그 학습을 담당하는
이 스크립트를 "딱 한 번" 돌려서 나온 산출물을 이후 계속 재사용해야 한다
(estimate_fixed_params.py와 같은 성격 -- 1회성 계산 후 하드코딩/저장, 매 실험마다
재실행 금지).

이례적 국면(anomalous cluster) 판정 규칙 (사전 고정, 결과 보고 나서 바꾸지 않음):
    국면별 평균 프로필에서 hist_vol_20d와 return_20d를 각각 클러스터 간 min-max
    정규화한 뒤 합산해서 가장 높은 클러스터를 "고변동성+급등 모멘텀" 국면으로 지정한다.
    (quant_unsupervised README가 다른 종목들에서 "국면 1(고변동성+급등 모멘텀)"이라고
    부른 것과 같은 직관을 기계적 규칙으로 고정한 것 -- 064350 결과를 보고 나서
    자의적으로 고르지 않기 위함)

산출물: models/064350_regime_model.joblib
    {
        "scaler": StandardScaler (fitted),
        "kmeans": KMeans (fitted),
        "anomalous_cluster": int,
        "dist_min": float,   # 학습 표본 전체에서 이례적 클러스터 중심까지 거리의 최솟값
        "dist_max": float,   # 같은 거리의 최댓값 (정규화용, 이후 재계산 안 함)
        "ticker": "064350.KS",
    }

사용법 (레포 루트에서):
    python -m src.scripts.fit_regime_model_064350
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.regime.price_data import load_price_data
from src.regime.regime_features import build_regime_dataset, REGIME_FEATURE_COLS
from src.regime.regime_clustering import fit_regime_model, label_regimes, describe_regimes

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = MODELS_DIR / "064350_regime_model.joblib"

TICKER = "064350.KS"
START = "2015-01-01"
END = "2026-07-18"  # feature_engineering_triple_barrier.py와 동일 기간


def pick_anomalous_cluster(profile: pd.DataFrame) -> int:
    """hist_vol_20d, return_20d를 각각 클러스터 간 min-max 정규화 후 합산 -> 최댓값 클러스터."""
    vol = profile["hist_vol_20d"]
    mom = profile["return_20d"]
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min()) if vol.max() != vol.min() else vol * 0
    mom_norm = (mom - mom.min()) / (mom.max() - mom.min()) if mom.max() != mom.min() else mom * 0
    combined = vol_norm + mom_norm
    return int(combined.idxmax())


def main():
    print(f"=== {TICKER} 국면 클러스터링 (1회성 -- 이후 재학습 금지) ===")
    df = load_price_data(TICKER, START, END)
    regime_df = build_regime_dataset(df)

    model = fit_regime_model(regime_df)  # seed=42 고정 (quant_unsupervised 기본값)
    print(f"선택된 국면 개수 (K): {model.n_clusters} (silhouette={model.silhouette:.3f})")

    labeled = label_regimes(regime_df, model)
    profile = describe_regimes(labeled)
    print("\n국면별 평균 프로필 (변동성/모멘텀):")
    print(profile)

    anomalous_cluster = pick_anomalous_cluster(profile)
    print(f"\n이례적 국면(anomalous_cluster) = {anomalous_cluster}")
    print(f"해당 국면 프로필:\n{profile.loc[anomalous_cluster]}")

    n_days_in_cluster = (labeled["regime"] == anomalous_cluster).sum()
    print(f"\n전체 {len(labeled)}일 중 이 국면에 속한 날: {n_days_in_cluster}일 "
          f"({n_days_in_cluster / len(labeled):.1%})")

    # 이례적 클러스터 중심까지의 거리를 학습 표본 전체에 대해 계산 -> 정규화용 min/max 고정
    X_scaled = model.scaler.transform(regime_df[REGIME_FEATURE_COLS])
    centroid = model.kmeans.cluster_centers_[anomalous_cluster]
    distances = np.linalg.norm(X_scaled - centroid, axis=1)
    dist_min, dist_max = float(distances.min()), float(distances.max())
    print(f"\n이례적 클러스터 중심까지 거리 범위: [{dist_min:.4f}, {dist_max:.4f}] "
          f"(regime_penalty.py 정규화에 이 값을 고정 사용)")

    artifact = {
        "scaler": model.scaler,
        "kmeans": model.kmeans,
        "anomalous_cluster": anomalous_cluster,
        "dist_min": dist_min,
        "dist_max": dist_max,
        "ticker": TICKER,
    }
    joblib.dump(artifact, OUT_PATH)
    print(f"\n저장 완료: {OUT_PATH}")
    print("-> backtest_sizing_ablation.py의 compute_anomaly_score_normalized()가 이 파일을 로드해서 씀.")
    print("   이 스크립트를 재실행하면 K-means가 다시 학습되므로, 결과를 보고 나서 재실행하지 말 것")
    print("   (재실행이 필요하면 pre-registration 문서에 addendum을 남길 것).")


if __name__ == "__main__":
    main()