"""
K-means 기반 시장 국면(regime) 클러스터링.
K=2~6 중 silhouette score가 가장 높은 K를 자동 선택.

quant_unsupervised/src/clustering/regime_clustering.py를 그대로 이식.
[변경] `from src.features.regime_features import ...` -> relative import로 수정
(패키지 경로가 quant_unsupervised와 다르므로 -- 로직 자체는 변경 없음, 재학습 아님).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .regime_features import REGIME_FEATURE_COLS


@dataclass
class RegimeModel:
    scaler: StandardScaler
    kmeans: KMeans
    n_clusters: int
    silhouette: float


def fit_regime_model(regime_df: pd.DataFrame, k_range=range(2, 7), seed: int = 42) -> RegimeModel:
    """regime_df: build_regime_dataset()이 반환한 데이터셋 (REGIME_FEATURE_COLS만 포함)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(regime_df[REGIME_FEATURE_COLS])

    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if best is None or score > best.silhouette:
            best = RegimeModel(scaler=scaler, kmeans=km, n_clusters=k, silhouette=score)
    return best


def label_regimes(regime_df: pd.DataFrame, model: RegimeModel) -> pd.DataFrame:
    """regime_df에 'regime' 컬럼을 추가해서 반환."""
    result = regime_df.copy()
    X_scaled = model.scaler.transform(result[REGIME_FEATURE_COLS])
    result["regime"] = model.kmeans.predict(X_scaled)
    return result


def describe_regimes(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """각 국면의 평균 변동성/모멘텀 프로필 -- 국면에 이름 붙일 때 참고."""
    return labeled_df.groupby("regime")[REGIME_FEATURE_COLS].mean().round(4)