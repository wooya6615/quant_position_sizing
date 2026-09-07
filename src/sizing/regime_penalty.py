"""
레짐 penalty 항 (regime_penalty) 계산 모듈
quant_unsupervised의 K-means 국면 클러스터링 인프라 재사용 (재학습 없음)

    regime_penalty = clip(1 - anomaly_score_normalized, 0.3, 1.0)

quant_unsupervised에서 이미 학습된 K-means 클러스터를 재사용한다 -- 단, quant_unsupervised
자체는 모델을 저장하지 않고 118990/052690에만 적용해봤을 뿐 064350에는 적용된 적이
없었으므로, quant_position_sizing/src/scripts/fit_regime_model_064350.py로 064350에
대해 딱 한 번 fit해서 저장한 산출물(models/064350_regime_model.joblib)을 재사용한다.
이례적 국면(anomalous_cluster) 판정 규칙과 그 산출 과정은 그 스크립트의 docstring 참고.

feature 4개 (quant_unsupervised와 동일, 새로 만들지 않음):
    hist_vol_20d, return_20d, bb_width, macd_hist

anomaly_score_normalized 계산:
    1. quant_unsupervised에서 저장된 K-means 모델(centroid) 로드
    2. 현재 시점 feature 벡터와 "이례적 국면" 클러스터 중심 간 유클리드 거리 계산
    3. 거리를 [0,1]로 min-max 정규화 (거리가 가까울수록 anomaly_score 높음 ->
       1 - normalized_distance 로 변환해서 "이례적일수록 1에 가깝게")

주의:
    정규화 min/max는 quant_unsupervised 학습 당시 기록된 값을 그대로 재사용할 것
    (이 실험에서 새로 min/max를 계산하면 그 자체가 새로운 파라미터 추정 =
    overfitting 위험 재도입).

사용법:
    다른 모듈에서 from src.sizing.regime_penalty import compute_regime_penalty 형태로 불러올 것
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

REGIME_PENALTY_CLIP_MIN = 0.3
REGIME_PENALTY_CLIP_MAX = 1.0

REGIME_FEATURES = ["hist_vol_20d", "return_20d", "bb_width", "macd_hist"]


def compute_regime_penalty(anomaly_score_normalized: pd.Series) -> pd.Series:
    """
    anomaly_score_normalized: [0,1], 1에 가까울수록 이례적 국면
    반환: regime_penalty, [0.3, 1.0] 범위 (이례적일수록 낮은 값 -> 사이즈 축소)
    """
    raw = 1 - anomaly_score_normalized
    return raw.clip(lower=REGIME_PENALTY_CLIP_MIN, upper=REGIME_PENALTY_CLIP_MAX).rename(
        "regime_penalty"
    )


if __name__ == "__main__":
    # TODO:
    # 1. quant_unsupervised에서 저장된 K-means centroid / 정규화 min-max 로드
    # 2. 064350 feature 4개(REGIME_FEATURES) 시계열 로드
    # 3. 클러스터 중심까지 거리 -> anomaly_score_normalized 계산
    # 4. compute_regime_penalty() 호출
    print("TODO: quant_unsupervised 저장 모델 경로 연결 필요 (재학습 금지)")