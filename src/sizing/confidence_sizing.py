"""
신뢰도 항 (m_confidence) 계산 모듈
López de Prado (2018) Ch.10 확률->베팅크기 공식 구현

    z = (p - 1/K) / sqrt(p*(1-p))       # K=2 (이진분류)
    m_confidence = 2*Phi(z) - 1          # Phi = 표준정규 CDF

입력: 기존 quant_xgboost triple-barrier 모델이 이미 출력하는 예측확률 p
      (재학습 없음 -- 저장된 예측값을 그대로 읽어서 사용)

주의:
    p < threshold(0.60)인 행은 애초에 진입하지 않으므로 이 함수가 호출되는
    시점에는 p가 항상 [0.60, 1.0] 구간이어야 함. 그 밖의 값이 들어오면
    업스트림(신호 필터링)에 문제가 있다는 뜻이므로 assert로 조기 발견.

사용법:
    다른 모듈에서 from src.sizing.confidence_sizing import compute_confidence_term 형태로 불러올 것
    (레포 루트에서 python -m src.scripts.estimate_fixed_params 실행 시 자동으로 로드됨)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_THRESHOLD = 0.60  # quant_xgboost 기존 신호와 동일 -- 재튜닝 금지


def compute_confidence_term(p: pd.Series, k: int = 2) -> pd.Series:
    """
    p: 예측확률 Series (이미 threshold를 통과한 행만 들어와야 함)
    반환: m_confidence Series, [2*Phi((0.60-0.5)/sqrt(0.6*0.4))-1, 1.0] 범위
    """
    assert (p >= ENTRY_THRESHOLD - 1e-9).all(), (
        f"p에 threshold({ENTRY_THRESHOLD}) 미만 값이 섞여 있음 -- "
        "업스트림 신호 필터링을 확인할 것"
    )
    z = (p - 1.0 / k) / np.sqrt(p * (1 - p))
    m_confidence = 2 * norm.cdf(z) - 1
    return pd.Series(m_confidence, index=p.index, name="m_confidence")


if __name__ == "__main__":
    # TODO: quant_xgboost 백테스트 로그(예측확률 컬럼 포함)를 로드해서
    #       compute_confidence_term()에 넣고 결과를 data/에 저장
    # path = DATA_DIR / "064350_triple_barrier_predictions.csv"
    # df = pd.read_csv(path, index_col=0, parse_dates=True)
    # df["m_confidence"] = compute_confidence_term(df["pred_proba"])
    # df.to_csv(DATA_DIR / "064350_confidence_term.csv")
    print("TODO: 기존 triple-barrier 예측 로그 경로 연결 필요")