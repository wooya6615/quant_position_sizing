"""
세 항을 곱셈적으로 결합해서 최종 포지션 사이즈 산출

    final_size = base_unit_size * m_confidence * vol_term * regime_penalty
    final_size = clip(final_size, 0, MAX_POSITION)

base_unit_size = 0.5 * kelly_fraction_estimate (half-Kelly)
kelly_fraction_estimate: estimate_fixed_params.py에서 딱 한 번 계산해서
                          아래 BASE_UNIT_SIZE 상수에 고정 (fold마다 재추정 금지)

사용법:
    (독립 실행 안 함 -- sizing 패키지 내부 모듈, relative import 사용)
    다른 모듈에서 from src.sizing.combine_sizing import combine 형태로 불러올 것
"""

from pathlib import Path

import pandas as pd

from .confidence_sizing import compute_confidence_term
from .volatility_sizing import compute_vol_term
from .regime_penalty import compute_regime_penalty

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# estimate_fixed_params.py 실행 결과로 채움 (2026-09, half-Kelly)
# 승률 p=0.597, 평균이익=0.1792, 평균손실=0.1046, odds b=1.713 -> Kelly f*=0.3624
BASE_UNIT_SIZE = 0.1812
MAX_POSITION = 1.0  # 자본 대비 최대 비중, 사전 고정


def combine(
    m_confidence: pd.Series,
    vol_term: pd.Series,
    regime_penalty: pd.Series,
    base_unit_size: float = BASE_UNIT_SIZE,
) -> pd.Series:
    assert base_unit_size > 0, (
        "BASE_UNIT_SIZE가 아직 0.0 -- estimate_fixed_params.py를 먼저 실행해서 "
        "half-Kelly 값을 채워야 함"
    )
    raw = base_unit_size * m_confidence * vol_term * regime_penalty
    return raw.clip(lower=0, upper=MAX_POSITION).rename("final_size")


if __name__ == "__main__":
    print("TODO: 세 모듈의 출력을 날짜 기준으로 join한 뒤 combine() 호출")