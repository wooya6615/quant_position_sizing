"""
변동성 항 (vol_term) 계산 모듈 -- Volatility targeting

    vol_term = clip(sigma_target / sigma_realized_20d, 0.3, 1.5)

sigma_target: 064350 전체 기간 realized vol의 "중앙값"으로 딱 한 번 계산해서
              아래 SIGMA_TARGET 상수에 고정값으로 박아넣을 것 (매 fold 재계산 금지
              -- fold마다 다시 계산하면 그 자체가 미래 정보 누수).
              -> estimate_fixed_params.py 실행 후 이 파일의 SIGMA_TARGET 값을 채울 것.

sigma_realized_20d: 매매 시점 기준 과거 20거래일 realized vol (과거 데이터만 사용,
                     당연히 shift 없이 그대로 써도 되는 값 -- 이미 과거 데이터임).

clip 범위 [0.3, 1.5]는 사전 고정 -- 데이터 보고 나서 바꾸지 말 것.

사용법:
    다른 모듈에서 from src.sizing.volatility_sizing import compute_vol_term 형태로 불러올 것
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# TODO: estimate_fixed_params.py 실행 결과로 채우기 (0.0이면 아직 미확정 상태)
SIGMA_TARGET = 0.0

VOL_TERM_CLIP_MIN = 0.3
VOL_TERM_CLIP_MAX = 1.5


def compute_realized_vol_20d(returns: pd.Series) -> pd.Series:
    """일별 수익률 Series -> 20거래일 rolling 연율화 변동성"""
    return returns.rolling(20).std() * np.sqrt(252)


def compute_vol_term(sigma_realized: pd.Series, sigma_target: float = SIGMA_TARGET) -> pd.Series:
    assert sigma_target > 0, (
        "SIGMA_TARGET이 아직 0.0 -- estimate_fixed_params.py를 먼저 실행해서 "
        "값을 채워야 함"
    )
    raw = sigma_target / sigma_realized
    return raw.clip(lower=VOL_TERM_CLIP_MIN, upper=VOL_TERM_CLIP_MAX).rename("vol_term")


if __name__ == "__main__":
    print("TODO: 064350 일별 수익률 로드 -> compute_realized_vol_20d -> compute_vol_term")