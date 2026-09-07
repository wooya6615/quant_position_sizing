"""
Triple-barrier 라벨 버전 BASE feature 데이터셋 생성 (BASE 전용 -- COMBINED/밸류에이션 없음).

[quant_position_sizing용 사본에서 정리한 부분]
- quant_xgboost 원본은 BASE + COMBINED(밸류에이션) 둘 다 만들었지만, 이 프로젝트는
  밸류에이션 라인이 이미 폐기 확정(PER<=0 버그로 flagship 결과 자체가 무효화됨)이라
  build_triple_barrier_dataset_combined()와 feature_engineering_valuation.py 의존성을
  뺐음. 이 사본은 build_triple_barrier_dataset() (BASE 전용) 하나만 남김.
- NUM_DAYS 기본값/실행 예시를 20 -> 30으로 교체함. num_days 축 PBO 검증
  (quant_xgboost/compute_pbo_num_days_064350.py, PBO 17.5%/낮음, logit +2.920)에서
  30이 최종 채택값으로 확정됐고, quant_position_sizing이 재사용하는 신호도 이
  설정(PT_SL=(2,1), NUM_DAYS=30, THRESHOLD=0.60) 기준이라 여기서도 20으로 두면
  다른 스크립트들이 찾는 데이터 파일과 어긋남.
- 산출 파일명에 `_base` 접미사를 붙임 -- quant_xgboost의 backtest_num_days30_064350.py
  / build_production_model_064350_base.py 등이 전부 `..._nd30_hl_base.csv` 형태를
  쓰는데 원본 __main__은 접미사가 없어서 그 스크립트들이 기대하는 파일명과 안 맞았음.

[체결 방식 최종 확정] "D일 종가까지 정보로 신호 -> D+1일 종가 체결"로 확정.
시가체결/하이브리드(종가라벨+시가체결) 등 여러 대안을 비교해본 결과, 시가체결은
배리어 라벨 자체가 더 시끄러워져서(진입일 하루치 장중 노이즈가 라벨에 섞임)
모델 신호 선별력이 떨어짐이 확인됨. 반면 종가체결은 그 노이즈가 원천 배제돼 더
깨끗한 학습 목표가 되고, 종가 동시호가(MOC) 주문으로 실제 체결도 가능한 정당한
가정이라 최종 채택.

[High/Low 반영] 청산(배리어 터치) 판정에 High/Low 반영. 기존엔 종가만으로 판정해서
장중에 배리어를 건드렸다가 종가에 회복된 경우를 놓쳤음 -- 실제 손절/익절 주문은
가격이 그 선에 닿는 즉시 체결되므로, High/Low로 장중 터치까지 감지하고 청산가도
그날 종가가 아니라 실제 배리어 트리거 가격을 쓰도록 labeling_triple_barrier.py의
apply_triple_barrier()/get_bins()를 확장함.

기존 feature_engineering.py의 add_label()(고정 horizon 이진분류)은 비교 기준선으로
그대로 남겨두고, labeling_triple_barrier.py의 triple-barrier 방식으로 새 라벨을
추가로 만듦. feature 계산 로직(모멘텀/변동성/거래량/상대강도)은 feature_engineering.py를
그대로 재사용 -- feature/모델은 최대한 기존과 동일하게 유지해서 "라벨 정의만 바꿨을 때
성능이 달라지는가"를 순수하게 비교하기 위함.

산출 컬럼:
    label_fixed      기존 고정 horizon=20 이진 라벨 (비교 기준선, triple-barrier의
                      num_days=30과는 별개 축 -- 서로 안 맞춰도 됨)
    label_tb         triple-barrier 라벨 (-1/0/1)
    label_tb_binary  label_tb > 0 -> 1, 아니면 0 (기존 이진분류 파이프라인과 바로 비교 가능,
                      호출부에서 df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)로 생성)
    ret_tb           triple-barrier 실현수익률 (가변 보유기간 -- future_return과 다름)

사용법 (레포 루트에서):
    python -m src.data.feature_engineering_triple_barrier

전제:
    feature_engineering.py, labeling_triple_barrier.py를 quant_xgboost에서 이 파일과
    같은 폴더(src/data/)로 복사해올 것.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    load_data, add_momentum_features, add_volatility_features,
    add_volume_features, add_relative_strength_features, add_label,
)
from labeling_triple_barrier import build_shifted_barrier_labels

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]


def build_triple_barrier_dataset(
    ticker: str = "064350.KS",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon_fixed: int = 20,        # label_fixed(비교 기준선)용 -- num_days와 별개 축, 안 맞춰도 됨
    cost_threshold: float = 0.012,  # 기존 h=20 실험 임계값과 동일 (DEFAULT_HORIZON_COST_MAP[20])
    pt_sl: tuple = (2, 1),          # [변경] production 채택값으로 기본값 교체 (기존 (1,1) 초기 실험값)
    vol_span: int = 20,
    num_days: int = 30,             # [변경] 20 -> 30, num_days 축 PBO 검증으로 최종 채택된 값
) -> pd.DataFrame:
    df, bench = load_data(ticker, benchmark, start, end)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)

    # 기존 고정 horizon 라벨 (비교 기준선)
    df = add_label(df, horizon=horizon_fixed, cost_threshold=cost_threshold)
    df = df.rename(columns={"label": "label_fixed"})

    # triple-barrier 라벨 (체결 지연 1일 + D+1 종가체결로 확정 + High/Low로 장중 터치 반영)
    close = df["Close"]
    has_high = "High" in df.columns
    has_low = "Low" in df.columns
    high = df["High"] if has_high else None
    low = df["Low"] if has_low else None
    if not (has_high and has_low):
        missing = [name for name, present in [("High", has_high), ("Low", has_low)] if not present]
        print(f"경고: 데이터에 {', '.join(missing)} 컬럼이 없어서 그만큼 장중 터치 판정 "
              f"정확도가 떨어짐 (없는 쪽은 종가로 대체됨 -- labeling_triple_barrier.py의 "
              f"처리 방식 확인할 것)")
    bins = build_shifted_barrier_labels(
        close, vol_span=vol_span, num_days=num_days, pt_sl=pt_sl, high=high, low=low,
    )

    df["label_tb"] = bins["label"]
    df["ret_tb"] = bins["ret"]
    # 백테스트에서 "이 거래를 며칠 보유했는지"로 다음 진입 시점을 정하는 데 씀
    # (fixed horizon과 달리 배리어 도달 시점이 거래마다 다름 -- 가변 보유기간)
    touch_pos = close.index.get_indexer(bins["touch_time"])
    entry_pos = close.index.get_indexer(bins.index)
    df["holding_rows_tb"] = pd.Series(touch_pos - entry_pos, index=bins.index)

    feature_cols = [
        "Close", "Volume",
        *FEATURE_COLS_BASE,
        "future_return", "label_fixed",
        "label_tb", "ret_tb", "holding_rows_tb",
    ]

    result = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=FEATURE_COLS_BASE + ["label_fixed", "label_tb"])
    return result


if __name__ == "__main__":
    TICKER_KRX = "064350"
    TICKER = "064350.KS"
    PT_SL = (2, 1)
    NUM_DAYS = 30  # [변경] 20 -> 30, production 채택값과 일치시킴
    # [변경] 다른 quant_xgboost 스크립트들(backtest_num_days30_064350.py 등)이 전부
    # `_base` 접미사가 붙은 파일명을 찾으므로 여기서도 맞춤 -- 접미사 없으면
    # estimate_fixed_params.py 등이 파일을 못 찾고 새로 생성해버림 (중복/불일치 위험)
    CONFIG_LABEL = f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"

    print(f"=== BASE ({TICKER_KRX}, {CONFIG_LABEL}) ===")
    dataset = build_triple_barrier_dataset(ticker=TICKER, pt_sl=PT_SL, num_days=NUM_DAYS)
    print(f"생성된 데이터셋 shape: {dataset.shape}")

    tb_dist = dataset["label_tb"].value_counts(normalize=True)
    print(f"triple-barrier 라벨(label_tb) 분포:\n{tb_dist}")
    if tb_dist.get(0.0, 0) < 0.02:
        print("경고: label_tb=0 비율이 여전히 2% 미만 -- 배리어가 너무 타이트할 수 있음.")

    out_path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}_base.csv"
    dataset.to_csv(out_path)
    print(f"저장 완료: {out_path}")