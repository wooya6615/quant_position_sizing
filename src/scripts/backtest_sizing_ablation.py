"""
064350 triple-barrier 신호(PT_SL=(2,1), NUM_DAYS=30, THRESHOLD=0.60) 위에
baseline(고정사이즈) vs treatment(레짐인식 동적사이징)을 5-seed walk-forward로 비교.

prereg 문서(docs/prereg_regime_aware_sizing.md) 4~5절의 검증 프로토콜/기각기준을
그대로 구현한다.

⚠️ 전제 (실행 전 반드시 완료):
    1. estimate_fixed_params.py 실행 -> SIGMA_TARGET, BASE_UNIT_SIZE 확정
       -> volatility_sizing.py, combine_sizing.py 상수에 채워넣기
    2. quant_unsupervised에서 저장된 K-means 모델(centroid) + 정규화 min/max를
       이 레포로 가져와서 regime_penalty.py의 anomaly_score 계산 로직 완성
       (지금은 미완성 -- compute_anomaly_score_normalized()가 TODO 상태)

이 두 전제가 안 채워진 채로 실행하면 assert에서 바로 멈추게 만들어 둠
(조용히 잘못된 값으로 진행하는 것 방지).

사용법 (레포 루트에서):
    python -m src.scripts.backtest_sizing_ablation   (레포 루트에서 실행)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.feature_engineering_triple_barrier import build_triple_barrier_dataset, FEATURE_COLS_BASE
from src.sizing.confidence_sizing import compute_confidence_term, ENTRY_THRESHOLD
from src.sizing.volatility_sizing import compute_realized_vol_20d, compute_vol_term, SIGMA_TARGET
from src.sizing.regime_penalty import compute_regime_penalty, REGIME_FEATURES
from src.sizing.combine_sizing import combine, BASE_UNIT_SIZE

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

TICKER_KRX = "064350"
TICKER = "064350.KS"
PT_SL = (2, 1)
NUM_DAYS = 30
THRESHOLD = ENTRY_THRESHOLD  # confidence_sizing.py와 반드시 동일해야 함 -- 신호 재튜닝 금지
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS
ROUND_TRIP_COST = 0.002
SEEDS = [42, 1, 7, 123, 2024]

# prereg 5절 기각기준
MDD_DETERIORATION_TOLERANCE = 0.0  # treatment MDD가 baseline보다 조금이라도 나빠지면 위반으로 취급
CONCENTRATION_YEAR_SHARE_LIMIT = 0.5  # 특정 연도가 개선분의 50% 이상이면 국면집중형


def config_label() -> str:
    return f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"


def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{config_label()}_base.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    else:
        df = build_triple_barrier_dataset(ticker=TICKER, pt_sl=PT_SL, num_days=NUM_DAYS)
        df.to_csv(path)
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def walk_forward_splits(n_rows, train_size, test_size, step, embargo):
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = range(start, start + train_size)
        test_start = start + train_size + embargo
        test_idx = range(test_start, test_start + test_size)
        splits.append((train_idx, test_idx))
        start += step
    return splits


def compute_anomaly_score_normalized(df: pd.DataFrame, row_idx: int) -> float:
    """
    TODO(전제 2): quant_unsupervised의 K-means centroid + 정규화 min/max를 로드해서
    실제 거리 기반 anomaly score를 계산할 것. 지금은 완성 전이라 실행 시 에러를 낸다
    -- placeholder 값(예: 0.5)으로 조용히 진행하면 regime_penalty가 항상 같은 값이
    나와서 "국면 penalty 효과가 있다/없다"는 결론 자체가 무의미해진다.
    """
    raise NotImplementedError(
        "quant_unsupervised의 K-means centroid/정규화 min-max를 연결해야 함 "
        "(regime_penalty.py 상단 주석 참고, 재학습 금지 -- predict/거리계산만)"
    )


def generate_trades_with_sizing(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """진입 시점/방향은 baseline과 treatment 완전히 동일 -- 사이즈 계산에 필요한
    m_confidence, vol_term, regime_penalty, final_size를 전부 같은 거래 로그에 붙여서 반환."""
    X = df[FEATURE_COLS_BASE]
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)
    realized_vol = compute_realized_vol_20d(df["Close"].pct_change())

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        if y_train.nunique() < 2:
            continue

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=seed,
        )
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

                    anomaly_score = compute_anomaly_score_normalized(df, row_idx)

                    trades.append({
                        "entry_date": df.index[row_idx],
                        "exit_date": df.index[exit_row],
                        "year": df.index[row_idx].year,
                        "pred_proba": proba[i],
                        "hist_vol_20d": realized_vol.iloc[row_idx],
                        "anomaly_score_normalized": anomaly_score,
                        "net_return": net_return,
                    })
                i += max(holding, 1)
            else:
                i += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df

    trades_df["m_confidence"] = compute_confidence_term(trades_df["pred_proba"]).values
    trades_df["vol_term"] = compute_vol_term(trades_df["hist_vol_20d"]).values
    trades_df["regime_penalty"] = compute_regime_penalty(trades_df["anomaly_score_normalized"]).values
    trades_df["final_size"] = combine(
        trades_df["m_confidence"], trades_df["vol_term"], trades_df["regime_penalty"]
    ).values

    return trades_df


def sharpe_ratio(pnl: pd.Series) -> float:
    if pnl.std() == 0 or len(pnl) < 2:
        return float("nan")
    trades_per_year = 252 / NUM_DAYS
    return (pnl.mean() / pnl.std()) * np.sqrt(trades_per_year)


def max_drawdown(pnl: pd.Series) -> float:
    cum = (1 + pnl).cumprod()
    running_max = cum.cummax()
    return ((cum - running_max) / running_max).min()


def run_one_seed(df: pd.DataFrame, seed: int) -> dict:
    trades = generate_trades_with_sizing(df, seed)
    if trades.empty:
        return {"seed": seed, "n_trades": 0}

    baseline_pnl = BASE_UNIT_SIZE * trades["net_return"]
    treatment_pnl = trades["final_size"] * trades["net_return"]

    return {
        "seed": seed,
        "n_trades": len(trades),
        "sharpe_baseline": sharpe_ratio(baseline_pnl),
        "sharpe_treatment": sharpe_ratio(treatment_pnl),
        "mdd_baseline": max_drawdown(baseline_pnl),
        "mdd_treatment": max_drawdown(treatment_pnl),
        "_trades": trades,  # 연도별 분해용, 요약 출력 시 제외
    }


def check_year_concentration(all_results: list) -> str:
    """prereg 5절 -- 특정 연도가 개선분의 50% 이상이면 국면집중형으로 태깅."""
    frames = []
    for r in all_results:
        if r.get("n_trades", 0) == 0:
            continue
        t = r["_trades"].copy()
        t["baseline_pnl"] = BASE_UNIT_SIZE * t["net_return"]
        t["treatment_pnl"] = t["final_size"] * t["net_return"]
        t["improvement"] = t["treatment_pnl"] - t["baseline_pnl"]
        frames.append(t)
    if not frames:
        return "판정불가 (거래 없음)"

    combined = pd.concat(frames)
    by_year = combined.groupby("year")["improvement"].sum()
    total_improvement = by_year.sum()
    if total_improvement <= 0:
        return "개선분 자체가 없음 -- 국면집중 여부 판단 불필요"

    max_share = (by_year.max() / total_improvement) if total_improvement != 0 else float("nan")
    flagged = max_share >= CONCENTRATION_YEAR_SHARE_LIMIT
    return f"최대 연도 기여 비중={max_share:.1%} {'[국면집중형 의심]' if flagged else '[정상]'}"


def judge(summary: pd.DataFrame) -> str:
    """prereg 5절 기각기준 그대로 적용."""
    valid = summary[summary["n_trades"] > 0]
    if len(valid) < len(SEEDS):
        return "[판정불가] 일부 seed에서 거래가 생성되지 않음"

    sharpe_improved = valid["sharpe_treatment"] > valid["sharpe_baseline"]
    mdd_not_worse = valid["mdd_treatment"] >= valid["mdd_baseline"] - MDD_DETERIORATION_TOLERANCE

    n_sharpe_improved = sharpe_improved.sum()
    n_mdd_worse = (~mdd_not_worse).sum()

    if n_sharpe_improved < len(SEEDS):
        return f"[실패] Sharpe 개선이 5/5가 아님 ({n_sharpe_improved}/{len(SEEDS)})"
    if n_mdd_worse >= 3:
        return f"[실패] MDD 악화가 3개 이상 seed에서 발생 ({n_mdd_worse}/{len(SEEDS)}) -- 방어적이지 않은 사이징"
    if (sharpe_improved & mdd_not_worse).all():
        return "[성공 - 배포 검토 가능] 5/5 seed에서 Sharpe 개선 + MDD 악화 없음"
    return "[부분통과] 배포 부적합 -- Sharpe/MDD 중 하나만 만족"


def main():
    df = load_dataset()

    all_results = [run_one_seed(df, seed) for seed in SEEDS]
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "_trades"} for r in all_results])
    print(summary.to_string(index=False))

    print(f"\n연도별 집중도 점검: {check_year_concentration(all_results)}")
    print(f"\n종합 판정: {judge(summary)}")


if __name__ == "__main__":
    main()