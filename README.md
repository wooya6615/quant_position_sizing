# quant_position_sizing

기존 프로젝트에서 유일하게 검증 통과한 triple-barrier + XGBoost 신호
(BASE, 064350, threshold=0.60, pt_sl=(2,1), num_days=30) 위에, **방향은 절대
건드리지 않고** "얼마나 살지(bet sizing)"만 새로 설계하는 실험.

> 자세한 가설/공식/검증기준은 [`docs/prereg_regime_aware_sizing.md`](docs/prereg_regime_aware_sizing.md),
> 방법론 근거는 [`docs/research_position_sizing_methodology.md`](docs/research_position_sizing_methodology.md) 참고.

## 핵심 아이디어

```
final_size = base_size × confidence_term(p) × volatility_term(σ) × regime_penalty(cluster)
```

- **confidence_term**: López de Prado 확률→베팅크기 공식 (기존 분류기의 예측확률을
  그대로 입력받음, 재학습 안 함)
- **volatility_term**: σ_target / σ_realized_20d, [0.3, 1.5] 클립
- **regime_penalty**: `quant_unsupervised`에서 이미 학습된 K-means 클러스터를
  재사용(재학습 안 함), [0.3, 1.0] 클립

곱셈적 결합이라 셋 중 하나라도 낮으면 최종 사이즈도 자동으로 축소된다.

## 구조

```
docs/
  prereg_regime_aware_sizing.md            # 가설/공식/검증기준 (사전등록, 필독)
  research_position_sizing_methodology.md  # 방법론 근거 (López de Prado, Kelly, vol targeting)
src/
  data/
    feature_engineering_triple_barrier.py  # quant_xgboost에서 복사해올 것 (아래 TODO 1)
  regime/                      # quant_unsupervised에서 이식 (국면 클러스터링, 064350 전용 fit)
    price_data.py
    regime_features.py
    regime_clustering.py
  sizing/                     # 사이징 공식 자체 -- 순수 계산 로직만, I/O 없음
    confidence_sizing.py        # 신뢰도항 m_confidence (López de Prado 확률->베팅크기)
    volatility_sizing.py         # 변동성항 vol_term (volatility targeting)
    regime_penalty.py             # 국면 penalty (anomaly_score_normalized를 입력받는 순수함수)
    combine_sizing.py              # 세 항 곱셈적 결합 -> final_size
  scripts/                    # 실행 진입점 -- 데이터 로드 + sizing 조합 + 백테스트
    estimate_fixed_params.py    # SIGMA_TARGET, BASE_UNIT_SIZE(half-Kelly) 1회성 계산
    fit_regime_model_064350.py   # 064350 국면 클러스터링 1회성 fit + 저장 (models/*.joblib)
    backtest_sizing_ablation.py   # baseline(fixed-size) vs treatment 5-seed 비교 + 판정
models/                   # fit_regime_model_064350.py가 저장하는 .joblib (gitignore 처리)
data/                     # 생성되는 중간 산출물 (gitignore 처리)
requirements.txt
```

`sizing/`은 공식 자체(입력→출력 순수함수)만 담고, `scripts/`가 데이터를 불러와서
`sizing/`을 조합해 실제로 백테스트를 돌린다 — quant_xgboost의 여러 `compute_pbo_*.py`
스크립트들이 `feature_engineering_triple_barrier.py`를 공유해서 쓰던 것과 같은 분리
원칙 (계산 로직과 실행 스크립트를 분리).

**`quant_unsupervised`와의 차이**: `quant_unsupervised`는 모델을 저장하지 않고 실행할
때마다 그 자리에서 새로 fit하며, 118990/052690에만 적용해봤을 뿐 064350에는 적용된
적이 없었다. 그래서 이 레포는 `quant_unsupervised`의 세 모듈(`price_data.py`,
`regime_features.py`, `regime_clustering.py`)을 `src/regime/`로 이식하고,
`fit_regime_model_064350.py`로 064350에 대해 딱 한 번 fit해서 그 결과
(`models/064350_regime_model.joblib`)를 이후 계속 재사용한다 — 이 스크립트를
재실행하는 것은 곧 재학습이므로, 결과를 보고 나서 재실행하지 않는다.

## 실행 전 반드시 채워야 할 것 (TODO)

1. **`feature_engineering.py`, `labeling_triple_barrier.py`를 `quant_xgboost`에서
   `src/data/`로 복사해올 것** (`feature_engineering_triple_barrier.py`가 이 둘을
   import함 -- 자기완결적 레포 컨벤션).
2. `estimate_fixed_params.py` 실행 → 콘솔에 나오는 `SIGMA_TARGET`, `BASE_UNIT_SIZE`
   값을 `src/sizing/volatility_sizing.py` / `src/sizing/combine_sizing.py` 상수에
   손으로 채워넣을 것 (자동 반영 안 함 — "데이터 보고 나서 조정 금지"를 코드로도
   강제하기 위해 의도적으로 수동 단계로 분리).
3. `fit_regime_model_064350.py` 실행 → `models/064350_regime_model.joblib` 생성.
   이 스크립트는 딱 한 번만 실행할 것 — 재실행하면 K-means가 다시 fit되므로 곧
   재학습이다.

## 실행 순서 (레포 루트에서, 패키지 구조라 `-m` 필요 — `quant_seq_model`과 동일한 이유)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 1. quant_xgboost에서 feature_engineering.py, labeling_triple_barrier.py를 src/data/로 복사해오기
# 2. 사전 고정 파라미터 계산 (SIGMA_TARGET, half-Kelly)
python -m src.scripts.estimate_fixed_params
# 3. 위 출력값을 src/sizing/volatility_sizing.py, src/sizing/combine_sizing.py 상수에 채워넣기
# 4. 064350 국면 클러스터링 1회성 fit + 저장 (재실행 금지)
python -m src.scripts.fit_regime_model_064350
# 5. baseline vs treatment 5-seed 비교 + 판정
python -m src.scripts.backtest_sizing_ablation
```

## 절대 규칙

- 기존 triple-barrier 분류기(방향, threshold=0.60)를 이 레포에서 재학습/재튜닝하지 않는다.
- `quant_unsupervised`의 K-means를 재학습하지 않는다 (predict만 수행).
- 사전 고정 파라미터(σ_target, clip 범위, cluster→penalty 매핑)를 결과를 보고 나서
  조정하지 않는다. 조정이 필요하면 PRE_REGISTRATION 문서에 addendum을 남기고
  조정 시도 자체를 기록한다.