# Pre-registration: 레짐인식 포지션 사이징 (Regime-Aware Bet Sizing)

**작성일**: 2026-09-04
**레포**: `quant_position_sizing`
**브랜치**: `experiment/regime-aware-position-sizing`

---

## 0. 배경 및 목적

`quant_xgboost`에서 유일하게 실전 배포 후보로 살아남은 신호(triple-barrier +
XGBoost BASE, 064350, pt_sl=(2,1)/num_days=30/threshold=0.60, 5-seed 풀링 통과)는
지금까지 **"진입할지 말지"만 결정**했고, 진입 시 베팅 크기는 항상 고정(전액 순차
진입)이었다.

이 실험은 새로운 신호를 찾는 게 아니라, **이미 검증된 신호 위에 "얼마나 베팅할지"를
결정하는 사이징 레이어를 추가**하는 것이 목적이다. 방법론 조사(2026-09,
`docs/research_position_sizing_methodology.md` 참고)에 따라 아래 세 요소를
곱셈적으로 결합한 공식을 직접 설계한다.

> **중요한 무결성 조건**: 기존 triple-barrier 신호(진입 여부, 방향, threshold=0.60)는
> 이 실험에서 **절대 재튜닝하지 않는다**. 사이징 레이어만 새로 추가한다. 신호 자체를
> 건드리면 이 실험의 검증 대상이 "사이징 효과"가 아니라 "신호 재튜닝 효과"로
> 오염된다.

---

## 1. 가설

**H1**: 모델 신뢰도(예측확률) + 변동성 + 국면 이례성을 결합한 동적 포지션 사이징이,
동일 신호에 대해 고정 사이즈(fixed-size) 진입보다 **위험조정수익률(Sharpe)을
개선하면서 동시에 MDD를 악화시키지 않는다**.

**H0 (귀무가설)**: 동적 사이징은 고정 사이즈 대비 Sharpe 개선이 없거나, Sharpe가
개선되더라도 MDD가 악화된다 (트레이드오프일 뿐 순수 개선이 아니다).

---

## 2. 사이징 공식 (사전 고정, 데이터 확인 전 확정)

```
final_size = base_unit_size × m_confidence × vol_term × regime_penalty
final_size = clip(final_size, 0, max_position)
```

### 2-1. 신뢰도 항 (m_confidence) — López de Prado 확률 기반 사이징

기존 triple-barrier XGBoost 모델이 이미 출력하는 예측확률 `p`를 그대로 재사용
(재학습 없음).

```
z = (p - 0.5) / sqrt(p × (1 - p))          # K=2 (이진분류)
m_confidence = 2 × Φ(z) - 1                 # Φ = 표준정규 CDF
```

- p = 0.60(현재 threshold) → m_confidence ≈ 0.16
- p = 0.75 → m_confidence ≈ 0.68
- p = 0.90 → m_confidence ≈ 0.97
- **사전 고정**: threshold=0.60 미만이면 애초에 진입 안 하므로 m_confidence는
  항상 [0.16, 1.0] 구간

### 2-2. 변동성 항 (vol_term) — Volatility targeting

```
vol_term = clip(sigma_target / sigma_realized_20d, 0.3, 1.5)
```

- `sigma_target`: 064350 전체 기간 realized vol의 **중앙값**으로 사전 고정
  (데이터 확인 후 딱 한 번 계산, 이후 고정값으로 코드에 하드코딩 — 매 fold마다
  다시 계산하지 않음. fold마다 다시 계산하면 그 자체가 look-ahead)
- `sigma_realized_20d`: 매매 시점 기준 과거 20거래일 realized vol (과거 데이터만 사용)
- clip 범위 [0.3, 1.5]는 사전 고정 — 변동성이 극단적으로 낮다고 3배씩 베팅하거나,
  극단적으로 높다고 거의 0으로 죽이는 걸 방지

### 2-3. 레짐 penalty 항 (regime_penalty) — quant_unsupervised 재사용

`quant_unsupervised`의 K-means 국면 클러스터링 인프라를 그대로 재사용 (재학습 없음,
기존 4개 feature: `hist_vol_20d`, `return_20d`, `bb_width`, `macd_hist`).

```
regime_penalty = clip(1 - anomaly_score_normalized, 0.3, 1.0)
```

- `anomaly_score_normalized`: 현재 시점이 속한 클러스터가 "고변동성+급등 모멘텀"
  국면(quant_unsupervised에서 이례적으로 확인된 프로필)에 얼마나 가까운지를
  [0,1]로 정규화한 거리 — 국면 클러스터 중심까지의 거리를 min-max 정규화
- clip 하한 0.3 — 국면이 아무리 이례적이어도 완전히 0으로 죽이지 않음 (신호 자체가
  이미 threshold=0.60을 통과했으므로 어느 정도 신뢰는 유지)

### 2-4. 전역 스케일 — Half-Kelly

```
base_unit_size = 0.5 × kelly_fraction_estimate
```

- `kelly_fraction_estimate`: 기존 백테스트 로그(triple-barrier 178~188건 거래)의
  실현 승률/손익비로 계산한 f* = (bp - q)/b, 데이터 확인 후 **한 번만 계산해서
  고정** (매 fold 재추정 안 함 — in-sample overfitting 방지)
- 0.5 = half-Kelly (표준 관행, 추정오차 방어)

---

## 3. 비교 설계 (Baseline vs Treatment)

| | Baseline (기존) | Treatment (신규) |
|---|---|---|
| 진입 신호 | triple-barrier XGBoost BASE, threshold=0.60 (동일) | 동일 |
| 포지션 사이즈 | 고정 (전액 순차 진입) | `final_size` 공식 |
| 거래 시점 | 동일 fold, 동일 진입/청산 규칙 | 동일 |

**둘 다 같은 신호·같은 거래 시점을 쓰고, 사이즈 결정 로직만 다르다.** 이렇게 해야
성능 차이가 순수하게 "사이징 효과"인지 확인 가능 (기존 프로젝트 ablation 컨벤션과
동일한 원칙).

---

## 4. 검증 프로토콜

1. **5-seed walk-forward** (42, 1, 7, 123, 2024), embargo = horizon(30일) — 기존
   064350 triple-barrier 검증과 동일한 fold 구성 재사용
2. **평가지표**: Sharpe ratio, MDD, 총수익률을 baseline과 treatment 각각 계산
3. **통과 기준**:
   - Sharpe: treatment > baseline, 5-seed 중 **5/5**에서 개선
   - MDD: treatment의 MDD가 baseline보다 **악화되지 않음** (같거나 개선), 5/5
   - 위 두 조건을 **동시에** 만족해야 통과. 하나만 만족하면 "트레이드오프"로 기록하고
     `[부분통과]`로 태깅 (배포 부적합)
4. **Deflated Sharpe Ratio 체크**: 사이징 공식 자체는 파라미터(sigma_target clip
   범위, regime penalty clip 범위, Kelly fraction)를 전부 사전 고정했으므로 다중
   시도 문제는 최소화되지만, 혹시 이번 실험 중 파라미터를 수정하게 되면 **시도
   횟수를 기록**하고 최종 Sharpe를 Bailey & López de Prado(2014) 공식으로 deflate
5. **연도별 분해**: 기존 프로젝트 컨벤션대로 손익을 연도별로 분해해서 국면집중형/
   구조적저하형 실패가 있는지 확인 (baseline 대비 treatment의 손실연도 개수 변화도
   체크 — 외국인보유율 실험에서 나온 "구조적 저하형" 실패 패턴 재확인용)

---

## 5. 기각 기준 (사전 명시)

다음 중 하나라도 해당하면 `[실패]` 태깅:

- Sharpe 개선이 5/5 seed에서 일관되지 않음 (노이즈 의심)
- Sharpe는 개선되나 MDD가 5/5 중 3개 이상에서 악화 (방어적이지 않은 사이징 = 설계
  실패)
- 연도별 분해에서 baseline 대비 손실연도 개수가 증가 (구조적 저하형 패턴 재발)
- 특정 연도 하나가 개선분의 50% 이상을 차지 (국면집중형)

---

## 6. Git 컨벤션

- 브랜치: `experiment/regime-aware-position-sizing`
- 커밋: `docs:` pre-registration 먼저 단독 커밋 → `feat:` 스크립트별 분리 커밋
- 병합: `--no-ff`
- 검증 종료 시 annotated tag (`v-regime-sizing-pass` / `v-regime-sizing-fail` /
  `v-regime-sizing-partial`)

---

## 7. 실행 순서 (계획)

`src/`가 역할별 하위 패키지(`data/`, `sizing/`, `scripts/`)로 나뉘어 있어서, 레포
루트에서 `-m` 옵션으로 실행한다 (`quant_seq_model`과 동일한 이유 — 중첩 패키지 구조).

```
python -m src.scripts.estimate_fixed_params      # sigma_target, kelly_fraction 1회 계산 후 코드에 하드코딩
# (m_confidence/vol_term/regime_penalty/combine은 src/sizing/ 밑의 순수 계산 모듈이라
#  단독 실행 대상이 아니라 위/아래 스크립트가 import해서 씀)
python -m src.scripts.backtest_sizing_ablation   # baseline(고정사이즈) vs treatment(동적사이징), 5-seed
```