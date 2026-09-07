# 방법론 리서치 요약 — 레짐인식 포지션 사이징

> 전체 리서치(페어트레이딩 포함)는 대화 중 별도 산출물로 제공됨. 여기는
> `quant_position_sizing`에 필요한 부분만 발췌/정리.

## 핵심 공식

### 1) 확률 → 베팅크기 (López de Prado, *Advances in Financial Machine Learning*, Ch.10)

```
z = (p - 1/K) / sqrt(p(1-p))       # K=클래스 수, 이진분류면 K=2
m = 2 * Φ(z) - 1                    # Φ = 표준정규 CDF
```

p가 base rate(0.5)면 m=0 (베팅 안 함), 신뢰도 증가에 따라 단조증가, ±1로 saturate.

### 2) Kelly criterion / half-Kelly

```
f* = (bp - q) / b          # 베팅형, b=odds, p=승률, q=1-p
```

Full Kelly는 추정오차에 매우 취약 (50%+ 드로다운 흔함). **half-Kelly는 최대
복리성장률의 약 75%를 유지하면서 분산을 약 75% 감소**시킴 — Thorp을 비롯한 실무
표준.

### 3) Volatility targeting

```
vol_term = sigma_target / sigma_realized
```

목표변동성을 유지하도록 exposure를 조정. 변동성이 튀면 자동으로 사이즈가
줄어드는 방어적 메커니즘.

### 4) 레짐/이상치 penalty

HMM/클러스터링으로 국면 확률을 추정하고, 불리한 국면일수록 exposure를 연속적으로
축소하는 접근이 학술·실무에 존재 (Guidolin & Timmermann 2007 등). Binary
on/off보다 연속 스케일링이 우월하다고 보고됨.

## 결합 방법론 — 왜 곱셈적(multiplicative)인가

- **곱셈적**: `size = base × s_conf × s_vol × s_regime`. 어느 한 요소가 0에
  가까우면 전체도 작아짐 (방어적). 각 항이 [0,1]이면 자연스럽게 bounded.
- **가산적(대안, 채택 안 함)**: 위험요소가 나빠도 다른 항이 보완해서 포지션이
  남을 수 있음 — 초심자 프로젝트에는 비방어적이라 판단해 배제.

## 검증 시 주의 — "사이징 파라미터 자체의 overfitting"

사이징 공식의 파라미터(Kelly fraction, vol clip 범위, regime penalty clip
범위)를 in-sample에서 반복 튜닝하는 것 자체가 새로운 형태의 데이터 마이닝이다.
이 프로젝트에서는 모든 파라미터를 **데이터 확인 전 사전 고정**하고, 딱 한 번만
추정 후 코드에 하드코딩하는 방식으로 이 위험을 통제한다 (prereg 문서 2절 참고).

다중 시도가 발생하면 Deflated Sharpe Ratio(Bailey & López de Prado 2014)로
보정할 것.

## 참고문헌

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  — Ch.3 (triple-barrier, meta-labeling), Ch.10 (bet sizing), Ch.11-12
  (백테스트 위험, cross-validation)
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio."
  *Journal of Portfolio Management*, 40(5), 94-107.
- Guidolin, M. & Timmermann, A. (2007). Regime-switching in portfolio
  optimization.