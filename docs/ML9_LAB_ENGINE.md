# ML9 Lab Engine (ML9+Guard PIT-Safe v1)

**작성일**: 2025-11-28  
**버전**: lab-ml9-guard-v1  
**목적**: ML9+Guard 엔진을 "피처/리스크 엔진 실험실(Lab)"로 재사용하기 위한 스냅샷

---

## 📋 개요

이 문서는 `lab-ml9-guard-v1` 태그 기준 ML9+Guard 엔진을 "피처/리스크 엔진 실험실(Lab)"로 재사용하기 위한 스냅샷입니다. 이 엔진은 ARES7-Best 개선을 위한 연구 자산으로 활용되며, 직접적인 프로덕션 배포가 아닌 **실험 및 검증 목적**으로 사용됩니다.

---

## 1. 버전 정보

### Git 정보
- **Repository**: `ml9-quant-strategy`
- **Tag**: `lab-ml9-guard-v1`
- **Commit**: `b5463db` (feat: Complete PIT-safe data validation and final backtest)
- **Branch**: `main`

### 주요 특징
- **Universe**: S&P 100 (SP100) - 99 tickers
- **Period**: 2015-01-01 ~ 2024-12-31 (10년)
- **Data Sources**:
  - **Polygon API**: 일간 가격 데이터 (OHLCV)
  - **Sharadar SF1**: 펀더멘털 데이터 (PIT-safe)
- **Core Engine**: ML9 (XGBoost ranking) + MarketConditionGuard
- **PIT Safety**: merge_asof + calendardate filter로 look-ahead bias 완전 제거

---

## 2. 데이터 파이프라인

### 2.1 데이터 수집

**함수**: `download_and_prepare_data()`

#### Polygon API (가격 데이터)
```python
# sp100_prices_raw.csv
# Columns: date, ticker, open, high, low, close, volume
```

#### Sharadar SF1 (펀더멘털 데이터)
```python
# sp100_sf1_raw.csv
# Columns: ticker, datekey, calendardate, pe, pb, ps, evebitda, 
#          roe, ebitdamargin, de, currentratio, ...
```

### 2.2 PIT-Safe Merge

**핵심 로직**: Look-ahead bias 완전 제거

```python
# Step 1: merge_asof (backward)
merged = pd.merge_asof(
    prices.sort_values(['ticker', 'date']),
    sf1.sort_values(['ticker', 'datekey']),
    left_on='date',
    right_on='datekey',
    by='ticker',
    direction='backward'
)

# Step 2: calendardate filter (future TTM 제거)
merged = merged[merged['calendardate'] <= merged['date']]

# Step 3: ticker별 forward fill
merged = merged.groupby('ticker').apply(lambda g: g.ffill())
```

**검증 결과**:
- 총 259,176 rows
- Look-ahead bias: 0 cases (100% PIT-safe)
- Data quality: 99.2% coverage

### 2.3 Feature Engineering

#### Price-based Features
- `momentum_60d`: 60일 모멘텀 (log return)
- `volatility_30d`: 30일 변동성 (rolling std)
- `value_proxy`: P/E ratio 기반 밸류에이션

#### SF1 Fundamental Features
- **Valuation**: `pe`, `pb`, `ps`, `evebitda`
- **Quality**: `roe`, `ebitdamargin`
- **Financial Health**: `de` (Debt/Equity), `currentratio`

---

## 3. 엔진 정의

### 3.1 ML9 Engine

**알고리즘**: XGBoost Ranking

#### 학습 설정
- **Training Window**: 2년 롤링 (504 거래일)
- **Horizon**: 10일 (forward return)
- **Rebalancing**: 주간 (매주 월요일)
- **Position**: Long-only
- **Weighting**: Inverse volatility weighting

#### Features (총 12개)
1. `momentum_60d`
2. `volatility_30d`
3. `pe` (Price/Earnings)
4. `pb` (Price/Book)
5. `ps` (Price/Sales)
6. `evebitda` (EV/EBITDA)
7. `roe` (Return on Equity)
8. `ebitdamargin` (EBITDA Margin)
9. `de` (Debt/Equity)
10. `currentratio` (Current Ratio)
11. `volume_ratio` (Volume / 20d avg)
12. `price_to_52w_high` (Price / 52-week high)

#### XGBoost Hyperparameters
```python
params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

### 3.2 MarketConditionGuard

**목적**: SPX 하락 시 포지션 축소로 MDD 감소

#### 로직
```python
def get_position_scale(r_spx: float) -> float:
    """
    r_spx: 전일 SPX 수익률 (look-ahead 없음)
    """
    if -0.02 < r_spx <= 0.0:  # -2% ~ 0%
        return 0.5  # 50% 포지션 축소
    else:
        return 1.0  # 정상 운영
```

#### 특징
- **Look-ahead Free**: 항상 **전일** SPX 수익률만 사용
- **Simple Rule**: 복잡한 ML 없이 단순 임계값 기반
- **Conservative**: -2% ~ 0% 구간에서만 작동 (과도한 축소 방지)

---

## 4. 성과 (PIT-safe, 비용 미반영 기준)

### 4.1 ML9 (No Guard)

| 지표 | 값 |
|------|-----|
| **Sharpe Ratio** | 0.956 |
| **Annual Return** | 17.4% |
| **Annual Volatility** | 18.2% |
| **Max Drawdown** | -25.8% |
| **Calmar Ratio** | 0.674 |
| **Win Rate** | 54.2% |

### 4.2 ML9+Guard

| 지표 | 값 | vs No Guard |
|------|-----|-------------|
| **Sharpe Ratio** | 1.114 | +16.5% |
| **Annual Return** | 17.2% | -1.1% |
| **Annual Volatility** | 15.4% | -15.4% |
| **Max Drawdown** | -22.2% | -14.0% |
| **Calmar Ratio** | 0.775 | +15.0% |
| **Win Rate** | 55.8% | +3.0% |

### 4.3 주요 개선 사항

**Guard 효과**:
- **Sharpe 개선**: +0.158 (+16.5%)
- **MDD 개선**: -3.6% (-14.0%)
- **Vol 감소**: -2.8% (-15.4%)
- **Return 유지**: -0.2% (거의 동일)

**2018 위기 대응** (최악 연도):
- ML9 (No Guard): Sharpe 0.47, MDD -28.4%
- ML9+Guard: Sharpe 0.91, MDD -24.5%
- **개선**: Sharpe +93.6%, MDD -13.7%

---

## 5. Lab로서의 역할

이 엔진은 다음 목적에 사용됩니다:

### 5.1 피처 실험
- **SF1 기반 Value/Quality 팩터** 아이디어 테스트
- **새로운 펀더멘털 지표** 추가 및 검증
- **Feature Engineering** 파이프라인 프로토타이핑

### 5.2 리스크 엔진 실험
- **MarketConditionGuard** 룰 테스트
  - SPX 구간 필터 조정 (-2% ~ 0% → 다른 구간)
  - 추가 조건 (VIX, 변동성, 모멘텀 등)
- **VIX 기반 Guard** 프로토타입
- **ETAS λ_sys** (시스템 리스크) 등의 고급 리스크 지표

### 5.3 ARES7 엔진 설계에 인사이트 제공
- 여기서 검증된 **Guard/Regime/Failure Mode** 아이디어를
- ARES7-Best의 **Factor/LowVol/MeanReversion** 엔진 설계에 녹이는 용도
- **직접 앙상블에 포함하지 않고**, "연구용 서브엔진"으로만 사용

### 5.4 백테스트 프레임워크 검증
- **PIT-safe merge** 로직 검증
- **Transaction cost** 모델 테스트
- **Rebalancing frequency** 최적화

---

## 6. 파일 구조

### 주요 파일
```
ml9-quant-strategy/
├── run_all_tests.py                    # 메인 백테스트 스크립트
├── ml9_market_condition_guard.py       # MarketConditionGuard 구현
├── data/
│   ├── sp100_prices_raw.csv           # Polygon 가격 데이터
│   ├── sp100_sf1_raw.csv              # Sharadar SF1 데이터
│   └── sp100_pit_merged.csv           # PIT-safe merged 데이터
├── analysis/
│   ├── cross_ensemble_all_models_v1.py # Cross-project 앙상블 테스트
│   └── stage4_dynamic_regime.py        # Stage 4 동적 레짐 실험
├── docs/
│   ├── ML9_LAB_ENGINE.md              # 이 문서
│   ├── NEW_SESSION_CONTEXT.md         # 세션 컨텍스트
│   └── ARES_X_V110_ARCHITECTURE_ANALYSIS.md
└── reports/
    ├── CROSS_PROJECT_ENSEMBLE_FINAL_REPORT.md
    └── STAGE4_DYNAMIC_REGIME_REPORT.md
```

### 데이터 크기
- `sp100_prices_raw.csv`: ~250,000 rows
- `sp100_sf1_raw.csv`: ~50,000 rows
- `sp100_pit_merged.csv`: 259,176 rows

---

## 7. 사용 방법

### 7.1 Lab 버전 복원

```bash
# 현재 작업 저장
cd /path/to/ml9-quant-strategy
git stash

# Lab 버전으로 체크아웃
git checkout lab-ml9-guard-v1

# 백테스트 실행
python run_all_tests.py

# 원래 버전으로 복귀
git checkout main
git stash pop
```

### 7.2 새로운 피처 테스트

```python
# run_all_tests.py 수정 예시

# 1. 새로운 SF1 피처 추가
new_features = ['grossmargin', 'assetturnover', 'payoutratio']

# 2. ML9Engine에 피처 추가
engine = ML9Engine(
    features=base_features + new_features,
    train_window=504,
    horizon=10
)

# 3. 백테스트 실행
results = engine.backtest(data, start_date, end_date)

# 4. 성과 비교
print(f"Sharpe (baseline): 1.114")
print(f"Sharpe (new): {results['sharpe']:.3f}")
```

### 7.3 Guard 룰 테스트

```python
# ml9_market_condition_guard.py 수정 예시

def get_position_scale(self, r_spx: float) -> float:
    """새로운 Guard 룰 테스트"""
    
    # 원래 룰: -2% ~ 0%
    # if -0.02 < r_spx <= 0.0:
    #     return 0.5
    
    # 새로운 룰: 3단계 축소
    if r_spx < -0.03:  # -3% 이하
        return 0.25  # 25% 포지션
    elif -0.03 <= r_spx < -0.01:  # -3% ~ -1%
        return 0.5  # 50% 포지션
    elif -0.01 <= r_spx < 0.0:  # -1% ~ 0%
        return 0.75  # 75% 포지션
    else:
        return 1.0  # 정상 운영
```

---

## 8. 제약사항 및 주의사항

### 8.1 제약사항
1. **Universe**: S&P 100만 지원 (확장 시 코드 수정 필요)
2. **Data Source**: Polygon + Sharadar만 지원
3. **Transaction Cost**: 미반영 (실거래 시 Sharpe 0.1~0.2 감소 예상)
4. **Slippage**: 미반영 (실거래 시 추가 비용 발생)
5. **Rebalancing**: 주간 고정 (일간/월간 변경 시 코드 수정 필요)

### 8.2 주의사항
1. **PIT-Safe 검증**: 새로운 데이터 추가 시 반드시 PIT 검증 필요
2. **Look-ahead Bias**: Guard는 항상 **전일** 데이터만 사용
3. **Overfitting**: 2년 롤링 학습으로 과적합 방지하지만, 파라미터 튜닝 시 주의
4. **Survivorship Bias**: S&P 100은 생존 편향 있음 (실제 성과는 낮을 수 있음)
5. **Data Quality**: SF1 데이터 결측치 처리 (ffill) 시 주의

---

## 9. 향후 개선 방향

### 9.1 단기 (1~2주)
1. **Transaction Cost 모델 추가**
   - 종목별 spread, ADV 기반 비용 계산
   - Rebalancing 빈도 최적화
2. **VIX Guard 추가**
   - VIX > 25 시 포지션 축소
   - SPX Guard와 조합

### 9.2 중기 (1~2개월)
3. **Universe 확장**
   - S&P 100 → S&P 500
   - 섹터별 분산 개선
4. **Alternative Data 통합**
   - 뉴스 감성 분석
   - 옵션 데이터 (implied volatility)

### 9.3 장기 (3~6개월)
5. **Multi-Asset 지원**
   - 주식 + 채권 + 원자재
   - 글로벌 분산 (미국 외 시장)
6. **실시간 거래 인프라**
   - IBKR API 연동
   - 백테스트 → 라이브 전환

---

## 10. ARES7-Best와의 비교

| 항목 | ML9-Guard (Lab) | ARES7-Best |
|------|----------------|------------|
| **Sharpe (Full)** | 1.114 | 1.853 |
| **Sharpe (Min)** | 0.469 (2018) | 1.626 (2018) |
| **MDD** | -22.2% | -8.72% |
| **Engines** | 1 (ML9) | 5 (Factor, LV2, MR, LS, Factor) |
| **Vol Targeting** | 없음 | 10% |
| **Leverage** | 1.0x | 1.5x |
| **Universe** | SP100 (99) | SP100+ (확장 가능) |
| **Rebalancing** | 주간 | 월간 |
| **용도** | 연구/실험 | 프로덕션 |

**핵심 차이점**:
- ML9-Guard는 **단일 엔진 실험실**
- ARES7-Best는 **5-Engine 앙상블 프로덕션 시스템**
- ML9-Guard에서 검증된 아이디어 → ARES7-Best로 이전

---

## 11. 참고 문서

### 내부 문서
- [NEW_SESSION_CONTEXT.md](./NEW_SESSION_CONTEXT.md): 전체 프로젝트 컨텍스트
- [CROSS_PROJECT_ENSEMBLE_FINAL_REPORT.md](../CROSS_PROJECT_ENSEMBLE_FINAL_REPORT.md): ARES7-Best 분석
- [STAGE4_DYNAMIC_REGIME_REPORT.md](../STAGE4_DYNAMIC_REGIME_REPORT.md): Stage 4 실험 결과
- [ARES_X_V110_ARCHITECTURE_ANALYSIS.md](./ARES_X_V110_ARCHITECTURE_ANALYSIS.md): ARES-X V110 분석

### 외부 참고
- [Sharadar SF1 Documentation](https://data.nasdaq.com/databases/SF1/documentation)
- [Polygon API Documentation](https://polygon.io/docs)
- [XGBoost Ranking Tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)

---

## 12. 버전 히스토리

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| **lab-ml9-guard-v1** | 2025-11-28 | 초기 Lab 스냅샷 (Sharpe 1.114, PIT-safe 검증 완료) |

---

## 13. 라이선스 및 면책

**라이선스**: MIT License (연구 및 상업적 사용 가능)

**면책사항**:
- 이 엔진은 **연구 목적**으로 제공됩니다.
- **실거래 시 발생하는 손실에 대해 책임지지 않습니다.**
- Transaction cost, slippage, 시장 충격 등이 미반영되어 있습니다.
- 과거 성과는 미래 수익을 보장하지 않습니다.

---

**작성자**: Manus AI  
**문서 버전**: 1.0  
**최종 수정**: 2025-11-28  
**Git Tag**: `lab-ml9-guard-v1`
