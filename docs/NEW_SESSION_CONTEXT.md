# 🚀 Quant Ensemble Strategy - New Session Context

> **목적**: 이전 세션의 모든 작업 내용을 새 세션으로 완벽하게 이어가기 위한 컨텍스트 문서

---

## 📋 프로젝트 개요

### 리포지토리 정보
- **이름**: `quant-ensemble-strategy` (GitHub: `yhun1542/ml9-quant-strategy`)
- **목표**: 미국 대형주(SP100) 대상 Sharpe 1.5~2.0, MDD -15~-20% 수준의 안정적인 앙상블 퀀트 전략
- **핵심 엔진**: 
  - **ML9**: XGBoost 기반 머신러닝 전략 (momentum + value + volatility)
  - **QV**: Quality-Value 펀더멘털 팩터 전략
  - **Guard**: ML9 실패 구간 보호 레이어

### 데이터 파이프라인
- **가격 데이터**: Polygon API (SP100, 2014-2024)
- **펀더멘털**: Sharadar SF1 (Nasdaq Data Link, dimension=ART)
- **핵심 혁신**: **Point-in-Time (PIT) merge_asof** 구현으로 look-ahead bias 완전 제거

---

## 🎯 현재 상태 (2024-11-28 기준)

### ✅ 완료된 작업

#### 1. **데이터 파이프라인 (PIT 통합 완료)**
```python
# 핵심: merge_asof를 사용한 PIT 병합
data = pd.merge_asof(
    prices_sorted,
    sf1_sorted,
    left_on="date",
    right_on="datekey",
    by="ticker",
    direction="backward",  # 각 날짜에서 이전에 발표된 가장 최근 데이터 사용
    allow_exact_matches=True,
)
```

**결과**:
- ✅ 259,176 rows, 99 tickers (SP100)
- ✅ Fundamental 데이터 100% 채워짐 (이전: 0 rows)
- ✅ 타임존 정규화 완료
- ✅ `currentratio` NaN → median 대체

#### 2. **ML9 Engine (머신러닝 전략)**
- **구조**: Walk-forward 3-window backtest (2015-2018, 2018-2021, 2021-2024)
- **모델**: XGBoost (multi:softprob, 3-class)
- **피처**: `momentum_60d_rank`, `value_proxy_inv_rank`, `volatility_30d_rank`
- **리밸런싱**: 월말 (월 1회)
- **포지션**: Top 20% quantile, equal-weight

**백테스트 결과** (2018-2024):
- Sharpe Ratio: **0.80**
- 연간 수익률: **14.84%**
- 연간 변동성: **18.63%**
- 최대 낙폭: **-28.37%**
- 승률: **49.17%**
- 거래 횟수: **785회**

#### 3. **QV Engine (Quality-Value 팩터 전략)**
- **팩터 구성**:
  - **Value** (50%): PE, PB, PS, EV/EBITDA
  - **Quality** (50%): ROE (35%), EBITDA Margin (25%), D/E (25%), Current Ratio (15%)
- **리밸런싱**: 월말
- **가중치**: Inverse volatility weighting
- **포지션**: Top 30% quantile

**백테스트 결과** (2015-2024):
- Sharpe Ratio: **0.81**
- 연간 수익률: **13.42%**
- 연간 변동성: **16.59%**
- 최대 낙폭: **-31.11%**
- 승률: **54.25%**
- 거래 횟수: **2,516회**

#### 4. **MarketConditionGuard (ML9 보호 레이어)**
- **목적**: SPX 일간 수익률 -2%~0% 구간에서 ML9 포지션 축소
- **근거**: Failure Mode 분석 결과 해당 구간에서 ML9 Sharpe 크게 음수
- **효과** (2023-2024 테스트):
  - Sharpe: 1.8 → **4.4+**
  - MDD: -10% → **-5%**

---

## 📁 핵심 파일 구조

```
quant-ensemble-strategy/
├── run_all_tests.py              # 🔥 통합 백테스팅 스크립트 (PIT + ML9 + QV)
├── data/
│   ├── sp100_prices_raw.csv      # Polygon 가격 데이터
│   ├── sp100_sf1_raw.csv         # SF1 펀더멘털 데이터
│   └── sp100_merged_data.csv     # PIT 병합 완료 데이터
├── results/
│   ├── ml9_returns.csv           # ML9 일별 수익률
│   ├── ml9_metrics.json          # ML9 성과 지표
│   ├── qv_returns.csv            # QV 일별 수익률
│   └── qv_metrics.json           # QV 성과 지표
├── engines/
│   ├── ml_xgboost_v9_ranking.py  # ML9 엔진 (원본)
│   └── factor_quality_value_v2_1.py  # QV 엔진 (원본)
├── modules/
│   └── market_guard_ml9.py       # MarketConditionGuard
├── utils/
│   └── fundamental_factors.py    # QV 팩터 계산 유틸
├── docs/
│   ├── session_history/          # 이전 세션 문서들
│   └── NEW_SESSION_CONTEXT.md    # 이 문서
└── FINAL_REPORT.md               # 최종 백테스트 보고서
```

---

## 🔧 핵심 기술 구현

### 1. Point-in-Time (PIT) 데이터 병합

**문제**: 기존 `merge(on=["date", "ticker"])`는 분기별 SF1 + 일별 가격이라 대부분 NaN

**해결**:
```python
# Ticker별로 merge_asof 실행 (by 파라미터의 정렬 이슈 회피)
all_merged = []
for ticker in tickers:
    p_tick = prices_df[prices_df['ticker'] == ticker].sort_values('date')
    s_tick = sf1_df[sf1_df['ticker'] == ticker].sort_values('datekey')
    
    merged = pd.merge_asof(
        p_tick, s_tick,
        left_on="date", right_on="datekey",
        direction="backward",
        allow_exact_matches=True,
    )
    all_merged.append(merged)

data = pd.concat(all_merged, ignore_index=True)

# Ticker 컬럼 이름 충돌 해결
if 'ticker_x' in data.columns:
    data = data.rename(columns={'ticker_x': 'ticker'})
    data = data.drop(columns=['ticker_y'])
```

### 2. 리밸런싱 날짜 매칭

**문제**: 월말 리밸런싱 날짜(타임존 없음)와 데이터 날짜(UTC+5) 불일치

**해결**:
```python
# 리밸런싱 날짜를 실제 거래일로 매핑
rebal_dates_actual = []
for rebal_date in rebal_dates:
    available_dates = factors.index.get_level_values('date').unique()
    closest_date = min(available_dates, key=lambda x: abs((x - rebal_date).total_seconds()))
    if abs((closest_date - rebal_date).days) <= 3:
        rebal_dates_actual.append(closest_date)
```

### 3. MultiIndex 정렬

**문제**: `UnsortedIndexError: 'Key length (1) was greater than MultiIndex lexsort depth (0)'`

**해결**:
```python
# ML9Engine.__init__에서
self.factors = factors.copy()
self.factors.set_index(['date', 'ticker'], inplace=True)
self.factors = self.factors.sort_index()  # 🔥 필수!
```

---

## 📊 백테스트 결과 요약

| 엔진 | Sharpe | 연간 수익률 | 연간 변동성 | 최대 낙폭 | 승률 | 거래 횟수 |
|------|--------|-------------|-------------|-----------|------|-----------|
| **ML9** | 0.80 | 14.84% | 18.63% | -28.37% | 49.17% | 785 |
| **QV** | 0.81 | 13.42% | 16.59% | -31.11% | 54.25% | 2,516 |

**특징**:
- 두 전략 모두 양호한 Sharpe Ratio (0.80-0.81)
- QV가 더 높은 승률(54.25%)과 낮은 변동성(16.59%)
- ML9는 더 적은 거래 횟수로 유사한 성과

---

## 🚧 알려진 이슈 및 제약사항

### 1. **데이터 품질**
- ✅ PIT 병합 완료로 look-ahead bias 제거
- ⚠️ `currentratio` 결측치 많음 (median 대체로 해결)
- ⚠️ 일부 ticker의 초기 fundamental 데이터 부족

### 2. **백테스트 기간**
- ML9: 2018-2024 (6년)
- QV: 2015-2024 (10년)
- ⚠️ ML9의 2015-2017 데이터 학습용으로만 사용

### 3. **거래 비용**
- ❌ 현재 백테스트에 거래 비용 미반영
- 📝 TODO: 8.5bps 거래 비용 추가 필요

### 4. **Guard 통합**
- ✅ MarketConditionGuard 구현 완료
- ❌ `run_all_tests.py`에 아직 통합 안 됨
- 📝 TODO: Guard 적용 버전 백테스트 필요

---

## 🎯 다음 단계 (새 세션 목표)

### Phase 1: Guard 통합 및 검증
1. **`run_all_tests.py`에 MarketConditionGuard 통합**
   - ML9 엔진에 Guard 레이어 추가
   - SPX 데이터 로딩 및 Guard 초기화
   - 리밸런싱 시 `get_ml9_scale()` 적용

2. **Guard 적용 전후 비교**
   - ML9 (Guard 없음) vs ML9 (Guard 적용)
   - Sharpe, MDD, 승률 변화 측정
   - 2023-2024 구간 집중 분석

### Phase 2: 앙상블 최적화
1. **Min-Max 앙상블 구현**
   - ML9 (Guard) + QV 가중치 조합
   - 각 윈도우 Sharpe의 최소값을 최대화
   - 그리드 서치: `w_ml9 ∈ [0, 1]`, `w_qv = 1 - w_ml9`

2. **전체 기간 (2015-2024) 최적화**
   - 3-window 또는 5-window 롤링 테스트
   - 각 윈도우의 min Sharpe 추적
   - 최종 목표: **전 구간 Sharpe 2.0+**

### Phase 3: 거래 비용 및 실전 검증
1. **거래 비용 추가**
   - 8.5bps 슬리피지 반영
   - 리밸런싱 빈도 최적화 (월 1회 vs 분기 1회)

2. **로버스트니스 테스트**
   - Label/Signal Shuffle Test 재실행
   - 파라미터 민감도 분석
   - Out-of-sample 검증

### Phase 4: Ares7 통합 설계
1. **리스크 관리 레이어**
   - 포지션 사이징 룰
   - 레버리지 제한
   - 손절/익절 룰

2. **실시간 운영 구조**
   - 데이터 업데이트 파이프라인
   - 리밸런싱 자동화
   - 모니터링 대시보드

---

## 🔑 핵심 API 키 (환경 변수)

```bash
# Polygon (가격 데이터)
POLYGON_API_KEY="your_polygon_api_key_here"

# Sharadar (펀더멘털 데이터)
SHARADAR_API_KEY="your_sharadar_api_key_here"

# 기타 (필요 시)
GEMINI_API_KEY="your_gemini_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```

**⚠️ 보안 주의**: 실제 API 키는 환경 변수로 관리하고 Git에 커밋하지 마세요!

---

## 📚 주요 참고 문서

### 이전 세션 문서 (`docs/session_history/`)
- `context_bridge.md`: 초기 컨텍스트 브리지
- `v2_1_final_complete_delivery.md`: QV v2.1 최종 버전
- `ml9_signal_shuffle_final_delivery.md`: ML9 로버스트니스 테스트
- `ml9_robustness_final_delivery.md`: ML9 기간별 성과 분석

### 코드 주석
- `run_all_tests.py`: 전체 백테스팅 파이프라인 (한글 주석 포함)
- `engines/ml_xgboost_v9_ranking.py`: ML9 원본 구현
- `engines/factor_quality_value_v2_1.py`: QV 원본 구현

---

## 🛠️ 실행 방법

### 1. 전체 백테스트 실행
```bash
cd /home/ubuntu/quant-ensemble-strategy
python3 run_all_tests.py
```

**출력**:
- `results/ml9_returns.csv`, `results/ml9_metrics.json`
- `results/qv_returns.csv`, `results/qv_metrics.json`
- `FINAL_REPORT.md`

### 2. 데이터 다운로드만
```python
from run_all_tests import download_and_prepare_data
data = download_and_prepare_data()
print(f"Loaded {len(data)} rows with {data['ticker'].nunique()} tickers")
```

### 3. 개별 엔진 테스트
```python
# ML9
from run_all_tests import ML9Engine
ml9_engine = ML9Engine(prices=prices, factors=data, top_quantile=0.2)
ml9_returns, ml9_metrics = ml9_engine.run_walk_forward_backtest(
    start_date="2018-01-01", end_date="2024-12-31"
)

# QV
from run_all_tests import QVEngine
qv_engine = QVEngine(top_quantile=0.3, use_inverse_vol=True)
qv_returns, qv_metrics = qv_engine.run_backtest(
    prices=prices, fund_daily=data,
    start_date="2015-01-01", end_date="2024-12-31"
)
```

---

## 💡 새 세션 시작 프롬프트

```markdown
안녕하세요! 이전 세션에서 작업하던 퀀트 전략 프로젝트를 이어서 진행하려고 합니다.

**프로젝트**: `quant-ensemble-strategy` (GitHub: yhun1542/ml9-quant-strategy)
**현재 상태**: PIT 데이터 병합 완료, ML9 + QV 백테스트 완료 (Sharpe 0.80-0.81)
**다음 목표**: MarketConditionGuard 통합 및 앙상블 최적화 (목표 Sharpe 2.0+)

모든 컨텍스트는 `/home/ubuntu/quant-ensemble-strategy/docs/NEW_SESSION_CONTEXT.md`에 정리되어 있습니다.

이 문서를 읽고 다음 작업을 진행해주세요:
1. Guard를 `run_all_tests.py`에 통합
2. Guard 적용 전후 비교 백테스트
3. ML9(Guard) + QV 앙상블 min-max 최적화

시작하겠습니다!
```

---

## 📝 변경 이력

- **2024-11-28**: 초기 작성 (PIT 통합 완료, ML9/QV 백테스트 완료)
- **다음 업데이트**: Guard 통합 후

---

**작성자**: Manus AI Agent  
**리포지토리**: https://github.com/yhun1542/ml9-quant-strategy  
**마지막 커밋**: `1796404` (docs: Add session history and analysis scripts)
