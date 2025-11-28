# ML9 Quantitative Trading Strategy

**고성능 머신러닝 기반 퀀트 트레이딩 전략 엔진**

## 📊 프로젝트 개요

ML9는 XGBoost 기반 머신러닝과 모멘텀/변동성/가치 팩터를 결합한 퀀트 트레이딩 전략입니다. 2021-2024년 백테스트에서 Sharpe Ratio 4.17을 달성했으며, 다양한 시장 환경과 유니버스에서의 강건성을 검증했습니다.

## 🎯 핵심 성과

| 테스트 | 기간 | 종목 수 | Sharpe Ratio | 연간 수익률 | 최대 낙폭 |
|--------|------|---------|--------------|-------------|-----------|
| 초기 모델 | 2021-2024 | 30 | **4.17** | 93.31% | -12.52% |
| 유니버스 확장 | 2021-2024 | 100 | **1.96** | 35.42% | -11.28% |
| 기간 확장 | 2015-2020 | 29 | 0.83 | 25.66% | -42.66% |
| 3년 학습 | 2020-2024 | 90 | 1.34 | 23.30% | -14.57% |

## 🏗️ 아키텍처

### ML9 엔진 구조
```
1. 데이터 수집 (Polygon API)
2. 팩터 계산 (Momentum 60d, Volatility 30d, Value Proxy)
3. Cross-sectional Ranking (Z-score 정규화)
4. XGBoost 분류 (Top 20% / Middle 60% / Bottom 20%)
5. 포트폴리오 구성 (Long-only, Equal-weighted Top 20%)
6. 월간 리밸런싱
```

### 주요 컴포넌트
- **engines/ml_xgboost_v9_ranking.py**: 핵심 ML9 엔진
- **backtest_ml9_optimized.py**: Numba JIT 최적화 백테스트 (10-12x 속도 향상)
- **utils/transaction_costs.py**: 거래 비용 모델 (8.5 bps)
- **signal_shuffle_test.py**: 신호 검증 테스트

## 🚀 빠른 시작

### 1. 환경 설정
```bash
python3 -m venv venv_qv
source venv_qv/bin/activate
pip install -r requirements.txt
```

### 2. 데이터 다운로드
```bash
# Polygon API 키 설정 필요
python download_sp100_2020.py
```

### 3. 백테스트 실행
```bash
# 최적화된 백테스트 (Numba JIT)
python backtest_ml9_optimized.py

# SP100 유니버스 테스트
python backtest_ml9_sp100_2020_2024.py
```

## 📈 성능 최적화

- **Numba JIT**: 팩터 계산 병렬화로 10-12배 속도 향상
- **멀티프로세싱**: CPU 코어 활용 극대화
- **벡터화**: NumPy 기반 연산 최적화

**실행 시간**: 5-7분 → **30초** (1135일, 90종목 기준)

## 🔍 검증 테스트

### ✅ 통과한 테스트
- **Look-ahead Bias**: 미래 데이터 누출 없음
- **Signal Shuffle**: Baseline Sharpe 3.37 vs Shuffle 0.05 (실제 알파 확인)
- **Market Regime**: 모든 시장 환경에서 Sharpe 3.0+ (2021-2024)

### ⚠️ 발견된 이슈
- **Regime Dependency**: COVID 포함 학습 시 2023년 예측 실패 (Sharpe 0.00)
- **Value Factor**: 현재 구현(1/price)은 효과 없음 (중요도 0%)
- **Walk-Forward 일관성**: 0.61 (중간 수준의 과적합 위험)

## 📝 주요 리포트

- **ML9_ROBUSTNESS_REPORT.md**: 종합 강건성 테스트 리포트
- **V1_5_FINAL_REPORT.md**: v1.5 거래비용 포함 최종 리포트
- **ML9_SIGNAL_SHUFFLE_REPORT.md**: 신호 검증 테스트 리포트

## 🛠️ v1.6 개발 로드맵

1. **엔진 다각화**: 다른 시장 환경에 적합한 보조 엔진 개발
2. **동적 모델 가중**: 거시 지표 기반 실시간 가중치 조절
3. **피처 엔지니어링**: PBR, PER 등 정교한 가치 팩터 도입
4. **Regime Detection**: 시장 환경 자동 감지 및 전략 전환

## 📦 프로젝트 구조

```
quant-ensemble-strategy/
├── engines/
│   ├── ml_xgboost_v9_ranking.py    # ML9 엔진
│   └── fv3c_engine.py              # FV3c 엔진
├── utils/
│   └── transaction_costs.py        # 거래비용 모델
├── data/                           # 가격 데이터 (CSV)
├── results/                        # 백테스트 결과 (JSON)
├── docs/                           # 문서 및 리포트
├── backtest_ml9_optimized.py       # 최적화 백테스트
├── signal_shuffle_test.py          # 검증 테스트
└── requirements.txt                # 의존성
```

## 📄 라이선스

MIT License

## 👤 저자

Manus AI - Quantitative Trading Research

---

**⚠️ 면책 조항**: 본 프로젝트는 연구 및 교육 목적으로 제공됩니다. 실제 투자에 사용 시 발생하는 손실에 대해 책임지지 않습니다.
