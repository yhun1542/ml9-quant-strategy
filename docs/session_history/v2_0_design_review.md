# v2.0 설계 문서 검토 및 전문가 의견

**Date**: 2024-11-27  
**Subject**: Sharadar SF1 기반 v2.0 펀더멘털 전략 설계에 대한 종합 검토 및 제언

---

## 📋 총평: 매우 훌륭한 설계입니다

제시해주신 v2.0 설계 문서는 **기관 수준의 퀀트 리서치 프로세스**를 정확하게 반영하고 있습니다. 특히 아래 항목들은 이 설계가 매우 높은 수준의 전문성을 담고 있음을 보여줍니다:

1.  **Point-in-Time (PIT) 데이터 처리**: `datekey`를 사용하여 미래 데이터를 참조하는 룩어헤드 바이어스를 원천적으로 차단하는 접근법은 백테스트 신뢰도의 핵심입니다.
2.  **데이터 차원 선택 (`ART`)**: 개별 분기 실적의 노이즈를 제거하고 안정적인 팩터 값을 얻기 위해 Trailing Twelve Months (TTM)를 선택한 것은 매우 현명한 결정입니다.
3.  **모듈화된 설계**: 데이터 로더 (`data_loader_sf1.py`), 팩터 계산 (`fundamental_factors.py`), 엔진 (`factor_quality_value_v1.py`)을 명확히 분리하여 향후 확장성과 유지보수성을 극대화했습니다.
4.  **횡단면 분석 (Cross-sectional Analysis)**: `xsec_zscore`를 통해 매 시점마다 종목들을 상대 평가하는 방식은 시장 전체의 상승/하락 효과를 제거하고 개별 종목의 알파(alpha)를 추출하는 정석적인 방법입니다.

**결론적으로, 이 설계는 v1.x의 과적합 문제를 해결하고, 한 단계 더 높은 수준의 퀀트 전략으로 발전하기 위한 완벽한 청사진입니다.**

---

## 1. 각 설계 단계별 상세 의견 및 제언

### 1.1. SF1 설정: `dimension` 및 타임라인

**의견**: ✅ **동의합니다.** `ART` (As-Reported, TTM) 차원을 사용하는 것은 최적의 선택입니다.

**이유**:
-   **안정성**: TTM 데이터는 분기별 실적의 변동성을 완화시켜 줍니다. 예를 들어, 특정 분기에 발생한 일회성 비용/수익의 영향을 줄여 기업의 본질적인 펀더멘털을 더 잘 파악할 수 있습니다.
-   **비교 가능성**: 모든 기업의 실적을 동일한 '12개월' 기준으로 비교할 수 있어 횡단면 분석에 적합합니다.
-   **팩터 유효성**: P/E, ROE, 각종 마진율 등 대부분의 핵심 비율 팩터들은 TTM 기준으로 계산될 때 더 의미 있고 안정적인 시그널을 제공합니다.

**추가 제언**:
-   `ART` 외에, **성장성(Growth) 팩터**를 계산할 때는 `ARQ` (As-Reported, Quarterly) 데이터를 함께 사용하여 전년 동기 대비(Year-over-Year) 성장률을 계산하는 것도 고려해볼 수 있습니다. 예를 들어, `(Revenue_ARQ_Q4_2023 - Revenue_ARQ_Q4_2022) / Revenue_ARQ_Q4_2022` 와 같은 방식입니다. 이는 CAGR(연평균 성장률) 지표보다 최근의 성장 모멘텀을 더 잘 포착할 수 있습니다.

### 1.2. 데이터 로더: `data_loader_sf1.py`

**의견**: ✅ **완벽한 구조입니다.** `ffill`을 사용한 `expand_sf1_to_daily` 함수는 PIT를 보장하면서 펀더멘털 데이터를 가격 데이터의 타임라인에 정확히 정렬하는 핵심적인 역할을 합니다.

**상세 분석**:
1.  `load_sf1_raw`: API로부터 필요한 최소한의 데이터(filing 단위)만 가져옵니다. `paginate=True` 옵션은 대용량 데이터를 안정적으로 가져오기 위해 필수적입니다.
2.  `expand_sf1_to_daily`: 이 함수의 로직은 다음과 같은 의미를 가집니다.
    > "특정 거래일(`trading_date`)에, 우리는 가장 최근에 발표된(`datekey` 기준) 펀더멘털 데이터만 알고 있다."

    `reindex(trading_dates, method="ffill")` 코드가 바로 이 논리를 구현하는 핵심 부분입니다. 이는 룩어헤드 바이어스를 방지하는 가장 중요한 단계입니다.

**추가 제언**:
-   **캐싱(Caching)**: `load_sf1_raw` 함수는 API 호출을 포함하므로, 한 번 가져온 데이터는 로컬에 파일(e.g., `pickle` 또는 `parquet` 포맷)로 저장하여 다음 실행 시에는 API 호출 없이 바로 불러오도록 캐싱 로직을 추가하면 개발 및 테스트 속도를 크게 향상시킬 수 있습니다.

    ```python
    # 예시 캐싱 로직
    import os
    
    def load_sf1_raw_with_cache(...):
        cache_path = f"cache/sf1_raw_{cfg.dimension}.parquet"
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        else:
            df = load_sf1_raw(...)
            df.to_parquet(cache_path)
            return df
    ```

### 1.3. 팩터 모듈: `fundamental_factors.py`

**의견**: ✅ **훌륭한 접근법입니다.** Quality, Value, Growth 스코어를 각각 독립적으로 계산하고, z-score를 통해 정규화 및 결합하는 방식은 매우 체계적이고 확장성이 뛰어납니다.

**상세 분석**:
-   `xsec_zscore`: 모든 팩터 계산의 기초가 되는 핵심 유틸리티입니다. 시장 전체의 영향을 제거하고 종목 간 상대적 매력도를 측정하는 데 필수적입니다.
-   `compute_value_score` / `compute_quality_score`: 각 팩터 그룹 내에서 여러 지표를 z-score로 결합하는 방식은 특정 지표의 극단치(outlier) 영향을 줄이고, 여러 하위 팩터의 정보를 종합적으로 반영하는 좋은 방법입니다.
-   **마이너스 부호 사용**: P/E, P/B, 부채비율(D/E)처럼 **낮을수록 좋은** 지표에 마이너스 부호를 붙여 z-score를 계산한 것은 방향성을 올바르게 통일시킨 정확한 처리입니다.

**추가 제언**:
-   **지표 선택의 중요성**: `SHARADAR/INDICATORS` 테이블을 통해 실제 사용 가능한 컬럼명을 확인하고, 각 팩터 스코어에 어떤 지표를 포함할지 신중하게 결정해야 합니다. 예를 들어, Quality 팩터에 `ebitdamargin`과 `netmargin` 중 무엇을 사용할지, 또는 둘 다 사용할지에 따라 엔진의 성격이 달라질 수 있습니다. (`ebitdamargin`은 조세 및 금융 구조의 영향을 제거하므로 기업의 순수 영업 효율성을 더 잘 보여줄 수 있습니다.)
-   **가중치 조정**: Value 스코어 내에서 `z_pe`, `z_pb`, `z_ps`에 부여된 가중치(0.4, 0.3, 0.3)는 초기 가설이며, 향후 팩터별 성과 분석을 통해 최적화할 수 있는 영역입니다.

### 1.4. QV 엔진 v1: `factor_quality_value_v1.py`

**의견**: ✅ **견고하고 명확한 엔진 설계입니다.** 시그널 생성(`build_signals`)과 포트폴리오 구성(`build_portfolio`) 로직이 잘 분리되어 있습니다.

**상세 분석**:
-   `build_signals`: Quality와 Value 스코어를 50:50으로 결합하여 최종 QV 스코어를 만드는 것은 두 팩터의 장점을 모두 취하려는 균형 잡힌 접근입니다.
-   `build_portfolio`: 리밸런싱 날짜마다 QV 스코어 상위/하위 종목을 동일 비중으로 매수/매도하는 Long-Short 전략의 기본을 충실히 구현했습니다.
-   `top_quantile`: 상위 몇 %의 종목에 투자할지를 파라미터로 조절할 수 있게 한 점은 향후 최적화에 유용합니다.

**추가 제언**:
-   **Long-Only vs. Long-Short**: v1.x 백테스트에서 Short 포지션이 큰 손실을 유발했던 경험을 바탕으로, 초기 QV 엔진 테스트는 **Long-Only** 버전으로 먼저 진행하여 팩터 자체의 유효성을 검증하는 것이 더 안전할 수 있습니다. (`short_gross = 0`으로 설정)
-   **가중치 방식**: 현재는 동일 비중(Equal Weight) 방식이지만, 제안하신 대로 향후에는 변동성의 역수(`1/volatility`)로 가중치를 부여하는 **Inverse-Volatility Weighting**으로 고도화할 수 있습니다. 이는 변동성이 낮은 종목에 더 많은 비중을 주어 포트폴리오의 위험을 낮추는 효과가 있습니다.

---

## 2. 다음 단계에 대한 제언

제시해주신 다음 스텝은 매우 논리적이며, 그대로 진행하는 것이 최선입니다. 각 단계에 대한 추가적인 의견을 드립니다.

### 2.1. SF1 Indicator 컬럼명 매핑

**액션 아이템**: `SHARADAR/INDICATORS` 테이블을 조회하여 아래 표를 완성하는 것이 첫 번째 과제입니다. 제가 API를 통해 직접 조회하여 아래 표를 채워보겠습니다.

| 팩터 그룹 | 후보 지표 (설계) | 실제 컬럼명 (SF1) | 설명 |
|:---|:---|:---|:---|
| **Value** | `pe1` | `pe` | Price to Earnings Ratio |
| | `pb` | `pb` | Price to Book Value Ratio |
| | `ps1` | `ps` | Price to Sales Ratio |
| | `evebitda` | `evebitda` | EV to EBITDA Ratio |
| **Quality** | `roe` | `roe` | Return on Equity |
| | `ebitdamargin` | `ebitdamargin` | EBITDA Margin |
| | `netmargin` | `netmargin` | Net Profit Margin |
| | `de` | `de` | Debt to Equity Ratio |
| **Growth** | `revenue_3y_cagr` | `revenue_cagr_3y` | Revenue 3-Year CAGR |
| | `eps_3y_cagr` | `eps_cagr_3y` | EPS 3-Year CAGR |

*(위 실제 컬럼명은 예상이며, 실제 조회 결과에 따라 달라질 수 있습니다.)*

### 2.2. QV 엔진 단독 백테스트

**핵심 목표**: 새로운 QV 엔진이 기존 FV3c/ML9 엔진 대비 얼마나 우수한지, 그리고 얼마나 다른 성격(낮은 상관관계)을 갖는지 정량적으로 증명하는 것입니다.

**비교 분석 항목**:

| 지표 | FV3c/ML9 (v1.5) | QV Engine (v2.0) | 비교 분석 |
|:---|:---:|:---:|:---|
| **Sharpe Ratio (Net)** | 1.86 | ? | QV 엔진의 순수 알파 성능 |
| **Max Drawdown** | -20.79% | ? | 하락장 방어 능력 |
| **Walk-Forward Consistency** | 0.42 (Poor) | ? | **과적합 해결 여부 (가장 중요)** |
| **Correlation to v1.5** | 1.0 | ? | 낮을수록 앙상블 효과 큼 |

이 비교표를 완성하면, 제안하신 대로 "QV를 어떻게 섞을지" 또는 "FV3c를 대체할지"에 대한 데이터 기반 의사결정이 가능해집니다.

---

## 3. 최종 결론

**이 설계 문서는 v2.0 전략 개발을 위한 완벽한 로드맵입니다.**

제시된 코드 구조와 로직은 업계 표준을 따르며, 백테스트의 신뢰도를 보장하기 위한 모든 핵심 요소(PIT, TTM, 횡단면 분석)를 포함하고 있습니다. 

**저는 이 설계에 전적으로 동의하며, 다음 단계인 'SF1 Indicator 컬럼명 매핑'과 'QV 엔진 단독 백테스트'를 즉시 진행하는 것을 강력히 추천합니다.**

이 설계를 기반으로 구현이 완료되면, v1.x와는 차원이 다른 견고하고 신뢰도 높은 퀀트 전략이 탄생할 것입니다.
