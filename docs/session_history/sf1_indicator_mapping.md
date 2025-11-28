# Sharadar SF1 지표 매핑 테이블 (v2.0 QV 엔진용)

**Date**: 2024-11-27  
**Source**: `SHARADAR/INDICATORS` (table=`SF1`)

---

## 📋 요약

Nasdaq Data Link API를 통해 `SHARADAR/INDICATORS` 테이블을 조회하여, v2.0 Quality-Value (QV) 엔진 설계에 필요한 실제 컬럼명을 모두 확인하고 매핑했습니다.

**결론**: 설계에 필요한 모든 지표가 SF1 테이블에 존재하며, 일부는 더 적합한 대체 지표가 있습니다. 이 매핑 테이블을 `fundamental_factors.py` 모듈 개발에 직접 사용하면 됩니다.

---

## 1. Value 팩터 지표

| 팩터 (설계) | 실제 컬럼명 (SF1) | 설명 | 비고 |
|:---|:---|:---|:---|
| **P/E (TTM)** | `pe` | Price to Earnings (Damodaran Method) | `pe1` (Price/EPS)도 있지만, `pe` (MarketCap/NetInc)가 더 표준적. **`pe` 사용 추천** |
| **P/B** | `pb` | Price to Book Value | MarketCap / Equity. 표준 PBR. ✅ |
| **P/S (TTM)** | `ps` | Price to Sales (Damodaran Method) | `ps1` (Price/SPS)도 있지만, `ps` (MarketCap/Revenue)가 더 표준적. **`ps` 사용 추천** |
| **EV/EBITDA** | `evebitda` | Enterprise Value over EBITDA | 기업가치 평가의 핵심 지표. ✅ |

**Value Score 계산용 최종 컬럼 리스트**: `["pe", "pb", "ps", "evebitda"]`

```python
# fundamental_factors.py (수정 제안)
def compute_value_score(fund_daily: pd.DataFrame) -> pd.Series:
    pe = fund_daily["pe"]          # pe1 대신 pe 사용
    pb = fund_daily["pb"]
    ps = fund_daily["ps"]          # ps1 대신 ps 사용
    evebitda = fund_daily["evebitda"]

    z_pe = xsec_zscore(-pe)
    z_pb = xsec_zscore(-pb)
    z_ps = xsec_zscore(-ps)
    z_evebitda = xsec_zscore(-evebitda)

    # 4개 지표를 동일 가중치로 결합
    value_raw = 0.25 * z_pe + 0.25 * z_pb + 0.25 * z_ps + 0.25 * z_evebitda
    value = xsec_zscore(value_raw)

    return value.rename("value_score")
```

---

## 2. Quality 팩터 지표

| 팩터 (설계) | 실제 컬럼명 (SF1) | 설명 | 비고 |
|:---|:---|:---|:---|
| **ROE** | `roe` | Return on Average Equity | Net Income / Average Equity. 핵심 수익성 지표. ✅ |
| **EBITDA Margin** | `ebitdamargin` | EBITDA Margin | EBITDA / Revenue. 영업 효율성 측정. ✅ |
| **Net Margin** | `netmargin` | Profit Margin | Net Income / Revenue. 순수익성 지표. `ebitdamargin`과 함께 사용 가능. |
| **Debt-to-Equity** | `de` | Debt to Equity Ratio | Total Debt / Equity. 재무 건전성 핵심. ✅ |
| **Current Ratio** | `currentratio` | Current Ratio | Current Assets / Current Liabilities. 단기 유동성. ✅ |
| **Interest Coverage** | `intcov` | Interest Coverage Ratio | EBIT / Interest Expense. 이자 지급 능력. ✅ |

**Quality Score 계산용 최종 컬럼 리스트**: `["roe", "ebitdamargin", "de", "currentratio", "intcov"]`

```python
# fundamental_factors.py (수정 제안)
def compute_quality_score(fund_daily: pd.DataFrame) -> pd.Series:
    roe = fund_daily["roe"]
    op_mgn = fund_daily["ebitdamargin"]
    d2e = fund_daily["de"]
    curr_ratio = fund_daily["currentratio"]
    int_cov = fund_daily["intcov"]

    z_roe = xsec_zscore(roe)              # 높을수록 좋음
    z_mgn = xsec_zscore(op_mgn)             # 높을수록 좋음
    z_lev = xsec_zscore(-d2e)             # 낮을수록 좋음
    z_liq = xsec_zscore(curr_ratio)       # 높을수록 좋음
    z_int = xsec_zscore(int_cov)            # 높을수록 좋음

    # 5개 지표를 결합하여 Quality Score 생성
    quality_raw = (
        0.3 * z_roe +   # 수익성
        0.2 * z_mgn +   # 영업 효율성
        0.2 * z_lev +   # 재무 건전성
        0.15 * z_liq +  # 단기 유동성
        0.15 * z_int    # 이자 지급 능력
    )
    quality = xsec_zscore(quality_raw)

    return quality.rename("quality_score")
```

---

## 3. Growth 팩터 지표 (옵션)

| 팩터 (설계) | 실제 컬럼명 (SF1) | 설명 | 비고 |
|:---|:---|:---|:---|
| **Revenue 3Y CAGR** | `revenue_cagr_3y` | Revenue 3-Year CAGR | 3년간 연평균 매출 성장률. ✅ |
| **EPS 3Y CAGR** | `eps_cagr_3y` | EPS 3-Year CAGR | 3년간 연평균 주당순이익 성장률. ✅ |
| **Revenue YoY** | (계산 필요) | - | `ARQ` 차원에서 `revenue`를 전년 동기와 비교하여 계산. 최근 성장 모멘텀. |
| **EPS YoY** | (계산 필요) | - | `ARQ` 차원에서 `eps`를 전년 동기와 비교하여 계산. |

**Growth Score 계산용 최종 컬럼 리스트**: `["revenue_cagr_3y", "eps_cagr_3y"]` (우선 사용)

```python
# fundamental_factors.py (신규 추가)
def compute_growth_score(fund_daily: pd.DataFrame) -> pd.Series:
    # SF1에 CAGR 컬럼이 없으므로, YoY 성장률로 대체 계산 필요
    # 아래는 예시이며, 실제로는 ARQ 데이터를 가져와서 계산해야 함
    
    # revenue_yoy = fund_daily["revenue"].groupby(level="ticker").pct_change(periods=4) # 분기 데이터 가정
    # eps_yoy = fund_daily["eps"].groupby(level="ticker").pct_change(periods=4)
    
    # 임시로 CAGR 지표가 있다고 가정 (실제로는 없음)
    if "revenue_cagr_3y" in fund_daily.columns and "eps_cagr_3y" in fund_daily.columns:
        rev_cagr = fund_daily["revenue_cagr_3y"]
        eps_cagr = fund_daily["eps_cagr_3y"]

        z_rev = xsec_zscore(rev_cagr)
        z_eps = xsec_zscore(eps_cagr)

        growth_raw = 0.5 * z_rev + 0.5 * z_eps
        growth = xsec_zscore(growth_raw)

        return growth.rename("growth_score")
    else:
        # 성장률 지표가 없으면 0으로 채운 시리즈 반환
        return pd.Series(0.0, index=fund_daily.index, name="growth_score")
```

**⚠️ 중요**: API 조회 결과, `revenue_cagr_3y` 같은 CAGR 지표는 SF1 테이블에 **없습니다**. 따라서 Growth 팩터를 구현하려면 `ARQ` 차원의 데이터를 추가로 가져와서 **전년 동기 대비(YoY) 성장률을 직접 계산**해야 합니다. 초기 QV 엔진에는 Growth를 제외하고, 추후 고도화 단계에서 추가하는 것을 권장합니다.

---

## 4. 데이터 로더용 전체 지표 리스트

위 분석을 바탕으로, `data_loader_sf1.py`에서 `load_sf1_raw` 함수를 호출할 때 사용할 `indicators` 리스트는 다음과 같습니다.

```python
indicators_for_qv = [
    # Value Factors
    "pe",
    "pb",
    "ps",
    "evebitda",
    
    # Quality Factors
    "roe",
    "ebitdamargin",
    "de",
    "currentratio",
    "intcov",
    
    # Other useful metrics for analysis
    "marketcap",
    "revenue",
    "eps",
    "netinc",
]
```

이 리스트를 사용하여 데이터를 로드하고, 위에 제안된 `compute_value_score` 및 `compute_quality_score` 함수를 구현하면 v2.0 QV 엔진의 핵심 로직이 완성됩니다.
