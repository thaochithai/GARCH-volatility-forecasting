# GARCH Family Models for Stock Volatility Forecasting

![Finance](https://img.shields.io/badge/Finance-Volatility%20Modeling-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Project Overview

This project implements and compares various GARCH (Generalized Autoregressive Conditional Heteroskedasticity) family models to forecast stock return volatility for tech companies. The analysis focuses on evaluating different volatility forecasting techniques and risk measures to determine which models provide the most accurate and reliable predictions across different market conditions.

## Table of Contents
- [Data Description](#data-description)
- [Model Implementation and Evaluation](#model-implementation-and-evaluation)
  - [GARCH Model Comparison](#garch-model-comparison)
  - [Moving Window Volatility Analysis](#moving-window-volatility-analysis)
  - [Value-at-Risk (VaR) Backtesting](#value-at-risk-var-backtesting)
  - [Expected Shortfall (ES) Backtesting](#expected-shortfall-es-backtesting)
  - [EWMA Model Evaluation](#ewma-model-evaluation)
- [Results and Findings](#results-and-findings)
- [Conclusions](#conclusions)

---

## Data Description

The analysis uses daily adjusted closing prices from various tech companies:

- **Microsoft (MSFT)**: Data from Jan 2012 to May 2020 (2,106 observations)
- **Apple (AAPL)**, **Facebook (FB)**, **Amazon (AMZN)**, **Tesla (TSLA)**: Jan 2012 to May 2020
- Additional tech stocks: **Dell (DELL)**, **IBM**, **Intel (INTC)**, **Roku (ROKU)**, **Disney (DIS)**, **Netflix (NFLX)**

### MSFT Return Statistics Summary

| Statistic | Value |
|-----------|-------|
| Minimum | -15.9453% |
| Maximum | 13.2929% |
| 1st Quartile | -0.62% |
| 3rd Quartile | 0.8480% |
| Median | 0.0689% |
| Mean | 0.1010% |
| Standard Deviation | 1.613678% |
| Skewness | -0.2493591 |
| Kurtosis | 12.86153 |

### Pre-modeling Tests

Before fitting the GARCH models, several statistical tests were performed to confirm the data characteristics:

#### Stationarity Test (Dickey-Fuller)

```
Dickey-Fuller = -13.725, Lag order = 12, p-value = 0.01
alternative hypothesis: stationary
```
- **Result**: p-value < 0.05 confirms the return series is stationary

#### Dependence Test (Ljung-Box)

```
Returns:            X-squared = 159.04, df = 10, p-value < 2.2e-16
Absolute Returns:   X-squared = 1025.1, df = 10, p-value < 2.2e-16
Squared Returns:    X-squared = 994.72, df = 10, p-value < 2.2e-16
```
- **Result**: p-values < 0.01 confirm the return series exhibits significant autocorrelation

#### Normality Tests

```
Shapiro test:    p-value = 4.560542e-37
Agostino test:   p-value = 4.026979e-06
Jarque test:     p-value = 0
```
- **Result**: All tests confirm non-normal distribution of returns

---

## Model Implementation and Evaluation

### GARCH Model Comparison

Four different GARCH-type models were implemented and compared:

- **ARCH(1)**: Basic Autoregressive Conditional Heteroskedasticity model
- **GARCH(1,1)**: Generalized ARCH incorporating volatility persistence
- **tGARCH(1,1)**: GARCH with Student's t-distribution for better fat-tail modeling
- **eGARCH(1,1)**: Exponential GARCH capturing asymmetric volatility response

#### Parameter Estimates and Significance

| Model | Parameter | Estimate | p-value |
|-------|-----------|----------|---------|
| **ARCH(1)** | ω | 0.001183 | 0.000001 |
|  | α₁ | 0.337150 | 0.000000 |
| **GARCH(1,1)** | ω | 0.001179 | 0.000000 |
|  | α₁ | 0.136873 | 0.000000 |
|  | β₁ | 0.819396 | 0.000000 |
| **tGARCH(1,1)** | ω | 0.000914 | 0.000040 |
|  | α₁ | 0.133520 | 0.000000 |
|  | β₁ | 0.853660 | 0.000000 |
| **eGARCH(1,1)** | ω | 0.000937 | 0.000045 |
|  | α₁ | -0.108446 | 0.000000 |
|  | β₁ | 0.953271 | 0.000000 |
|  | σ₁ | 0.215522 | 0.000000 |

All parameters are highly significant at the 1% level, confirming the presence of ARCH/GARCH effects in the Microsoft return series.

#### Model Comparison with Log-Likelihood and AIC

| Model | Log-Likelihood | AIC |
|-------|----------------|-----|
| ARCH(1) | 6089.575 | -5.7783 |
| GARCH(1,1) | 6146.779 | -5.8284 |
| tGARCH(1,1) | 6170.503 | -5.8533 |
| eGARCH(1,1) | 6165.739 | -5.8488 |

- **tGARCH(1,1)** has the lowest AIC, indicating it provides the best fit to the data when penalizing for model complexity
- **eGARCH(1,1)** is a close second, suggesting the asymmetric volatility response is important

#### Likelihood Ratio Tests

| Unrestricted Model | Restricted Model | LR Statistic | Critical Value (1%) | Conclusion |
|-------------------|-----------------|--------------|---------------------|------------|
| GARCH(1,1) | ARCH(1) | 114.408 | 6.63 | Reject H₀ |
| tGARCH(1,1) | ARCH(1) | 161.856 | 9.21 | Reject H₀ |
| eGARCH(1,1) | ARCH(1) | 152.326 | 6.63 | Reject H₀ |
| tGARCH(1,1) | GARCH(1,1) | 47.448 | 6.635 | Reject H₀ |
| eGARCH(1,1) | GARCH(1,1) | 37.918 | 6.635 | Reject H₀ |

The likelihood ratio tests consistently show that more complex models provide statistically significant improvements over simpler ones.

#### Volatility Forecasts

The models were used to forecast volatility for the next 10 days (May 18th to May 28th, 2020). Below are visualizations of the forecasts for each model:

![AR(1)-ARCH(1) Forecast](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/AR1_ARCH1_forecast.png)
*AR(1)-ARCH(1) Volatility Forecast*

![AR(1)-GARCH(1,1) Forecast](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/AR1_GARCH11_forecast.png)
*AR(1)-GARCH(1,1) Volatility Forecast*

![AR(1)-tGARCH(1,1) Forecast](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/AR1_tGARCH11_forecast.png)
*AR(1)-tGARCH(1,1) Volatility Forecast*

![AR(1)-eGARCH(1,1) Forecast](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/AR1_eGARCH11_forecast.png)
*AR(1)-eGARCH(1,1) Volatility Forecast*

All models generally predict decreasing volatility in the immediate future, followed by an increase. The eGARCH model shows a more pronounced decrease in consecutive days compared to other models.

#### Residual Analysis

The residuals from ARCH(1) and GARCH(1,1) models were analyzed to check model adequacy:

- Both models exhibit negative skewness (ARCH: -0.127, GARCH: -0.044)
- Anderson-Darling test p-value = 2.844e-07 for both models, rejecting normality
- Ljung-Box test indicated autocorrelation in ARCH(1) residuals (p-value = 4.794e-05)
- GARCH(1,1) residuals showed no significant autocorrelation (p-value = 0.01125)

---

### Moving Window Volatility Analysis

Three tech stocks (Facebook, Apple, Amazon) were used to analyze the sensitivity of window length on volatility and correlation estimates.

#### Volatility for Different Window Lengths

![10-Day Window Volatility](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/window10_volatility.png)
*10-Day Moving Window Volatility*

![100-Day Window Volatility](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/window100_volatility.png)
*100-Day Moving Window Volatility*

![1000-Day Window Volatility](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/window1000_volatility.png)
*1000-Day Moving Window Volatility*

#### Correlation for Different Window Lengths

![10-Day Window Correlation](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/window10_correlation.png)
*10-Day Moving Window Correlation*

![100-Day Window Correlation](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/window100_correlation.png)
*100-Day Moving Window Correlation*

![1000-Day Window Correlation](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/window1000_correlation.png)
*1000-Day Moving Window Correlation*

#### Window Length Sensitivity Analysis

- **10-day window**: Shows high volatility and frequent large fluctuations due to 10% data change with each new observation
- **100-day window**: More stable but still responsive to market changes (0.33% change per observation)
- **1000-day window**: Most stable but potentially less relevant to current market conditions

The tradeoff between responsiveness and stability highlights the importance of selecting an appropriate window length for different applications.

---

### Value-at-Risk (VaR) Backtesting

#### Historical Simulation (HS) vs. EWMA

A portfolio of Facebook, Apple, and Tesla stocks with equal weights was used to compare HS and EWMA methods for VaR estimation:

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|------------------|-----------------|----------------|
| EWMA | 21 | 62 | 2.964 | 0.0205082 |
| HS | 21 | 35 | 1.680 | 0.003370 |

![HS vs EWMA VaR Backtesting](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/HS_vs_EWMA_backtest.png)
*Historical Simulation vs. EWMA VaR Backtesting Results*

- HS method showed better performance with a VR closer to 1
- EWMA was more volatile and reactive to market changes
- Both methods exceeded the expected number of violations, but EWMA performed significantly worse

#### Normal GARCH, Student-t GARCH, and HS Comparison

Individual backtests were performed for Apple, Facebook, and Amazon stocks:

**Apple (AAPL) Results:**

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|------------------|-----------------|----------------|
| HS | 21 | 25 | 1.119 | 0.008002 |
| nGARCH | 21 | 35 | 1.679 | 0.015280 |
| tGARCH | 21 | 38 | 1.617 | 0.016381 |

![AAPL VaR Backtesting](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/AAPL_VaR_backtest.png)
*AAPL VaR Backtesting Results*

**Facebook (FB) Results:**

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|------------------|-----------------|----------------|
| HS | 21 | 26 | 1.190 | 0.008167 |
| nGARCH | 21 | 39 | 1.652 | 0.017898 |
| tGARCH | 21 | 33 | 1.785 | 0.026519 |

![FB VaR Backtesting](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/FB_VaR_backtest.png)
*FB VaR Backtesting Results*

**Amazon (AMZN) Results:**

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|------------------|-----------------|----------------|
| HS | 21 | 26 | 1.057 | 0.010348 |
| nGARCH | 21 | 39 | 1.119 | 0.014279 |
| tGARCH | 21 | 33 | 1.306 | 0.014476 |

![AMZN VaR Backtesting](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/AMZN_VaR_backtest.png)
*AMZN VaR Backtesting Results*

Across all three stocks, the HS method consistently outperformed GARCH models in terms of violation ratio, though it was less responsive to volatility changes.

---

### Expected Shortfall (ES) Backtesting

ES backtesting was performed for Dell, IBM, and Intel using GARCH(1,1) with Student-t distribution.

**Dell Technologies (DELL) Results:**

```
VaR Backtest Report:
Expected Exceed: 4.5
Actual VaR Exceed: 7
Actual %: 1.6%
Kupiec Test p-value: 0.264
Christoffersen Test p-value: 0.48

ES Backtest Report:
Expected Exceed: 4
Actual Exceed: 7
p-value: 0.0202
Decision: Reject H0 (Mean of Excess Violations > 0)
```

**IBM Results:**

```
VaR Backtest Report:
Expected Exceed: 16.1
Actual VaR Exceed: 27
Actual %: 1.7%
Kupiec Test p-value: 0.013
Christoffersen Test p-value: 0.035

ES Backtest Report:
Expected Exceed: 16
Actual Exceed: 27
p-value: 7.72e-09
Decision: Reject H0 (Mean of Excess Violations > 0)
```

**Intel (INTC) Results:**

```
VaR Backtest Report:
Expected Exceed: 16.1
Actual VaR Exceed: 16
Actual %: 1%
Kupiec Test p-value: 0.982
Christoffersen Test p-value: 0.851

ES Backtest Report:
Expected Exceed: 16
Actual Exceed: 16
p-value: 0.00018
Decision: Reject H0 (Mean of Excess Violations > 0)
```

For all three stocks, the ES backtests indicated that the GARCH models underestimated the tail risk, even when VaR violations were at acceptable levels (as with Intel).

---

### EWMA Model Evaluation

A portfolio of Netflix, Roku, and Disney was created to backtest the EWMA model using various methods:

#### Violation Ratio and VaR Volatility

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|------------------|-----------------|----------------|
| EWMA | 21 | 53 | 2.516 | 0.018808 |

![EWMA Backtest](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/EWMA_backtest.png)
*EWMA Backtest Results*

#### Statistical Tests for Netflix (NFLX)

Unconditional coverage test and independence test were applied to evaluate the EWMA model for Netflix stock:

```
Violation Ratio: 1.827 (38 actual violations vs. 21 expected)
Unconditional Coverage Test:
  - Test Statistic: 57.663
  - Critical Value: 3.841
  - p-value: 3.11e-14
  - Decision: Reject H0 (Correct Exceedances)

Independence Test:
  - Test Statistic: 59.376
  - Critical Value: 5.991
  - p-value: 1.28e-13
  - Decision: Reject H0 (Independent Violations)
```

Based on the rule of thumb where VR ∈ [0.3, 0.5] or VR ∈ [1.5, 2] indicates a bad model, and VR > 2 indicates a useless model, the EWMA approach with VR = 2.516 for the portfolio and VR = 1.827 for Netflix is classified as a bad to useless model for these tech stocks.

![Netflix EWMA Backtest](https://raw.githubusercontent.com/username/stock-volatility-garch/main/results/figures/NFLX_EWMA_backtest.png)
*Netflix EWMA Backtest Results*

---

## Results and Findings

1. **GARCH Model Comparison**:
   - tGARCH(1,1) and eGARCH(1,1) consistently outperformed simpler models
   - Likelihood ratio tests confirmed that more complex models provided statistically significant improvements
   - All models forecast decreasing volatility in the short term followed by an increase

2. **Moving Window Analysis**:
   - Window length significantly impacts volatility and correlation estimates
   - Shorter windows (10 days) capture recent dynamics but exhibit high variability
   - Longer windows (1000 days) provide stability but may be slow to react to market changes
   - Medium windows (100 days) offer a balance between responsiveness and stability

3. **VaR Backtesting**:
   - Historical Simulation consistently outperformed EWMA and GARCH models across different stocks
   - EWMA was particularly poor with violation ratios often exceeding 2
   - GARCH models were more responsive to market changes but had higher violation ratios

4. **ES Backtesting**:
   - All tested GARCH models underestimated tail risk
   - Even when VaR estimates were acceptable (e.g., Intel), ES backtests indicated problems
   - This highlights the importance of looking beyond VaR to assess tail risk properly

5. **EWMA Evaluation**:
   - EWMA performed poorly for tech stocks with violation ratios consistently above 1.5
   - Statistical tests strongly rejected both correct exceedances and independence hypotheses
   - High VaR volatility indicates EWMA is very reactive to market changes but not in a way that produces reliable risk estimates

## Conclusions

1. **Model Selection**: For volatility forecasting of tech stocks, tGARCH and eGARCH models provide superior performance compared to simpler models, capturing both persistence and asymmetry in volatility.

2. **Risk Measurement**: Historical Simulation consistently outperforms parametric methods (EWMA, GARCH) for VaR estimation in tech stocks, suggesting that empirical distributions better capture the true risk characteristics than parameterized models.

3. **Window Sensitivity**: The choice of window length involves a crucial tradeoff between responsiveness and stability. Different applications and market conditions may require different window lengths.

4. **Tail Risk Assessment**: VaR alone is insufficient for comprehensive risk management. ES backtesting reveals that models with acceptable VaR performance may still severely underestimate tail risk.

5. **Practical Applications**: For risk management in tech stocks, a combination of Historical Simulation for VaR estimation and advanced GARCH models (tGARCH, eGARCH) for volatility forecasting provides the most reliable results.

These findings contribute to the understanding of volatility dynamics in tech stocks and provide practical guidance for risk managers and quantitative analysts working with these securities.
