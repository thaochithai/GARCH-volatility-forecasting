# Financial Return Series Volatility Forecasting using GARCH-family models
![Finance](https://img.shields.io/badge/Finance-Volatility%20Modeling-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

This repository contains analysis and implementation of various volatility forecasting models for financial time series. The project explores different GARCH-type models, moving window volatility, correlation estimations, and various backtesting techniques for Value at Risk (VaR) and Expected Shortfall (ES).

## Techniques & Key Highlights

### Methods Implemented
- **GARCH-type Models**: ARCH(1), GARCH(1,1), tGARCH(1,1), eGARCH(1,1)
- **Moving Window Analysis**: Applied to volatility and correlation across multiple window lengths (10, 100, 1000)
- **VaR and ES Estimation**: Historical Simulation (HS), Exponentially Weighted Moving Average (EWMA)
- **Backtesting Methods**: Violation ratios, VaR volatility, Kupiec's POF test, Christoffersen independence test
- **Statistical Tests**: Unit root tests, dependence tests, normality tests, parameter significance, likelihood ratio tests

### Key Findings
- **Model Selection**: eGARCH and tGARCH consistently outperformed simpler models (ARCH, standard GARCH) based on AIC and likelihood ratio tests
- **Window Size Sensitivity**: Smaller window sizes (10) produced highly volatile estimates while larger windows (1000) missed recent market trends
- **VaR Performance**: Historical Simulation provided more stable but slower-adapting VaR estimates than EWMA or GARCH models
- **Model Adequacy**: For most tech stocks analyzed, even sophisticated models showed violation ratios exceeding expected levels, especially during high volatility periods
- **Risk Underestimation**: Expected Shortfall backtests revealed that many models systematically understate tail risk even when VaR appears adequate
- **EWMA Assessment**: EWMA models for streaming service stocks exhibited poor performance with violation ratios >2, indicating unreliable risk estimates

## Table of Contents
1. [GARCH Models Analysis](#1-garch-models-analysis)
2. [Moving Window Volatility and Correlations](#2-moving-window-volatility-and-correlations)
3. [Backtesting Historical Simulation and EWMA](#3-backtesting-historical-simulation-and-ewma)
4. [Backtesting HS and GARCH](#4-backtesting-hs-and-garch)
5. [Expected Shortfall Backtesting](#5-expected-shortfall-backtesting)
6. [EWMA Backtesting with Multiple Methods](#6-ewma-backtesting-with-multiple-methods)
7. [Advanced Backtests for EWMA](#7-advanced-backtests-for-ewma)

## 1. GARCH Models Analysis

### Data Summary
The analysis uses daily observations of Microsoft Corporation (MSFT) stock prices between January 1, 2012, and May 18, 2020 (2,106 observations). All data are adjusted prices to ensure reliable testing results.

### Summary Statistics of Microsoft Stock Returns
| Statistic | Value |
|-----------|-------|
| Min | -15.9453% |
| Max | 13.2929% |
| 1st Quartile | -0.62% |
| 3rd Quartile | 0.8480% |
| Median | 0.0689% |
| Mean | 0.1010% |
| Standard Deviation | 1.613678% |
| Skewness | -0.2493591 |
| Kurtosis | 12.86153 |

### Stationary, Dependence, and Normality Tests
- **Dickey-Fuller Test**: p-value = 0.01, rejecting the null hypothesis of non-stationarity at 1% significance level.
- **Ljung-Box Test**: All test results show p-values < 0.01, rejecting the independence hypothesis at 1% significance level.
- **Normality Tests**: Shapiro, Agostino, and Jarque tests all reject normality at 1% significance level.

### Model Estimation
Four models were compared: ARCH(1), GARCH(1,1), tGARCH(1,1), and eGARCH(1,1) with Student-t distribution.

#### Parameter Significance Test Results
| Model | Parameters | Estimates | p-values |
|-------|------------|-----------|----------|
| ARCH(1) | ω | 0.001183 | 0.000001 |
|        | α₁ | 0.337150 | 0.000000 |
| GARCH(1,1) | ω | 0.001179 | 0.000000 |
|            | α₁ | 0.136873 | 0.000000 |
|            | β₁ | 0.819396 | 0.000000 |
| tGARCH(1,1) | ω | 0.000914 | 0.000040 |
|             | α₁ | 0.133520 | 0.000000 |
|             | β₁ | 0.853660 | 0.000000 |
| eGARCH(1,1) | ω | 0.000937 | 0.000045 |
|             | α₁ | -0.108446 | 0.000000 |
|             | β₁ | 0.953271 | 0.000000 |
|             | σ₁ | 0.215522 | 0.000000 |

All parameters are highly significant at 1% level. According to AIC, eGARCH is the best fit model (-5.8488), followed by tGARCH (-5.8533), with ARCH(1) performing worst (-5.7783).

#### Likelihood Ratio Tests
| Unrestricted Model | Restricted Model | LR | Restrictions | Critical Value (1%) |
|-------------------|-----------------|-----|--------------|---------------------|
| GARCH(1,1) | ARCH(1) | 114.4082 | 1 | 6.63 (0.000) |
| tGARCH(1,1) | ARCH(1) | 161.8561 | 2 | 9.21 (0.000) |
| eGARCH(1,1) | ARCH(1) | 152.3263 | 1 | 6.63 (0.000) |
| tGARCH(1,1) | GARCH(1,1) | 47.44797 | 1 | 6.635 (0.000) |
| eGARCH(1,1) | GARCH(1,1) | 37.91812 | 1 | 6.635 (0.000) |

All tests suggest larger models (GARCH, tGARCH, eGARCH) significantly outperform smaller ones (ARCH).

### Volatility Forecasts
The models were used to forecast returns and volatility for 10 days (May 18-28, 2020). Most models predict decreasing volatility followed by an increase, except for eGARCH which shows a consistent sharp decrease.

![ARCH Forecast](media/image1.png)
![GARCH Forecast](media/image2.png)
![tGARCH Forecast](media/image3.png)
![eGARCH Forecast](media/image4.png)

### Residuals Analysis
The residuals from both ARCH(1) and GARCH(1,1) models show negative skewness (-0.127 and -0.044 respectively), indicating non-normal distribution. Anderson-Darling tests confirm non-normality with p-values of 2.844e-07 for both models.

Ljung-Box tests reveal autocorrelation in ARCH(1) residuals (p < 0.01), while GARCH(1,1) residuals show independence (p > 0.01).

![Residuals Analysis](media/image5.png)
![Residuals Analysis](media/image6.png)

## 2. Moving Window Volatility and Correlations

### Data Summary
The portfolio consists of three "Big Five" technology companies (equal weighting):
- Facebook (FB): Social media giant with tremendous growth
- Apple (AAPL): Consumer electronics and software leader
- Amazon (AMZN): E-commerce and cloud computing powerhouse

Data spans from January 2012 to May 20, 2020 (2,107 observations).

### Volatility for Various Window Lengths
Moving window volatility was calculated for window lengths of 10, 100, and 1000 observations.

![Window Length 10](media/image7.png)
![Window Length 100](media/image8.png)
![Window Length 1000](media/image9.png)

### Correlation for Various Window Lengths
Correlations between assets were calculated for window lengths of 10, 100, and 1000 observations.

![Correlation Window Length 10](media/image10.png)
![Correlation Window Length 100](media/image11.png)
![Correlation Window Length 1000](media/image12.png)

### Analysis of Window Length Sensitivity
- **Window Length 10**: Each new observation changes 10% of the data, causing high fluctuations. Observations at extremes move more dramatically, resulting in volatile estimates.
- **Window Length 300**: Each new observation changes only 0.33% of the data, providing more stability.
- **Window Length 1000**: Much less sensitive to extremes but potentially less relevant to current market conditions.

## 3. Backtesting Historical Simulation and EWMA

### Data Summary
The portfolio consists of three stocks (equal weighting):
- Facebook (FB): Social media giant
- Apple (AAPL): Consumer electronics leader
- Tesla (TSLA): Electric vehicle and clean energy company

Data spans from January 2012 to May 20, 2020 (2,107 observations).

### Portfolio Return Summary Statistics
| Statistic | Value |
|-----------|-------|
| Min | -16.5666% |
| Max | -10.9948% |
| 1st Quartile | -0.8096% |
| 3rd Quartile | 1.1169% |
| Median | 0.1655% |
| Mean | 0.1106% |
| Standard Deviation | 1.836152% |
| Skewness | -0.4745738 |
| Kurtosis | 9.513039 |

### Backtesting Results
Using a window length of 1,000 observations for 1% VaR:

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|-------------------|-----------------|----------------|
| EWMA | 21 | 62 | 2.964427 | 0.0205082 |
| HS | 21 | 35 | 1.679842 | 0.003369895 |

The EWMA method performs poorly with a violation ratio (VR) of 2.96, far from the ideal 1.00. The Historical Simulation (HS) method shows better results with a VR of 1.68, though still not ideal.

![Backtesting HS and EWMA](media/image13.png)

The graph shows that EWMA VaRs are more volatile than HS. EWMA quickly adapts to high volatility periods and sharply adjusts VaR forecasts, while HS reacts more slowly to changing risk profiles.

## 4. Backtesting HS and GARCH

### Data Summary
The portfolio consists of three stocks (equal weighting):
- Apple (AAPL): Consumer electronics and software leader
- Facebook (FB): Social media giant
- Amazon (AMZN): E-commerce and cloud computing leader

Data spans from January 2012 to May 21, 2020 (2,108 observations).

### Backtesting Results for Apple (AAPL)
Using a window length of 500 observations for 1% VaR:

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|-------------------|-----------------|----------------|
| HS | 21 | 25 | 1.119403 | 0.008001864 |
| nGARCH | 21 | 35 | 1.679104 | 0.01527992 |
| tGARCH | 21 | 38 | 1.616915 | 0.01638113 |

![Backtesting AAPL](media/image14.png)

The Historical Simulation performs best with a VR of 1.12, while normal GARCH and Student-t GARCH show higher VRs (1.68 and 1.62). However, the GARCH models respond faster to market movements due to their higher volatility (0.015 and 0.016).

### Backtesting Results for Facebook (FB)
Using a window length of 500 observations for 1% VaR:

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|-------------------|-----------------|----------------|
| HS | 21 | 26 | 1.189689 | 0.008167021 |
| nGARCH | 21 | 39 | 1.652346 | 0.01789841 |
| tGARCH | 21 | 33 | 1.784534 | 0.02651924 |

![Backtesting FB](media/image15.png)

All three models have high violation ratios, with FB returns showing more fluctuations than AAPL. Student-t GARCH, with its higher VaR volatility of 0.027, is more sensitive to return movements.

### Backtesting Results for Amazon (AMZN)
Using a window length of 500 observations for 1% VaR:

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|-------------------|-----------------|----------------|
| HS | 21 | 26 | 1.057214 | 0.01034832 |
| nGARCH | 21 | 39 | 1.119403 | 0.0142786 |
| tGARCH | 21 | 33 | 1.30597 | 0.01447579 |

![Backtesting AMZN](media/image16.png)

The HS method shows the best results with a VR close to 1. For AMZN returns, normal GARCH performs better than Student-t GARCH, though both respond faster to market changes than HS.

## 5. Expected Shortfall Backtesting

### Data Summary
Three technology stocks were selected for individual backtesting with a window length of 500 observations:
- Dell Technologies (DELL): Computer hardware and services
- IBM: Global technology and cloud computing provider
- Intel (INTC): Semiconductor manufacturer

Data spans from January 2012 to May 22, 2020 (2,109 observations).

### VaR and ES Backtest for Dell (DELL)
- **VaR Backtest**: 7 actual exceedances vs. 4.5 expected at 99% confidence level
  - Kupiec test p-value: 0.264 (fail to reject null at 1%)
  - Christoffersen test p-value: 0.48 (fail to reject null at 1%)
- **ES Backtest**: p-value: 0.02, rejecting null hypothesis at 5% level
  
The ES backtest indicates inadequacy in the GARCH(1,1) model for Dell at 5% significance level.

### VaR and ES Backtest for IBM
- **VaR Backtest**: 27 actual exceedances vs. 16.1 expected at 99% confidence level
  - Kupiec test p-value: 0.013 (fail to reject null at 1%)
  - Christoffersen test p-value: 0.035 (reject null at 5%)
- **ES Backtest**: p-value: 7.72e-09, rejecting null hypothesis at 1% level

Both VaR and ES backtests indicate that the GARCH(1,1) model fails to accurately forecast IBM's risk at the 5% and 1% significance levels, respectively.

### VaR and ES Backtest for Intel (INTC)
- **VaR Backtest**: 16 actual exceedances (matching expected) at 99% confidence level
  - Kupiec test p-value: 0.982 (fail to reject null at 1%)
  - Christoffersen test p-value: 0.851 (fail to reject null at 1%)
- **ES Backtest**: p-value: 0.0002, rejecting null hypothesis at 1% level

While the VaR model appears adequate, the ES backtest indicates that the model systematically understates the underlying level of risk at the 1% significance level.

## 6. EWMA Backtesting with Multiple Methods

### Data Summary
The portfolio consists of three top streaming service providers (weighted as 1/4, 1/4, 1/2):
- Netflix (NFLX): Leading streaming media provider
- Roku (ROKU): Digital media player manufacturer
- Disney (DIS): Diversified entertainment conglomerate

Data spans from January 2012 to May 23, 2020 (2,111 observations).

### Violation Ratio and VaR Volatility
Using a window length of 500 observations for 1% VaR:

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|-------------------|-----------------|----------------|
| EWMA | 21 | 53 | 2.515723 | 0.01880755 |

The EWMA method performs poorly with a VR of 2.52, well above the ideal 1.00.

![EWMA Backtesting](media/image17.png)

The graph shows highly volatile VaR estimates that respond quickly to market changes.

### Coverage and Independence Tests
- Actual exceedances: 16 vs. expected: 31 at 95% confidence level
- Unconditional test p-value: 0.0015 (reject null at 5%)
- Conditional test p-value: 0.0044 (reject null at 5%)

Both tests reject the null hypotheses of correct exceedances and independence, indicating that the EWMA model is not suitable for VaR forecasting for this portfolio.

## 7. Advanced Backtests for EWMA

### Data Summary
This analysis focuses on Netflix (NFLX) stock returns from January 1, 2012, to May 23, 2020 (2,110 observations).

### EWMA Backtest Results
Using a minimum window length of 30 observations for 1% VaR:

| Method | Expected Violations | Actual Violations | Violation Ratio | VaR Volatility |
|--------|---------------------|-------------------|-----------------|----------------|
| EWMA | 21 | 39 | 1.826923 | 0.02572906 |

According to standard assessment rules:
- VR ∈ [0.8, 1.2]: Model is good
- VR ∈ [0.5, 0.8] or [1.2, 1.5]: Model is acceptable
- VR ∈ [0.3, 0.5] or [1.5, 2]: Model is bad
- VR < 0.3 or > 2: Model is useless

With a VR of 1.83, the EWMA model is classified as "bad" for Netflix returns.

![Advanced EWMA Backtesting](media/image18.png)

### Statistical Tests for Violation Ratios and Clustering

#### Unconditional Coverage Test
- Actual exceedances: 38 vs. expected: 104 at 95% confidence level
- Test statistic: 57.66
- p-value: 3.11e-14 (reject null at 5%)

The test strongly rejects the null hypothesis of correct exceedances, indicating that the model systematically understates risk.

#### Independence Test (for Clustering)
- Test statistic: 59.38
- p-value: 1.28e-13 (reject null at 5%)

The test rejects the null hypothesis of no clustering, indicating that violations are not independent. This means today's violation can predict tomorrow's violation, suggesting an inadequate VaR model for Netflix returns.
