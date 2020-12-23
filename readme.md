# Pymetrics - Econometrics with Python

A custom econometrics library for Python with an emphasis on R-like output to
the console. Only dependencies are Numpy and Scipy. 

## Main References

Box, Jenkins, Reinsel & Ljung (2016), *Time Series Analysis: Forecasting and
Control*, 5th edition.  Hamilton (1994), *Time Series Analysis*, 1st edition.
Wooldridge (2010), *Econometric Analysis of Cross-Section and Panel Data*, 2nd
edition.


## Overview of Module

### General Utilities

| Class/Function  | Innner Method(s)                | Description                                                            |
| :-------------- | :------------------------------ | :--------------------------------------------------------------------- |
| `_ols_`         |                                 | Compute and hold OLS (via QR) estimates.                               |
| `_derivs_`      | `.pderivs()` <br/> `.hessian()` | Compute first and second partial derivatives of a continuous function. |
| `_signi_`       |                                 | Yield signifiance codes for *p*-values similar to those in R.          |
| `_output_`      |                                 | Print a regression summary similar to that in R.                       |

The `cross_section.py` submodule is for methods pertaining to cross-sectional data.

| Class/Function | Innner Method(s)             | Description                                              |
| :------------- | :--------------------------- | :------------------------------------------------------- |
| `linreg`       | `.f_test` <br/> `.summary()` | The OLS linear regression model.                         |
| `poissreg`     |                              | Poisson regression estimated via QMLE.                   |
| `negbinreg`    |                              | Negative binomial regression estimated via 2-stage QMLE. |

The `time_series.py` submodule is for methods pertaining to time series data. 

| Class/Function | Innner Method(s)         | Description                                                                      |
| :------------- | :----------------------- | :------------------------------------------------------------------------------- |
| `embed`        |                          | Lags an (*n x k*) array *m* times, returning an *(n - m) x ((k x m) + k)* array. |
| `covar`        |                          | Autocovariances of an array up to the *m*th lag (default *N*).                   |
| `autocorr`     | `.print()`               | ACF/PACF of a uni array. Can print a correlogram into the console.               |
| `crosscorr`    |                          | Cross-correlations for *m* lags of a multivariate array.                         |
| `qstat`        | `.print()`               |  *Q*-statistics for uni/multi arrays as well as model objects.                   |
| `adf`          |                          | The ADF test for unit roots. Supports trends and order selection via AIC/BIC.    |
| `arima`        | `.forecast()`            | ARMA(*p,q*) via CSS and/or MLE. Supports forecasting.                            |
| `var`          | `.stat()` <br/> `.irf()` | VAR(*p*) via OLS. Supports stationarity checks and multiplier analysis.          |
| `var_order`    |                          | Sequential likelihood tests for selecting VAR order                              |

* Currently performs no integration, so the name, for now, is a slight misnomer.

## A Small Benchmark

A small comparison between this library and similar methods in R. The data pertains to Hermès International S.A. equity in the period Jan. 2007 to Nov. 2020 (N = 3545). I worked with this data for my thesis, and it's quite an interesting asset compared to others like it.

### Some Preliminary Exploratory Analysis

As can be seen, the data behaves much as would be expected of financial returns.

![alt text](https://github.com/mhoirup/pymetrics/blob/master/plots/lineplot.png?raw=true)
![alt text](https://github.com/mhoirup/pymetrics/blob/master/plots/returns.png?raw=true)
![alt text](https://github.com/mhoirup/pymetrics/blob/master/plots/histogram.png?raw=true)
![alt text](https://github.com/mhoirup/pymetrics/blob/master/plots/ecdf.png?raw=true)
![alt text](https://github.com/mhoirup/pymetrics/blob/master/plots/correlations.png?raw=true)


### Model Estimation

ARMA(*p*,*q*) models are estimated on the series of returns. The custom `arima()` methods finds initial parameter values using the Hanna-Rissanen algorithm, and then estimates parameters via conditional-sum-of-squares, for then to finalise estimation via conditional MLE. Initial values for xi's and residuals are set to zero. Processes are estimated with orders < 3, with and without an intercept, and with selection based on the Akaike information criterium.

```python

best_aic, orders = 1e+9, range(3)
orders = [[p, q, c] for p in orders for q in orders for c in [True, False]]
orders = np.array(orders)[2:] # remove p = q = 0, because that's no fun

for order in orders:
    model = arima(x, order=order[:2], mean=order[2])
    
    if best_aic > model.aic:
        best_model = model
        best_aic = model.aic

best_model.summary()
# ARIMA(0,1) with non-zero mean
# Residuals:
#      Min        Q1    Median        Q3       Max
#  -0.1242   -0.0082    0.0001    0.0086    0.1548
#
# Coefficients:
#               Estimate Std. Error z-value Pr(>|z|)
# (Intercept)   0.000653   0.000293   2.227  0.02595 *
# theta1       -0.010712   0.016797  -0.638  0.52366
#
# Error variance: 0.000305,  Log-Likelihood: 9315.921
# AIC=-18627.843,  AICc=-18627.839,  BIC=-18615.497

resids, Np = best_model.residuals, best_model.parameters.size
qstat(resids, [8, 12, 16], Np).output()
#    m   dof     Q(m)  p-value
# -------------------------------
#   8     6   11.405   0.0766 .
#  12    10   25.450   0.0046 **
#  16    14   31.416   0.0048 **

```

```R

orders = expand.grid(0:2, 0, 0:2, c(T, F))[-c(1, 10),] 
best.bic = 1e+9

for (i in seq(nrow(orders))) {
    model = forecast::Arima(x, order = as.numeric(orders[i,1:3]),
                  include.mean = orders[i,4])
    if (model$aic < best.aic) {
        best.model = model
        best.aic = model$aic
    }
}

best.model
# Series: x
# ARIMA(0,0,1) with non-zero mean
#
# Coefficients:
#           ma1   mean
#       -0.0107  7e-04
# s.e.   0.0169  3e-04
#
# sigma^2 estimated as 0.0003052:  log likelihood=9315.92
# AIC=-18625.84   AICc=-18625.84   BIC=-18607.32

resids = best.model$residuals; Np = length(best.model$coef)
sapply(c(8, 12, 16), function(m) {
    Box.test(resids, m, 'Ljung-Box', Np)
})
#           [,1]             [,2]             [,3]
# statistic 11.39967         25.42684         31.36198
# parameter 6                10               14
# p.value   0.07678209       0.004592546      0.004931413
# method    "Box-Ljung test" "Box-Ljung test" "Box-Ljung test"
# data.name "resids"         "resids"         "resids"

```

We find that in both instances an MA(1) with non-zero mean is selected. Discrepancies do occur, most likely due to differences in how the inverted information matrix is computed, but overall results are very similar. Residuals still exhibit significant autocorrelations, as shown by the subsequent Q statistics (of the Ljung-Box type).




