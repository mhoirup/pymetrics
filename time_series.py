import numpy as np
from pymetrics.utils import regr, _signi_, _output_
from pymetrics.cross_section import linreg
from scipy.stats import chi2, norm
from scipy.optimize import minimize


def embed(x, lags):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    nrows, nvars = x.shape if x.ndim == 2 else [x.size, 1]
    matrix = np.zeros([nrows - lags, (nvars * lags) + nvars])
    for t in range(lags, nrows):
        if x.ndim == 1: xt = np.flip(x[(t - lags):(t + 1)])
        else: xt = np.array([x[t-i] for i in range(lags + 1)])
        matrix[t - lags] = xt.reshape(1, matrix.shape[1])
    
    return matrix


def covar(x, nobs=None):
    if x.ndim != 1:
        raise ValueError('only supported for univariate series')
    
    if nobs is None: nobs = x.size
    Gamma, sigma = np.zeros([nobs, nobs]), np.mean((x - np.mean(x)) ** 2)
    rho = autocorr(x, nobs, type='acf').rho
    for i in range(nobs):
        row = np.insert(rho[:(nobs - (i + 1))], 0, 1)
        Gamma[i] = np.concatenate([np.flip(rho[:i]), row])
    
    return Gamma * sigma


class autocorr:
    def __init__(self, x, max_lag=30, corr='acf'):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        N, lags, xdm = x.size, np.arange(1, max_lag+1), x - np.mean(x)
        denom = np.sum(xdm ** 2)
        rho = [np.sum(xdm[i:] * xdm[:(N-i)]) / denom for i in lags]
        rho, ci = np.array(rho), norm.isf(.025) / (N ** .5)
        self.rho, self.ci, self.corr, self.max_lag = rho, ci, corr, max_lag
        
        if corr == 'pacf':
            prho = np.diag([rho[0]] * max_lag)
            prho[1,1] = (rho[1] - rho[0] ** 2) / (1 - rho[0] ** 2)
            for i in range(2, max_lag):
                prev, new = np.trim_zeros(prho[:,i-2]), np.trim_zeros(prho[:,i-1])
                new = np.concatenate([prev - new * np.flip(prev), new])
                prho[:i,i-1] = new
                numerator = rho[i] - np.sum(new * np.flip(rho[:i]))
                denominator = 1 - np.sum(new * rho[:i])
                prho[i,i] = numerator / denominator
            
            self.rho = np.diag(prho)
    
    def result(self):
        
        corr, max_lag = self.corr.upper(), self.max_lag
        lags = np.arange(1, self.max_lag + 1)
        print(' ' * 2, corr, 'of series:')
        print(' ' * 2, 'Confidence interval:', '{:.3f}'.format(self.ci))
        print(''); print(corr, 'by lag:')
        splits = np.split(lags, np.arange(8, (max_lag - (max_lag % 8)) + 1, 8))
        for split in splits:
            rm = ['{:.3f}'.format(r) for r in self.rho[split - 1]]
            print(('{:>7} '*len(split)).format(*split))
            print(('{:>7} '*len(split)).format(*rm))
        print('')
        print(*['*' if ri > np.abs(self.ci) else ' ' for ri in self.rho])
        print('-' * 60)
        print(*['*' if ri < -np.abs(self.ci) else ' ' for ri in self.rho])


def crosscorr(data, max_lag=24):
    if type(data).__name__ == 'VectorAutoRegression':
        X, C = data.residuals, data.Sigma
        nrows = X.shape[0]
    else:
        X, nrows = data - np.mean(data, 0), data.shape[0]
        C = X.T.dot(X) * (1 / (nrows-1))
    D, nvars = np.linalg.inv(np.diag(np.diag(C) ** .5)), X.shape[1]
    R = np.zeros([max_lag + 1, nvars, nvars])
    R[0] = D.dot(C).dot(D)
    for i in range(1, max_lag + 1):
        Ci = X[i:].T.dot(X[:(nrows-i)]) / nrows
        R[i] = D.dot(Ci).dot(D)
    return R


class qstat:
    def __init__(self, x, lags, fitdf=0):
        self.dimensions, m, self.fitdf = x.ndim, lags, fitdf
        if not isinstance(m, np.ndarray):
            m = np.array(m)
        m = m[m != 0]
        self.lags = m
        if self.dimensions == 1:
            N, rho = x.size, autocorr(x, max_lag=np.max(m)).rho
            term, self.dfreedom = N * (N + 2), m - fitdf
            self.statistics = [np.sum(rho[:i] ** 2 / (N-i)) * term for i in m]
            self.p_values = 1 - chi2.cdf(self.statistics, self.dfreedom)
        else:
            R, N = crosscorr(x, max_lag=np.max(m) + 1), x.shape[0]
            R0, statistics = np.linalg.inv(R[0]), np.zeros(np.max(m) + 1)
            R0, self.dfreedom = np.kron(R0, R0), (m * x.shape[1] ** 2) - fitdf
            for i in lags:
                b = np.array([c for c in R[i].T]).reshape(R[i].size).T
                statistics[i] += b.T.dot(R0).dot(b) * (1/(N-i))
            self.statistics = np.cumsum(statistics[1:]) * (N ** 2)
            self.dfreedom = m * x.shape[1] ** 2
            self.p_values = 1 - chi2.cdf(self.statistics, self.dfreedom)
    
    def output(self):
        pformat = lambda s: '{0:>4} {1:>5} {2:>8} {3:>8} {4}'.format(*s)
        print(pformat(['m', 'dof', 'Q(m)', 'p-value', ''])); print('-' * 33)
        m = self.lags
        for i in range(self.dfreedom.size):
            strings = [
                    m[i], self.dfreedom[i],
                    '{:.3f}'.format(self.statistics[i]),
                    '{:.4f}'.format(self.p_values[i]),
                    _signi_(self.p_values[i])
                ]
            print(pformat(strings))
        print('')


class adf:
    def __init__(self, series, max_lag=10, trend='none', criterium='aic'):
        if trend not in ['none', 'drift', 'time']:
            raise ValueError('test_type must be either none, drift or time')
        if criterium not in ['aic', 'bic']:
            raise ValueError('criterium must be either aic or bic')
        if not isinstance(series, np.ndarray):
            series = np.array(series)
        x = series
        dx = np.diff(x)
        data = np.insert(embed(dx, max_lag), 1, x[max_lag:dx.size], 1)
        nrows, nvars = data.shape
        lag_names = ['x.lag1']
        if trend in ['drift', 'time']:
            data = np.insert(data, 1, np.ones(nrows), 1)
            lag_names.insert(0, '(Intercept)')
            if trend == 'time':
                data = np.insert(data, 3, np.arange(max_lag+1, x.size), 1)
                lag_names.insert(2, 'time.trend')
            nvars = data.shape[1]
        best_crit, X, y = 1e+8, data[:,1:], data[:,0]
        
        for i in range(nvars - max_lag, nvars):
            model = linreg(y, X[:,range(i)], intercept=False)
            info_crit = np.sum(model.residuals ** 2) * nrows
            info_crit += 2 * i if criterium == 'aic' else i * np.log(nrows)
            if info_crit < best_crit:
                best_crit, unrestricted = info_crit, model
                p = (i - len(lag_names) + 1)
                lag_labels = ['x.diff.lag' + str(i) for i in range(1, p)]
        
        lag_names = [label for s in [lag_names, lag_labels] for label in s]
        unrestricted.summary(names=lag_names)
        statistics = [unrestricted.scores[0 if test_type == 'none' else 1]]
        
        if test_type in ['drift', 'trend']:
            indices = [[0,1]] if test_type == 'drift' else [[0,1,2], [1,2]]
            for idx in indices:
                statistics.append(unrestricted.f_test(idx)[0])
        
        stats_formatted = (' {:.3f} '* len(statistics)).format(*statistics)
        print('Value of test-statistic(s):', stats_formatted)
        print(''); print('Critical values for test statistic(s):')
        print(' '*4, ('{:>5} ' * 3).format(*['1pct', '5pct', '10pct']))
        critvals = {
            'none':  {'tau1': [-2.58, -1.95, -1.62]}, 
            'drift': {'tau2': [-3.43, -2.86, -2.57], 
                      'phi1': [ 6.43,  4.59,  3.78]},
            'trend': {'tau3': [-3.96, -3.41, -3.12],
                      'phi2': [ 6.09,  4.68,  4.03],
                      'phi3': [ 8.27,  6.25,  5.34]}
        }
        for stat in critvals[trend].keys():
            print(stat, ('{:>5} '*3).format(*critvals[test_type][stat]))
        print('')


class arima:
    def __init__(self, x, order=[0,0], mean=True, method='CSS-ML'):
        
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim > 1:
            raise ValueError('series must be univariate')
        self.series, self.order, self.mean = x, order, mean
        N, Np = x.size, np.sum(self.order) + self.mean
        
        if self.order[1] == 0:
            X = embed(x, self.order[0])
            initparams = regr(X[:,0], X[:,1:], const=False).beta
        else:
            m = int(np.round(N/3)) + Np
            Rm, rho = np.zeros([m, m]), autocorr(x, m).rho
            for i in range(m):
                Ri = np.insert(rho[:(m - (i+1))], 0, 1)
                Rm[i] = np.concatenate([np.flip(rho[:i]), Ri])
            phi = np.linalg.inv(Rm).dot(rho)
            z = [x[i] - np.sum(x[(i-m):i] * phi) for i in range(m, N)]
            Z = embed(np.array(z), self.order[1])
            Zy, Zx = Z[:,0], Z[:,1:]
            if self.order[0] != 0:
                X = embed(x[m:], self.order[0])[:,1:]
                Nx, Nz = X.shape[0], Zx.shape[0]
                if Nx != Nz:
                    delta = np.abs(Nx - Nz)
                    if Nx > Nz:
                        X = X[delta:]
                    else:
                        Zy, Zx = Zy[delta:], Zx[delta:]
                Zx = np.concatenate([X, Zx], 1)
            initparams = regr(Zy, Zx, const=False).beta
        
        def _resids_(params):
            z = x - np.mean(x) if self.mean else x
            p, q = self.order
            zp = embed(np.insert(z, 0, np.zeros(p)), p)
            zp = zp[:,0] - np.sum(zp[:,1:] * params[:p], 1)
            theta, res = params[p:][::-1], np.zeros(q)
            for i in range(zp.size):
                aq = np.sum(res[-q:] * theta) if q > 0 else 0
                res = np.append(res, zp[i] - aq)
            return res[q:]
     
        def _loglike_(params):
            N2, rss = N/2, np.sum(_resids_(params) ** 2)
            sigma, lnpi = rss / N, np.log(2 * np.pi)
            return -N2 * lnpi - N2 * np.log(sigma) - (rss / (2 * sigma))
        
        if method in ['CSS', 'CSS-ML']:
            CSS = lambda params: np.sum(_resids_(params) ** 2)
            self.optim = minimize(CSS, initparams, tol=1e-3)
        if method in ['ML', 'CSS-ML']:
            initparams = initparams if method == 'ML' else self.optim.x
            neg_loglike = lambda params: -_loglike_(params)
            self.optim = minimize(neg_loglike, initparams, tol=1e-3)
        
        params = self.optim.x
        resids = _resids_(params)
        dx = np.diag(np.repeat(1e-4, Np - self.mean))
        self.sigma = np.sum(resids ** 2) / resids.size
        X = np.array([_resids_(params + dxi) - resids for dxi in dx]).T / 1e-4
        if self.mean: X = np.insert(X, 0, np.ones(X.shape[0]), 1)
        self.param_cov = np.linalg.inv(X.T.dot(X)) * self.sigma
        
        self.parameters, self.residuals = params, resids
        if self.mean:
            self.parameters = np.insert(self.parameters, 0, np.mean(x))
        self.standard_errors = np.diag(self.param_cov) ** .5
        self.scores = self.parameters / self.standard_errors
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.scores)))
        if method in ['ML', 'CSS-ML']:
            self.loglike = -self.optim.fun
        else:
            self.loglike = _loglike_(self.optim.x)
        
        self.aic = -2 * self.loglike + 2 * Np
        self.bic = self.aic + (np.log(N) - 2) * Np
        self.aicc = self.aic + ((2 * Np ** 2 + 2 * Np) / (N - Np))
    
    def summary(self):
        mean = 'non-zero' if self.mean else 'zero'
        print('ARIMA({0},{1})'.format(*self.order), 'with', mean, 'mean')
        orders, names = [np.arange(1, i + 1) for i in self.order], list()
        for i in range(2):
            if orders[i].size != 0:
                greek = 'phi' if i == 0 else 'theta'
                for j in orders[i]: names.append(greek + str(j))
        if self.mean:
            names.insert(0, '(Intercept)')
        _output_(self, names)
        rnd = lambda v, d: '{:.{d}f}'.format(v, d=d); print('')
        info = [rnd(v, d) for v,d in zip([self.sigma, self.loglike], [6, 3])]
        print('Error variance: {0},  Log-Likelihood: {1}'.format(*info))
        info_crit = [rnd(v, 3) for v in [self.aic, self.aicc, self.bic]]
        print('AIC={0},  AICc={1},  BIC={2}'.format(*info_crit))
    
    def forecast(self, origin=None, horizon=10):
        x, parameters = self.series, self.optim.x
        mu = np.mean(x) if self.mean else 0 
        p, q = self.order
        phi, theta = parameters[:p][::-1], parameters[p:][::-1]
        if origin is None: origin = x.size
        fr, fx = self.residuals[:origin], x[:origin]
        for _ in range(horizon):
            fxi = np.sum(fr[-q:] * theta) if q != 0 else 0
            fxi = np.sum(fx[-p:] * phi) + fxi if p != 0 else fxi
            fx, fr = np.append(fx, mu + fxi), np.append(fr, 0)
        
        forecast_interval = 1.96 * self.sigma ** .5
        self.forecasts = {
            'mean': fx[origin:], 
            'LO95': fx[origin:] - forecast_interval,
            'HI95': fx[origin:] + forecast_interval
            }


class var:
    def __init__(self, series, order, intercept=True):
        if not isinstance(series, np.ndarray):
            series = np.array(series)
        self.intercept, self.p = intercept, order
        self.n, self.k = series.shape
        
        if order >= 0:
            data = embed(series, order)
            Z, X = data[:,:self.k], data[:,self.k:]
            if intercept: X = np.insert(X, 0, np.ones(self.n - order), 1)
            XX, XZ = np.linalg.inv(X.T.dot(X)), X.T.dot(Z)
            beta = XX.dot(XZ)
            A = Z - X.dot(beta)
            self.parameters = beta
        elif order == 0 and intercept is True:
            self.parameters = np.mean(series, 0)
            A = series - self.parameters
        
        self.Sigma = A.T.dot(A) / (self.n - order)
        
        if order != 0:
            V = np.kron(self.Sigma, XX)
            self.standard_errors = np.diag(V) ** .5
        
        self.residuals = A
        lndet = np.linalg.slogdet(self.Sigma)[1]
        info_crit_penalties = [
                (2 / self.n) * order * self.k ** 2,
                (np.log(self.n) / self.n) * order * self.k ** 2,
                (2 * np.log(np.log(self.n)) / self.n) * order * self.k ** 2
                ]
        info_crit = [lndet + penalty for penalty in info_crit_penalties]
        self.aic, self.bic, self.hq = info_crit
    
    def stationarity(self, return_eigen=False):
        B = self.parameters[self.intercept:]
        Phi = np.hstack(B.reshape((self.p, self.k, self.k)))
        
        if self.p > 1:
            dim1 = self.k if self.p == 2 else self.k * (self.p - 1)
            I, O = np.identity(dim1), np.zeros([dim1, self.k])
            Phi = np.vstack([Phi, np.hstack([I, O])])
        
        eigen = np.abs(np.linalg.eig(Phi)[0])
        stationary, nonstationary = 'Process stationary', 'Process not stationary'
        print(stationary) if np.all(eigen < 1) else print(nonstationary)
        if return_eigen: return eigen
    
    def irf(self, orthogonalise=True, max_lag=12):
        Phi = self.parameters[self.intercept:].reshape([self.p, self.k, self.k])
        Psi = np.zeros(max_lag, self.k, self.k)
        Psi = np.concatenate([np.identity(self.k), Psi], 0)
        
        for i in range(1, max_lag + 1):
            m = np.min([i, self.p])
            Psi[i] += np.sum([Phi[j].dot(Psi[i-j]) for j in range(1, m + 1)])
        
        if orthogonalise:
            U = np.linalg.cholesky(self.Sigma)
            for i in range(1, max_lag + 1):
                Psi[i] = Psi[i].dot(U.T)
        
        return Psi


def var_order(series, max_lag=10):
    if not isinstance(series, np.ndarray):
        series = np.array(series)
    
    n, k = series.shape
    determinants = np.zeros(max_lag + 1)
    pf = lambda s: '{0:>4} {1:>9} {2:>9} {3:>9} {4:>9} {5:>9} {6}'.format(*s)
    print(pf(['p', 'AIC', 'BIC', 'HQ', 'M(p)', 'p.value', ' ']))
    print('-' * 56)
    
    for i in range(max_lag + 1):
        model = VectorAutoregression(series, i, intercept=True)
        A = model.residuals
        MLSigma = A.T.dot(A) / (n - max_lag)
        determinants[i] = np.linalg.det(MLSigma)
        statistic, p_value = 0, 0
        if i != 0:
            ddetr = np.log(determinants[i] / determinants[i-1])
            statistic = -(n - max_lag - 1.5 - k * i) * ddetr
            p_value = 1 - chi2.cdf(statistic, k * k)
        
        info_crit = [model.aic, model.bic, model.hq]
        strings = [
            str(i),
            *['{:.4f}'.format(s) for s in [*info_crit, statistic, p_value]],
            _signi_(p_value)
        ]
        print(pf(strings))

