import numpy as np
from pymetrics.utils import regr, _output_, _derivs_
from scipy.stats import t, f, norm
from scipy.special import gamma
from scipy.optimize import minimize

class linreg:
    def __init__(self, endo, exo, intercept=True, robust=True):
        
        if isinstance(exo, pd.DataFrame):
            self.names = exo.columns
        
        if not map(lambda z: isinstance(z, np.ndarray), [endo, exo]):
            endo, exo = np.array(endo), np.array(exo)
        
        self.n, self.p = exo.shape
        self.intercept = intercept
        self.endo, self.exo = endo, exo
        self.dof = self.n - self.p - intercept
        
        if hasattr(self, 'names') and intercept is True:
            self.names = self.names.insert(0, '(Intercept)')
        
        regression = _ols_(endo, exo, fit_constant=intercept)
        self.parameters= regression.beta
        self.residuals = regression.resids
        rss = np.sum(self.residuals ** 2)
        X = np.insert(exo, 0, np.ones(self.n), 1) if self.intercept else X
        XX, self.sigma = np.linalg.inv(X.T.dot(X)), rss / self.dof
        
        if robust:
            B = [np.outer(xi, xi) * ui ** 2 for xi, ui in zip(X, self.residuals)]
            self.var_cov = XX.dot(np.sum(B, 0)).dot(XX)
        else:
            self.var_cov = XX * self.sigma
        
        self.standard_errors = np.diag(self.var_cov) ** .5
        self.scores = self.parameters / self.standard_errors
        self.p_values = 2 * (1 - t.cdf(np.abs(self.scores), self.dof))
        self.rsq = 1 - (rss / np.sum((endo - np.mean(endo)) ** 2))
        self.adj_rsq = 1 - (1 - self.rsq) * ((self.n - 1) / self.dof)
        self.sig_overall = self.f_test(range(self.p))
    
    def f_test(self, idx):
        
        if not all([type(xj) == type(idx[0])] for xj in idx):
            raise ValueError('restrictions must be of same type')
        
        if all([type(xj) == str for xj in idx]):
            if not hasattr(self, 'names'):
                raise ValueError(
                    'regressors must have named columns to be indexed by strings'
                )
            
            idx = [self.names.index(xj) for xj in idx]
        
        regression = _ols_(self.endo, np.delete(self.exo, idx, 1))
        rss1, rss2 = np.sum(regression.resids ** 2), np.sum(self.residuals ** 2)
        statistic = ((rss1 - rss2) / len(idx)) / (rss2 / self.dof)
        p_value = 1 - f.cdf(statistic, len(idx), self.dof)
        return [statistic, p_value]
    
    def summary(self):
        _output_(self)
        rnd = lambda v: '{:.4f}'.format(v)
        f, rsd = self.sig_overall, rnd(np.std(self.residuals))
        print('')
        print('Reisidual standard error', rnd(rsd), 'on', self.dof,
                'degrees of freeom')
        print('Multiple R-squared:', rnd(self.rsq),' Adjusted R-squared:',
                rnd(self.adj_rsq))
        
        fpv = '< 1e-4' if f[1] < 1e-4 else rnd(f[1])
        print('F-statistic: ', rnd(f[0]), 'on',
                self.parameters.size - model.intercept, 'and',
                self.n - self.parameters.size, 'DF,  p-value:', fpv)
        print('')



class poissreg:
    def __init__(self, endo, exo, intercept=True):
        
        if isinstance(exo, pd.DataFrame):
            self.names = exo.columns
            if intercept: self.names = self.names.insert(0, '(Intercept)')
        
        if not map(lambda z: isinstance(z, np.ndarray), [endo, exo]):
            endo, exo = np.array(endo), np.array(exo)
        
        self.n, self.p = exo.shape
        self.intercept = intercept
        self.dof = self.n - self.p - self.intercept
        self.endo, self.exo = endo, exo
        
        if intercept: exo = np.insert(exo, 0, np.ones(self.n), 1)
        X, y = exo, endo
        mu = lambda b: np.exp(X.dot(b))
        
        def _loglike_(params):
            m = mu(params)
            return -np.sum(y * np.log(m) - m)
        
        self.optim = minimize(_loglike_, np.repeat(0, X.shape[1]), tol=1e-6)
        self.parameters = self.optim.x
        
        Jacobian = _derivs_(mu).pderivs(self.parameters).T
        m = mu(self.parameters)
        scores = Jacobian.T * (y - m) / m
        A = np.sum([np.outer(Ji, Ji) / mi for Ji, mi in zip(Jacobian, m)], 0)
        A, S = np.linalg.inv(A), np.sum([np.outer(si, si) for si in scores.T],0)
        
        self.var_param = A.dot(S).dot(A)
        self.standard_errors = np.diag(self.var_param) ** .5
        self.scores = self.parameters / self.standard_errors
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.scores)))
        self.residuals = y - m
        self.fitted_values = m


class negbinreg:
    def __init__(self, endo, exo, intercept=True):
        
        if isinstance(exo, pd.DataFrame):
            self.names = exo.columns
            if intercept: self.names = self.names.insert(0, '(Intercept)')
        
        if not map(lambda z: isinstance(z, np.ndarray), [endo, exo]):
            endo, exo = np.array(endo), np.array(exo)
        
        self.n, self.p = exo.shape
        self.intercept = intercept
        self.dof = self.n - self.p - self.intercept
        self.endo, self.exo = endo, exo
        
        if intercept: exo = np.insert(exo, 0, np.ones(self.n), 1)
        mu = lambda b: np.exp(exo.dot(b))
        
        poiss = PoissonRegression(y, x, intercept=intercept)
        mu, res = poiss.fitted_values, poiss.residuals
        alpha = np.sum((res ** 2 - mu) / mu ** 2) / self.dof
        theta = np.append(poiss.parameters, alpha)
        
        def _loglike_(theta):
            alpha, mu = theta[-1], np.exp(exo.dot(theta[:-1]))
            lgamma = lambda z: np.log(gamma(z))
            return -np.sum(
                lgamma(alpha + y) - lgamma(alpha) - lgamma(y + 1)
                + alpha * np.log(alpha) + y * np.log(mu + (y == 0))
                - (alpha + y) * np.log(alpha + mu)
                )
        
        self.optim = minimize(_loglike_, theta, tol=1e-4)



