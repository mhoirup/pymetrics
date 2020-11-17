import numpy as np

class regr(object):
    def __init__(self, y, x, const=True):
        
        # Barebone ols regression. Works with one or more regressors.
        # TODO: Add regression through the origin?
        
        if y.ndim > 1 or x.ndim > 2:
            raise ValueErorr('too many dimensions')
        
        if not map(lambda z: isinstance(z, np.ndarray), [x, y]):
            x, y = np.array(x), np.array(y)
        
        if x.ndim == 1:
            ydm, xdm = y - np.mean(y), x - np.mean(x)
            b = np.sum(xdm * ydm) / np.sum(xdm ** 2)
            self.fitted = x * b
            if const:
                b = np.array([np.mean(y) - (b * np.mean(x)), b])
                self.fitted += b[0]
            self.beta = b
        else:
            if const: x = np.insert(x, 0, np.ones(x.shape[0]), 1)
            Q, R = np.linalg.qr(x)
            self.beta = np.linalg.inv(R).dot(Q.T).dot(y)
            self.fitted = x.dot(self.beta)
        
        self.resids = y - self.fitted


class _derivs_(object):
    def __init__(self, fun, epsilon=1e-3, *args, **kwargs):
        
        # A class to compute and hold partial derivatives and hessian matrices.
        # Initialised with a function, with optional arguments being epsilon
        # and any fixed arguments to the function.
        
        self.fun = lambda x0: fun(x0, *args, **kwargs)
        self.epsilon = epsilon
    
    def pderivs(self, x0):
        
        f, epsilon = self.fun, self.epsilon
        dx, fx = np.diag(np.repeat(epsilon, x0.size)) + x0, f(x0)
        pderivs = np.array([(f(dxi) - fx) for dxi in dx])
        return pderivs / epsilon
    
    def hessian(self, x0):
        f, epsilon, m = self.fun, self.epsilon, x0.size
        dx, fx = np.diag(np.repeat(epsilon, m)), f(x0)
        H = np.diag([(f(x0-dxi) - 2*fx + f(x0+dxi)) for dxi in dx])
        H /= epsilon ** 2
        
        for i in range(m):
            dxp, dxm = dx[i] + x0, dx[i] - x0
            for j in range(i+1, m-1):
                dy = dx[j]; fdx = f(dxp+dy) - f(dxp-dy) - f(dxm+dy) + f(dxm-dy)
                H[i,j], H[j,i] = [fdx / (4 * epsilon ** 2)] * 2
        
        return H

class _param_optim_(object):
    def __init__(self, model, theta, fun=None, fisher_info=None, 
                 optim_method='nr', verbose=True, epsilon=1e-4, step=.1, 
                 max_iter=1000
                 ):
        
        # Most likely obsolete from now on, since I got scipy.minimize
        # working. Keeping it just in case. 
        
        if optim_method is None:
            raise ValueError('optimisation method must be specified')
        
        pformat = lambda strings: '{0:>22} {1:>11}'.format(*strings)
        i, stoprule = 0, 1
        header = ['Stop Rule (<= {0})'.format(epsilon), 'Iteration']
        
        if optim_method == 'nr':
            if fun is None: 
                raise ValueError('likelihood function missing')
            
            if verbose: print(pformat(header), '\n', '-' * 35)
            # The Newton-Raphson algorithm for maximum likelihood estimation
            # of theta. Similarly to the IWLS scheme, the stopping rule,
            # the dot product of the score and direction, is printed
            # (if verbose=True).
            derivs = _derivatives_(fun)
            if fisher_info is None:
                fisher_info = derivs.hessian
            
            while stoprule >= epsilon and i <= max_iter:
                s, I = derivs.pderivs(theta), fisher_info(theta)
                direction = I.dot(s)
                theta_new = theta + (step * direction)
                stoprule = np.abs(s.dot(direction))
                if verbose: print(pformat(['{:.4f}'.format(stoprule), i]))
                theta, i = theta_new, i + 1
            
        
        self.theta = theta
        self.n_iter = i


def _signi_(p_value):
    codes = ['***', '**', '*', '.', '', '']
    signi = p_value <= np.array([0.001, 0.01, 0.05, 0.1, 1])
    return codes[np.min(np.where(signi == True))]


def _output_(model, names=None):
    
    if names is not None and len(names) != model.parameters.size:
        raise ValueError('len(names) != len(parameters)')
    
    if names == None:
        if hasattr(model, 'names'):
            names = model.names
        else:
            names = ['X' + str(i) for i in range(model.parameters.size)]
        if model.intercept:
            names[0] = '(Intercept)'
    
    qtiles = [np.quantile(model.residuals, q) for q in np.linspace(0, 1, 5)]
    qtiles = ['{:.4f}'.format(qtile) for qtile in qtiles]
    print('Residuals:')
    print(('{:>8}  '*5).format(*['Min', 'Q1', 'Median', 'Q3', 'Max']))
    print(('{:>8}  '*5).format(*qtiles)); print('')
    
    print('Coefficients:')
    names = [name[:10] + '..' if len(name) > 12 else name for name in names]
    header = ['', 'Estimate', 'Std. Error', 't-value', 'Pr(>|t|)']
    
    if hasattr(model, 'loglike'):
        for i in [3,4]: header[i] = header[i].replace('t', 'z')
    
    print('{0:>12} {1:>9} {2:>10} {3:>7} {4:>8}'.format(*header))
    fmt = lambda x, d, w: '{:>{w}}'.format('{:.{d}f}'.format(x, d=d), w=w)
    
    for i in range(model.parameters.size):
        parameter_row = [
            '{:<12}'.format(names[i]),
            fmt(model.parameters[i], 6, 9),
            fmt(model.standard_errors[i], 6, 10),
            fmt(model.scores[i], 3, 7),
            fmt(model.p_values[i], 5, 8),
            _signi_(model.p_values[i])
        ]
        print(*parameter_row)

