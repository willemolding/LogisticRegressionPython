"""
A logistic regression classifier that uses a similar interface to scikit learn. 
"""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class BinaryLogisticRegression:

    def __init__(self, C=1.):
        self.C = C

    def fit(self, X, y):
        # check inputs here

        self._beta = np.zeros((X.shape[1] + 1, 1))
        result = fmin_l_bfgs_b(_binary_objective, self._beta, args=(X, y, self.C))

        self._beta = result[0]
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        # add the constant
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        XBeta = np.dot(X, self._beta).reshape((-1,1))

        probs = 1./(1. + np.exp(-XBeta))
        return np.hstack((1 - probs, probs))


def _binary_objective(beta, X, y, C):
    """
    The objective function. Returns the regularized negative log-likelihood and gradient.
    C is the inverse regularization parameter
    """

    # add a constant column to X to simplify the intercept
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # make y a column vector for array broadcasting reasons
    y = y.reshape((-1,1))

    # precomptue the features multiplied by the coefficients
    XBeta = np.dot(X, beta).reshape((-1,1))

    # also precompute the exponents as they are used several times
    exp_XBeta = np.exp(XBeta)

    # the negative log-likelihood
    neg_ll = C*np.sum(np.log(1. + exp_XBeta) - y*XBeta, axis=0) + 0.5*np.inner(beta, beta)

    # the gradient of the negative log-likelihood
    grad_neg_ll = C*np.sum((1./ (1. + exp_XBeta))*exp_XBeta*X - y*X, axis=0) + beta

    return neg_ll, grad_neg_ll




