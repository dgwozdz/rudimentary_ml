#==============================================================================
### Rudimentary ML: Ordinary Least Squares
### Author: Damian Gwóźdź
### Date: 31JAN2018
### Last modified: -
#==============================================================================

# 0. Loading libraries

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# 1. Implementation

def ols(X, y, intercept = True):
    if intercept:
        vec_ones = np.ones((X.shape[0], 1))
        X = np.hstack((vec_ones, diabetes.data))
        
    betas = np.matmul(
            np.matmul(
                    np.linalg.inv(np.matmul(np.transpose(X), X)),
                    np.transpose(X)),
                    y)
    return betas

# 2. Example

diabetes = datasets.load_diabetes()
ols_no_intercept = ols(diabetes.data, diabetes.target, intercept = False)
ols_intercept = ols(diabetes.data, diabetes.target)

# 3. Check
lm = LinearRegression(fit_intercept = False)
lm.fit(diabetes.data, diabetes.target)
print(lm.intercept_, lm.coef_)
print(ols_no_intercept)

lm = LinearRegression(fit_intercept = True)
lm.fit(diabetes.data, diabetes.target)
print(lm.intercept_, lm.coef_)
ols_intercept
