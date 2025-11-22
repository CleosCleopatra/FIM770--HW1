from scipy.optimize import fsolve
import numpy as np

def f(x, r):
    return 1/5 + 7/10 / (1+ np.exp(80*(1-x)))-r*x**4

for r in range(0,1, 0.001):
    root = fsolve(f, x0=0.5, args=(r,))