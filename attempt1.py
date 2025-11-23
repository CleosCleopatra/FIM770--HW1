from scipy.optimize import fsolve
from scipy.misc import derivative
import numpy as np


def f(x, r):
    z = np.clip(80*(1-x), -500, 500) #Is this good/allowed?
    return 1/5 + 7/10 / (1+ np.exp(z))-r*x**4

rs=np.arange(0, 1, 0.001)

guesses =[0.0, 0.3, 0.6]
branches = [[] for _ in guesses]
stability = [[] for _ in guesses]

def dfdx(x, r):
    h = 1e-6
    return (f(x+h, r) - f(x-h, r)) / (2*h)



for r in rs:
    for i, guess in enumerate(guesses):
        root = fsolve(f, x0=guess, args=(r,))
        if len(root) > 0:
            root = root[0]
        else: 
            continue
        branches[i].append(root)
        f_prime = dfdx(root, r)
        stability[i].append(f_prime < 0)
        print(f"guesses is {guesses} and root is {root}")
        guesses[i] = root
        #roots_r.append(root)
    #roots_r = list(set(np.round(roots_r, 6)))
    
import matplotlib.pyplot as plt


plt.figure(figsize=(8,6))
for branch, stable in zip(branches, stability):
    branch = np.array(branch)
    stable = np.array(stable)
    plt.plot(rs[stable], branch[stable], 'b-', linewidth =2)
    plt.plot(rs[~stable], branch[~stable], 'r--', linewidth=2)


plt.xlabel('r')
plt.ylabel('x* (Fixed points)')
plt.title('Bifurcation Diagram of Climate Model')
plt.legend()
plt.show()