
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def F(x, r):
    g = 1.0 / (1.0 + np.exp(np.clip(80*(1 - x), -500, 500)))
    return 0.2 + 0.7 * g - r * x**4

def dFdx(x, r):
    z = np.clip(80*(1 - x), -500, 500)
    g = 1.0 / (1.0 + np.exp(z))
    return 0.7 * 80 * g * (1 - g) - 4 * r * x**3


def find_roots_for_r(r, x_min=0.0, x_max=2.0, n_intervals=800):
    xs = np.linspace(x_min, x_max, n_intervals + 1)
    Fx = F(xs, r)
    roots = []
    for i in range(n_intervals):
        a, b = xs[i], xs[i+1]
        Fa, Fb = Fx[i], Fx[i+1]
        if np.isnan(Fa) or np.isnan(Fb):
            continue
        if Fa * Fb < 0:
            try:
                xr = brentq(lambda x: F(x, r), a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
                roots.append(xr)
            except ValueError:
                pass

    roots = np.array(sorted(roots))
    if roots.size == 0:
        return np.array([]), np.array([])
    keep = [0]
    for i in range(1, len(roots)):
        if abs(roots[i] - roots[keep[-1]]) > 1e-6:
            keep.append(i)
    roots = roots[keep]
    stab = dFdx(roots, r) < 0
    return roots, stab

rs = np.linspace(0.0, 1.0, 501)
all_r = []
all_x = []
all_stable = []
root_counts = []

for r in rs:
    roots, stab = find_roots_for_r(r)
    root_counts.append(np.sum(stab))
    for x,s in zip(roots, stab):
        all_r.append(r)
        all_x.append(x)
        all_stable.append(s)

all_r = np.array(all_r)
all_x = np.array(all_x)
all_stable = np.array(all_stable)

bif_rs = []
for i in range(1, len(root_counts)):
    if root_counts[i] != root_counts[i-1]:
        bif_rs.append(0.5 * (rs[i] + rs[i-1]))
bistable_mask = np.array(root_counts) >= 2
r_bistable = rs[bistable_mask]
if r_bistable.size > 0:
    r_lo, r_hi = r_bistable[0], r_bistable[-1]

# Plot
plt.figure(figsize=(8,6))


#for r_vals, x_vals, stab_vals in all_r, all_x, all_stable:
#    r_vals = np.array(r_vals); x_vals = np.array(x_vals); stab_vals = np.array(stab_vals)
#    plt.plot(r_vals[stab_vals], x_vals[stab_vals], '-', linewidth = 2)
    
lable_added = False
stable_r_list = []
stable_x_list =  []
label_added = False
for i, r in enumerate(rs):
    roots, stab = find_roots_for_r(r)
    for x, s in zip(roots, stab):
        if s: 
            if not lable_added:
                plt.plot(r,x, 'go', label = 'stable')
                lable_added = True
                print("lable aded")
            else:
                plt.plot(r,x,'go')
#plt.scatter(all_r[all_stable], all_x[all_stable], s=12, c='green', label='stable')
plt.plot(all_r[~all_stable], all_x[~all_stable], ':', label = 'unstable')
#plt.plot(all_r[all_stable], all_x[all_stable], '-', linewidth=1.8, label='stable')
#plt.plot(all_r[~all_stable], all_x[~all_stable], ':', linewidth=1.8, label='unstable')

for br in bif_rs:
    if br > 0.2:
        plt.axvline(br, linestyle='--', linewidth=1.0, color='gray')
        plt.text(br, 0.95, 'saddle-node', rotation=90, va='top', ha='right', fontsize=8)
    print(br)

if len(bif_rs) > 0:
    r_c = max(bif_rs)
else:
    r_c = None

if r_c is not None:
    roots_rc, stab_rc = find_roots_for_r(r_c)
    if len(roots_rc) > 0:
        derivs = np.abs(dFdx(roots_rc, r_c))
        x_c = roots_rc[np.argmin(derivs)]
        print(f"rc = {r_c: .3f} and x* = {x_c: .3f}")
plt.xlabel('r (dimensionless)')
plt.ylabel('x* (dimensionless temperature)')
plt.title('Bifurcation diagram: fixed points x* vs r')
plt.legend(frameon=False)
plt.tight_layout()

#png_path = '/mnt/data/bifurcation_climate.png'
#pdf_path = '/mnt/data/bifurcation_climate.pdf'
#plt.savefig(png_path)
#plt.savefig(pdf_path)
plt.show()

#png_path, pdf_path, "num bifurcation intervals detected", len(bif_rs)
