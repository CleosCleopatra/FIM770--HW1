import numpy as np
from matplotlib import pyplot as plt

#Grid
r_min, r_max = -4, 4 #What should the interval be
h_min, h_max = -4, 4 #int?
nr, nh = 400, 400
r = np.linspace(r_min, r_max, nr)
h = np.linspace(h_min, h_max, nh)
R, H = np.meshgrid(r,h)
D = R**2 + 4*H

#Masks
two_fp = D > 0
one_fp = np.isclose(D, 0, atol = 1e-6)
no_fp = D < 0


fig, ax = plt.subplots(figsize=(8,8))

Z = np.zeros_like(D, dtype = int)
Z[no_fp] = 0
Z[one_fp] = 1
Z[two_fp] = 2
cs = ax.contourf(R, H, Z, levels=[-0.5, 0.5, 1.5, 2.5])

#draw bifurcation curve
r_curve = np.linspace(r_min, r_max, 400)
h_curve = -r_curve ** 2 / 4
ax.plot(r_curve, h_curve, linewidth =2)

#regions
ax.text(0, 2.5, 'Two real fixed points, 1 stable and 1 unstable', ha = 'center', va='center', fontsize = 12)
ax.text(0, -3, 'No real fixed points', ha='center', va='center', fontsize=12)
ax.text(0, -0.2, 'Curve', ha='center', va='center', fontsize=12) #????



ax.set_xlabel('r')
ax.set_ylabel('h')
ax.set_title("Bifurcation diagram ")

ax.set_xlim(r_min, r_max)
ax.set_ylim(h_min, h_max)
ax.grid(True, linestyle=':')

ax.legend(loc='lower left')

fig.tight_layout()
plt.show()


"""




















def fixed_points(r,h):
    sqr = np.sqrt(r**2+4*h)
    c1 = r + sqr
    c2 = r - sqr

    return c1/2 == 0, c2/2==0

fixed_points_list_r = []
fixed_points_list_h = []
rs = np.linspace(-1,1, 401)
hs = np.linspace(-0.6,0.6,401)
R, H = np.meshgrid(rs, hs, indexing='xy')

Delta = R**2 + 4*H
region = np.zeros_like(Delta, dtype=int)
region[Delta < 0] = 0 
region[np.isclose(Delta, 0, atol=1e-9)] = 2
region[Delta > 0] = 1

S = np.zeros_like(Delta)
mask = Delta > 0
S[mask] = np.sqrt(Delta[mask])
x_plus = 0.5 * (R + S)
x_minus = 0.5 *(R- S)

fix, ax = plt.subplots(figsize=(7,6))

cmap = plt.get_cmap('Pastel1')
im = ax.contourf(R, H, region, levels=[-0.5, 0.5, 1.5, 2.5], alpha = 0.7, cmap = cmap)

r_plot = np.linspace(-1, 1, 400)
h_parab = -r_plot**2/4.0
ax.plot(r_plot, h_parab, label= "bigurcation curve")

ax.plot(R[mask], x_plus[mask])
ax.plot(R[mask], x_minus[mask])

ax.axhline(0)
ax.axvline(0)
from matplotlib.patches import Patch
legend_elms = [
    Patch(facecolor=cmap(0), label="no real"),
    Patch(facecolor=cmap(1), edgecolor='k', label="two fixed"),
    Patch(facecolor=cmap(2), label="doubel fixed")
]

ax.legend(handles=legend_elms)

ax.set_xlim(-1, 1)
ax.set_ylim(-0.6, 0.6)
ax.set_xlabel('r')
ax.set_ylabel('h')
ax.set_title('Regions in (h,r): number and type of fixed points\n' +
             'x* = (r ± sqrt(r^2+4h))/2.  Stable: + branch; Unstable: - branch')
plt.tight_layout()

plt.show()

for r in rs:
    for h in hs:
        if fixed_points(r,h):
            fixed_points_list_r.append(r)
            fixed_points_list_h.append(h)
from matplotlib import pyplot as plt
plt.scatter(fixed_points_list_h, fixed_points_list_r)
plt.show()


#png_path, pdf_path, "num bifurcation intervals detected", len(bif_rs)
"""