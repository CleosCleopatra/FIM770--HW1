import matplotlib.pyplot as plt
import numpy as np

rs = np.linspace(-4,4, 400)
hs = np.linspace(-4,4,400)
R, H = np.meshgrid(rs,hs)
D = R**2 + 4*H

pos = D >= 0

#fixed points
x_plus = np.where(pos, (R + np.sqrt(D))/2, np.nan)
x_minus = np.where(pos, (R - np.sqrt(D))/2, np.nan)





fig = plt.figure()
ax = plt.axes(projection='3d')
#cs = ax.contourf(R,H,Z, levels = [-0.5, 0.5, 1.5, 2.5])
ax.plot_surface(R, H, x_plus, color='red')
ax.plot_surface(R, H, x_minus)

ax.set_xlabel("r")
ax.set_ylabel("h")
ax.set_zlabel("x*")
ax.set_title("Fixed point surface")

plt.show()