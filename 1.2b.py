import matplotlib.pyplot as plt
import numpy as np

rs = np.linspace(-4,4, 400)
hs = np.linspace(-4,4,400)

def calc(r,h):
    sqr = np.sqrt(r**2 + 4*h)
    c11 = r + sqr
    c12 = r - sqr
    return c11/2, c12/2

r_list, h_list = [], []
x_list=[]
for r in rs:
    for h in hs:
        x1, x2 = calc(r,h)
        x_list.append(x1)
        x_list.append(x2)
        r_list.append(r)
        r_list.append(r)
        h_list.append(h)
        h_list.append(h)



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(h_list, r_list, x_list)
plt.show()