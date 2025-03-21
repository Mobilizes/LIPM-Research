import numpy as np
import matplotlib.pyplot as plt

t_dbl = 0.1
t = np.linspace(0, t_dbl, 100)

x0 = 1
v0 = 2
a0 = 1
xt = 1
vt = -2

c0 = v0
c1 = a0
c2 = (12 * (xt - x0) - 3 * vt * t_dbl - 9 * v0 * t_dbl - 3 * a0 * t_dbl**2) / t_dbl**3
c3 = (vt - (c0 + c1 * t_dbl + c2 * t_dbl**2)) / t_dbl**3

vel = c3 * t**3 + c2 * t**2 + c1 * t + c0
pos = c3 * t**4 / 4 + c2 * t**3 / 3 + c1 * t**2 / 2 + c0 * t + x0
acc = 3 * c3 * t**2 + 2 * c2 * t + c1

plt.plot(t, vel)
plt.plot(t, pos)
plt.show()
