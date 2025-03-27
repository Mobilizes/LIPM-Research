import numpy as np
import matplotlib.pyplot as plt

t_dbl = 0.3
t = np.linspace(0, t_dbl, 37)

x0 = 102.2510873860025163016147885
v0 = 9.519918775276269747268638657
a0 = 33.93199626676337224502524489
xt = 106.8102184494792936207048545
vt = 9.76078870859097705615563666

c0 = v0
c1 = a0
c2 = (12 * (xt - x0) - 3 * vt * t_dbl - 9 * v0 * t_dbl - 3 * a0 * t_dbl**2) / t_dbl**3
c3 = (vt - (c0 + c1 * t_dbl + c2 * t_dbl**2)) / t_dbl**3

# c0 = 10.09434908950435686316820855
# c1 = 29.30484939134779159756484123
# c2 = -12992.36438914566198372879550
# c3 = 126993.1589523218406775314708
#
# print(c0)
# print(c1)
# print(c2)
# print(c3)

vel = c3 * t**3 + c2 * t**2 + c1 * t + c0
pos = c3 * t**4 / 4 + c2 * t**3 / 3 + c1 * t**2 / 2 + c0 * t + x0
acc = 3 * c3 * t**2 + 2 * c2 * t + c1

plt.plot(t, vel)
plt.plot(t, pos)
plt.show()
