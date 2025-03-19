import numpy as np
import matplotlib.pyplot as plt


def compute_velocity_coefficients(v0, a0, vt, at, T):
    delta_v = vt - v0

    a = (-2 * delta_v + T * (a0 + at)) / T**3
    b = (3 * delta_v - T * (2*a0 + at)) / T**2
    c = a0
    d = v0

    return (a, b, c, d)


t_dbl = 0.1
px0 = 0.7499999999999999969443173731
vx0 = 1.040580659917997861863933705
ax0 = 3.698428719590751575975803334
vxt = 1.040580659917997861863933705
axt = -3.698428719590751575975803334

py0 = 0.0629464390079180365856365816
vy0 = 0.6782356521365885656334054542
ay0 = 2.410583159127707231663843249
vyt = 0.6782356521365885656334054542
ayt = -2.410583159127707231663843249

vxa, vxb, vxc, vxd = compute_velocity_coefficients(vx0, ax0, vxt, axt, t_dbl)
vya, vyb, vyc, vyd = compute_velocity_coefficients(vy0, ay0, vyt, ayt, t_dbl)

t = np.linspace(0, t_dbl, 100)

velocity_x = vxa * t**3 + vxb * t**2 + vxc * t + vxd
velocity_y = vya * t**3 + vyb * t**2 + vyc * t + vyd

position_x = vxa * t**4 / 4 + vxb * t**3 / 3 + vxc * t**2 / 2 + vxd * t + px0
position_y = vya * t**4 / 4 + vyb * t**3 / 3 + vyc * t**2 / 2 + vyd * t + py0

acceleration_x = 3 * vxa * t**2 + 2 * vxb * t + vxc
acceleration_y = 3 * vya * t**2 + 2 * vyb * t + vyc

plt.figure(figsize=(10, 6))

plt.subplot(3, 3, 1)
plt.plot(t, position_x, color='blue')
plt.ylabel('Position (x)')
plt.title('CoM Motion Profile')
plt.grid(True)

plt.subplot(3, 3, 2)
plt.plot(t, position_y, color='blue')
plt.ylabel('Position (y)')
plt.grid(True)

plt.subplot(3, 3, 3)
plt.plot(position_x, position_y, color='red')
plt.grid(True)

plt.subplot(3, 3, 4)
plt.plot(t, velocity_x, color='blue')
plt.ylabel('Velocity (x)')
plt.title('CoM Motion Profile')
plt.grid(True)

plt.subplot(3, 3, 5)
plt.plot(t, velocity_y, color='blue')
plt.ylabel('Velocity (y)')
plt.grid(True)

plt.subplot(3, 3, 6)
plt.plot(velocity_x, velocity_y, color='red')
plt.grid(True)

plt.subplot(3, 3, 7)
plt.plot(t, acceleration_x, color='blue')
plt.ylabel('Acceleration (x)')
plt.title('CoM Motion Profile')
plt.grid(True)

plt.subplot(3, 3, 8)
plt.plot(t, acceleration_y, color='blue')
plt.ylabel('Acceleration (y)')
plt.grid(True)

plt.subplot(3, 3, 9)
plt.plot(acceleration_x, velocity_y, color='red')
plt.grid(True)

plt.tight_layout()
plt.show()
