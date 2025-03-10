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
px0 = 0.1
vx0 = 0.32
ax0 = 0.3
vxt = 0.32
axt = -0.6

py0 = 0.5
vy0 = -0.44
ay0 = -1.93
vyt = 0.44
ayt = 1.93

vxa, vxb, vxc, vxd = compute_velocity_coefficients(vx0, ax0, vxt, axt, t_dbl)
vya, vyb, vyc, vyd = compute_velocity_coefficients(vy0, ay0, vyt, ayt, t_dbl)

t = np.linspace(0, t_dbl, 100)

velocity_x = vxa * t**3 + vxb * t**2 + vxc * t + vxd
velocity_y = vya * t**3 + vyb * t**2 + vyc * t + vyd

position_x = vxa * t**4 / 4 + vxb * t**3 / 3 + vxc * t**2 / 2 + vxd * t + px0
position_y = vya * t**4 / 4 + vyb * t**3 / 3 + vyc * t**2 / 2 + vyd * t + py0

plt.figure(figsize=(10, 6))

# Position plot
plt.subplot(2, 3, 1)
plt.plot(t, velocity_x, color='blue')
plt.ylabel('Velocity (x)')
plt.title('CoM Motion Profile')
plt.grid(True)

# Velocity plot
plt.subplot(2, 3, 2)
plt.plot(t, velocity_y, color='blue')
plt.ylabel('Velocity (y)')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(velocity_x, velocity_y, color='red')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(t, position_x, color='blue')
plt.ylabel('Position (x)')
plt.title('CoM Motion Profile')
plt.grid(True)

# Velocity plot
plt.subplot(2, 3, 5)
plt.plot(t, position_y, color='blue')
plt.ylabel('Position (y)')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(position_x, position_y, color='red')
plt.grid(True)

plt.tight_layout()
plt.show()
