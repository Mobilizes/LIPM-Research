import numpy as np
import matplotlib.pyplot as plt

def quintic_coefficients(t, x0, v0, a0, xT, vT, aT):
    m = np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [t**5, t**4, t**3, t**2, t, 1],
        [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0],
        [20*t**3, 12*t**2, 6*t, 2, 0, 0]
    ])
    bc = np.array([x0, v0, a0, xT, vT, aT])
    coeff = np.linalg.solve(m, bc)
    return coeff

t_dbl = 0.1
px0 = 0.7499999999999999969443173731
vx0 = 1.040580659917997861863933705
ax0 = 3.698428719590751575975803334
pxt = px0 + vx0 * t_dbl
vxt = 1.040580659917997861863933705
axt = -3.698428719590751575975803334

py0 = 0.0629464390079180365856365816
vy0 = 0.6782356521365885656334054542
ay0 = 2.410583159127707231663843249
pyt = py0 + vy0 * t_dbl
vyt = 0.6782356521365885656334054542
ayt = -2.410583159127707231663843249

ax, bx, cx, dx, ex, fx = quintic_coefficients(t_dbl, px0, vx0, ax0, pxt, vxt, axt)
ay, by, cy, dy, ey, fy = quintic_coefficients(t_dbl, py0, vy0, ay0, pyt, vyt, ayt)

t = np.linspace(0, t_dbl, 100)

position_x = ax * t**5 + bx * t**4 + cx * t**3 + dx * t**2 + ex * t + fx
position_y = ay * t**5 + by * t**4 + cy * t**3 + dy * t**2 + ey * t + fy

velocity_x = 5 * ax * t**4 + 4 * bx * t**3 + 3 * cx * t**2 + 2 * dx * t + ex
velocity_y = 5 * ay * t**4 + 4 * by * t**3 + 3 * cy * t**2 + 2 * dy * t + ey

acceleration_x = 20 * ax * t**3 + 12 * bx * t**2 + 6 * cx * t + 2 * dx
acceleration_y = 20 * ay * t**3 + 12 * by * t**2 + 6 * cy * t + 2 * dy

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
