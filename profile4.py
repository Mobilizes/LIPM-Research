import numpy as np
import matplotlib.pyplot as plt

# Parameters
t_dbl = 0.1  # Duration of the double support phase
t = np.linspace(0, t_dbl, 100)

# Boundary conditions (initial and final)
x0 = 102.2510873860025163016147885
v0 = 9.519918775276269747268638657
a0 = 0.0
xt = 106.8102184494792936207048545
vt = 9.76078870859097705615563666
at = 0.0

# Boundary conditions for quintic polynomial (position, velocity, acceleration)
A = np.array([
    [1, 0, 0, 0, 0, 0],              # x(0) = x0
    [0, 1, 0, 0, 0, 0],              # v(0) = v0
    [0, 0, 2, 0, 0, 0],              # a(0) = a0
    [1, t_dbl, t_dbl**2, t_dbl**3, t_dbl**4, t_dbl**5],  # x(t_dbl) = xt
    [0, 1, 2*t_dbl, 3*t_dbl**2, 4*t_dbl**3, 5*t_dbl**4], # v(t_dbl) = vt
    [0, 0, 2, 6*t_dbl, 12*t_dbl**2, 20*t_dbl**3]         # a(t_dbl) = at
])

b = np.array([x0, v0, a0, xt, vt, at])
coefficients = np.linalg.solve(A, b)  # Solve for coefficients

# Position, velocity, acceleration
position = np.polyval(coefficients[::-1], t)
velocity = np.polyval(np.polyder(coefficients[::-1], 1), t)
acceleration = np.polyval(np.polyder(coefficients[::-1], 2), t)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, velocity)
plt.ylabel('Velocity')
plt.title('Flattened Velocity Profile (Quintic Polynomial)')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, acceleration)
plt.ylabel('Acceleration')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, np.gradient(acceleration, t))  # Jerk (da/dt)
plt.xlabel('Time')
plt.ylabel('Jerk')
plt.grid(True)

plt.tight_layout()
plt.show()
