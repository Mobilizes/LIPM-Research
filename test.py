import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import qpsolvers

dt = 0.2
h = 0.5
g = 9.81
w_j = 0.1
x_k = 1.0
y_k = 1.0

X0 = np.array([0.5, 0.2, 0.1])[np.newaxis].T
Y0 = np.array([0.5, 0.2, 0.1])[np.newaxis].T

preview_steps = 50

X = [X0]
Y = [Y0]
ux = []
uy = []

for i in range(preview_steps - 1):
    Cx = X[-1][0] + dt * X[-1][1] + dt**2 / 2.0 * X[-1][2] - h/g * X[-1][2] - x_k
    Cy = Y[-1][0] + dt * Y[-1][1] + dt**2 / 2.0 * Y[-1][2] - h/g * Y[-1][2] - y_k
    D = dt**3 / 6.0 - h/g * dt

    jerk_penalty = 0.1

    Px = np.array([[D**2 + jerk_penalty]])
    qx = np.array([2 * Cx * D])
    ux.append(qpsolvers.solve_qp(Px, qx, solver="quadprog"))
    print(ux[-1][0], end=', ')

    Py = np.array([[D**2 + jerk_penalty]])
    qy = np.array([2 * Cy * D])
    uy.append(qpsolvers.solve_qp(Py, qy, solver="quadprog"))
    print(uy[-1][0])

    A = np.array([[1.0, dt, dt**2 / 2.0], [0.0, 1.0, dt], [0.0, 0.0, dt]])
    b = np.array([dt**3 / 6.0, dt**2 / 2.0, dt])[np.newaxis].T

    X.append(A @ X[-1] + b * ux[-1][0])
    Y.append(A @ Y[-1] + b * uy[-1][0])

x_data = [state[0, 0] for state in X]
y_data = [state[0, 0] for state in Y]

fig, ax = plt.subplots()
line, = ax.plot(x_data, y_data, 'o', label="Trajectory")

ax.grid(True)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(x_data[:i+1], y_data[:i+1])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(X),
                              init_func=init, interval=100, blit=True)

plt.show()
