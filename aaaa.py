import matplotlib.pyplot as plt
import numpy as np

T = .1
h = .8
g = 9.81

A = np.array([[1., T, T**2 / 2.0], [0., 1., T], [0., 0., 1.]])
b = np.array([T**3 / 6., T**2 / 2., T]).reshape(3, 1)
c = np.array([1., 0., h / g])

step = 20

for i in range(1, 2):
    x_k = np.array([0., 0., 0.]).reshape(3, 1)
    X = []
    Zx = []
    u = i * 0.1
    for _ in range(step):
        x_k = A @ x_k + b * u
        X.append(x_k[0].item())
        Zx.append((c @ x_k).item())

    # plt.plot(np.linspace(0, int(round(T * step)), step), X, 'o')
    # plt.plot(np.linspace(0, int(round(T * step)), step), Zx, '-')

for i in range(1, 2):
    y_k = np.array([0., 0., 0.]).reshape(3, 1)
    Y = []
    Zy = []
    u = i * 0.1
    for _ in range(step):
        y_k = A @ y_k + b * u
        Y.append(y_k[0].item())
        Zy.append((c @ y_k).item())

    # plt.plot(np.linspace(0, int(round(T * step)), step), Y, 'o')
    # plt.plot(np.linspace(0, int(round(T * step)), step), Zy, '-')

plt.plot(X, Y, "o")
plt.show()
