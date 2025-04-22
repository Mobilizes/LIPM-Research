from walk import Walk

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

def main():
    model = Walk()

    com_x = []
    com_y = []
    zmp_x = []
    zmp_y = []

    left_foot_x = []
    left_foot_y = []
    right_foot_x = []
    right_foot_y = []
    p_ref_x = []
    p_ref_y = []

    max_data = 100
    while len(left_foot_x) <= max_data:
        model.step()

        left_foot_x.append(model.left_foot[0][0])
        left_foot_y.append(model.left_foot[0][1])
        right_foot_x.append(model.right_foot[0][0])
        right_foot_y.append(model.right_foot[0][1])

        p_ref_x.append(model.p_ref_step[0][0])
        p_ref_y.append(model.p_ref_step[0][1])

        com_x.append(model.com[0][0][0])
        com_y.append(model.com[1][0][0])

        zmp_x.append(model.zmp[0][0])
        zmp_y.append(model.zmp[1][0])

        if model.initial:
            continue

        # print(f"Left Foot: {model.left_foot[0]}")
        # print(f"Right Foot: {model.right_foot[0]}")
        # print(f"Support State: {model.support_step[0]}")
        # print()
        #
        # plt.plot(model.p_ref_step[0][0], model.p_ref_step[0][1], 'go')
        # plt.plot(model.left_foot[0][0], model.left_foot[0][1], 'bo')
        # plt.plot(model.right_foot[0][0], model.right_foot[0][1], 'ro')
        #
        # plt.show()


    fig, ax = plt.subplots()

    p_ref_plot, = ax.plot(p_ref_x, p_ref_y, 'go', label='Reference Point')
    left_foot_plot, = ax.plot(left_foot_x, left_foot_y, 'bo', label='Left Foot')
    right_foot_plot, = ax.plot(right_foot_x, right_foot_y, 'ro', label='Right Foot')
    com_plot, = ax.plot(com_x, com_y, 'p', label='CoM', color='purple')
    zmp_plot, = ax.plot(zmp_x, zmp_y, 'p', label='ZMP', color='cyan')
    ax.legend()

    def update(i):
        p_ref_plot.set_data(p_ref_x[i], p_ref_y[i])
        left_foot_plot.set_data(left_foot_x[i], left_foot_y[i])
        right_foot_plot.set_data(right_foot_x[i], right_foot_y[i])
        com_plot.set_data(com_x[i], com_y[i])
        zmp_plot.set_data(zmp_x[i], zmp_y[i])

        return left_foot_plot, right_foot_plot, p_ref_plot, com_plot, zmp_plot

    ani = animation.FuncAnimation(fig, update, frames=max_data, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
