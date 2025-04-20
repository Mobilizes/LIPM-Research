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

        p_ref_x.append(model.p_ref[0][0])
        p_ref_y.append(model.p_ref[0][1])

        # print(f"Start Swing: {model.start_swing_foot}")
        print(f"Left Foot: {model.left_foot}")
        print(f"Right Foot: {model.right_foot}")
        print(f"Support State: {model.support_state}")
        # print(f"State: {model.state}")
        # print(f"Phase time: {model.t_phase}")

        plt.plot(model.left_foot[0][0], model.left_foot[0][1], 'bo')
        plt.plot(model.right_foot[0][0], model.right_foot[0][1], 'ro')
        plt.plot(model.p_ref[0][0], model.p_ref[0][1], 'go')

        plt.show()


    # fig, ax = plt.subplots()
    #
    # left_foot_plot, = ax.plot(left_foot_x, left_foot_y, 'bo', label='Left Foot')
    # right_foot_plot, = ax.plot(right_foot_x, right_foot_y, 'ro', label='Right Foot')
    # p_ref_plot, = ax.plot(p_ref_x, p_ref_y, 'go', label='Reference Point')
    # ax.legend()
    #
    # def update(i):
    #     left_foot_plot.set_data(left_foot_x[i], left_foot_y[i])
    #     right_foot_plot.set_data(right_foot_x[i], right_foot_y[i])
    #     p_ref_plot.set_data(p_ref_x[i], p_ref_y[i])
    #
    #     return left_foot_plot, right_foot_plot, p_ref_plot
    #
    # ani = animation.FuncAnimation(fig, update, frames=max_data, blit=True)
    # plt.show()

if __name__ == "__main__":
    main()
