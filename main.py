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

    while len(model.p_ref) > 970:
        model.step()

        com_x.append(model.com[0][0])
        com_y.append(model.com[1][0])
        zmp_x.append(model.zmp[0])
        zmp_y.append(model.zmp[1])

        left_foot_x.append(model.left_foot[0])
        left_foot_y.append(model.left_foot[1])
        right_foot_x.append(model.right_foot[0])
        right_foot_y.append(model.right_foot[1])
        p_ref_x.append(model.p_ref[0][0])
        p_ref_y.append(model.p_ref[0][1])

        # fig, ax = plt.subplots()
        # com, = ax.plot(com_x[-1], com_y[-1], 'bo', markersize=8, label='Current CoM')
        # line, = ax.plot(com_x, com_y, 'r-')
        # # for p_ref in model.p_ref:
        # #     ax.plot(p_ref[0], p_ref[1], 'bo', markersize=8, color="green")
        # # ref, = ax.plot(model.p_ref[:, 0], model.p_ref[:, 1], 'b', color="red")
        # ax.set_xlabel('X Position [m]')
        # ax.set_ylabel('Y Position [m]')
        # ax.set_title('Center of Mass Trajectory')
        # ax.grid(True)
        # ax.legend()
        #
        # plt.show()

    zmp_x[0] = float(zmp_x[0])
    zmp_y[0] = float(zmp_y[0])

    fig, ax = plt.subplots()
    com, = ax.plot(com_x, com_y, 'bo', markersize=8, label='Current CoM')
    zmp, = ax.plot(zmp_x, zmp_y, 'go', markersize=8, label='ZMP')
    line, = ax.plot([], [], 'r-')
    p_ref_line = ax.plot(p_ref_x, p_ref_y, 'o', markersize=8, label="Reference", color="gray")[0]
    left_foot_line = ax.plot(left_foot_x, left_foot_y, 'o', markersize=8, label='Left Foot', color="red")[0]
    right_foot_line = ax.plot(right_foot_x, right_foot_y, 'o', markersize=8, label='Right Foot', color="maroon")[0]
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_title('Center of Mass Trajectory')
    ax.grid(True)
    ax.legend()

    def animate(i):
        com.set_data(com_x[i], com_y[i])
        zmp.set_data(zmp_x[i], zmp_y[i])
        line.set_data(com_x[:i+1], com_y[:i+1])
        p_ref_line.set_data(p_ref_x[i], p_ref_y[i])
        left_foot_line.set_data(left_foot_x[i], left_foot_y[i])
        right_foot_line.set_data(right_foot_x[i], right_foot_y[i])

        return com, zmp, line, p_ref_line, left_foot_line, right_foot_line

    ani = animation.FuncAnimation(
        fig, animate,
        frames=len(com_x),
        interval=30,
        blit=True
    )

    plt.show()

if __name__ == "__main__":
    main()
