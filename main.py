import copy
import math
import itertools

from decimal import Decimal as D

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from LIPM_3D import LIPM3D
from visual import LIPM3D_Visual


def main():
    left_foot = np.array([D('0.5'), D('-0.1'), D('0.0')])
    right_foot = np.array([D('0.5'), D('0.1'), D('0.0')])
    t_sup = D('0.5')
    t_dbl = D('0.0')
    z_c = D('1.2')
    x_speed = D('0.5')
    y_speed = D('0.0')
    a_speed = D('0.0')
    a = D('0.1')
    b = D('1.0')

    lipm = LIPM3D(
        left_foot,
        right_foot,
        t_sup=t_sup,
        z_c=z_c,
        y_offset=D('0.0'),
        support_leg="left",
        t_dbl=t_dbl,
        a=a,
        b=b)
    vis = LIPM3D_Visual(lipm)

    lipm.x_speed = x_speed
    lipm.y_speed = y_speed
    lipm.a_speed = a_speed

    dt = D('0.05')
    # max_steps = 65
    # while lipm.n < max_steps:
    #     lipm.step(dt)
    #     lipm_history.append(copy.deepcopy(lipm))

    # while True:
    #     vis.update_lipm(dt)
    #     vis.project_3d_gait(1)
    #     vis.visualize_com_vel_trajectory(2, dt)
    #     vis.project_walk_pattern(3)
    #     vis.visualize_com_acc_trajectory(4, dt)
    #     plt.show()

    def update(frame):
        plt.figure("LIPM").clf()

        vis.update_lipm(dt)
        vis.project_3d_gait(1)
        vis.visualize_com_vel_trajectory(2, dt)
        vis.project_walk_pattern(3)
        vis.visualize_com_acc_trajectory(4, dt)

    ani = FuncAnimation(plt.figure("LIPM"), update, frames=range(100), repeat=True)
    plt.show()


if __name__ == "__main__":
    main()
