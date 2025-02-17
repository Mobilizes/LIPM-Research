import copy
import math
import itertools

from decimal import Decimal as D

import matplotlib.pyplot as plt
import numpy as np

from LIPM_3D import LIPM3D
from visual import LIPM3D_Visual


def main():
    left_foot = np.array([D('0.5'), D('-0.1'), D('0.0')])
    right_foot = np.array([D('0.5'), D('0.1'), D('0.0')])
    t_sup = D('0.5')
    z_c = D('0.8')
    x_speed = D('0.1')
    y_speed = D('0.0')
    a_speed = D('0.0')
    a = D('1.0')
    b = D('1.0')

    lipm = LIPM3D(
        left_foot,
        right_foot,
        t_sup=t_sup,
        z_c=z_c,
        y_offset=D('0.0'),
        support_leg="left",
        a=a,
        b=b)
    vis = LIPM3D_Visual(lipm)

    lipm.x_speed = x_speed
    lipm.y_speed = y_speed
    lipm.a_speed = a_speed

    lipm_history = []
    dt = D('0.1')
    max_steps = 65
    while lipm.n < max_steps:
        lipm.step(dt)
        lipm_history.append(copy.deepcopy(lipm))

    for i in lipm_history:
        vis.lipm = i
        vis.com_history.append(
            ((i.x_t, i.y_t), (i.vx_t, i.vy_t), (i.ax_t, i.ay_t)))
        vis.project_3d_gait(1)
        vis.visualize_com_vel_trajectory(2, dt)
        vis.project_walk_pattern(3)
        vis.visualize_com_acc_trajectory(4, dt)
        plt.show()
        print(f"{i.get_distance_between_legs()}, {i.s_theta_1}, {i.s_theta_1 / D(np.pi) * D("180.0")}")


if __name__ == "__main__":
    main()
