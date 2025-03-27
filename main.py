import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from decimal import Decimal as D

from LIPM_3D import LIPM3D
from visual import LIPM3D_Visual


def main():
    left_foot = np.array([D('100'), D('1.0'), D('0.0')])
    right_foot = np.array([D('100'), D('-1.0'), D('0.0')])
    t_sup = D('0.7')
    t_dbl = D('0.0')
    z_c = D('0.8')
    x_speed = D('5.0')
    y_speed = D('0.0')
    a_speed = D('0.0')
    a = D('1')
    b = D('10')

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

    dt = D('0.025')
    # dt = D("0.1")
    # dt = D("0.008")

    # while True:
    #     vis.update_lipm(dt)
    #     vis.project_3d_gait(1)
    #     vis.visualize_com_vel_trajectory(2, dt)
    #     vis.project_walk_pattern(3)
    #     vis.visualize_com_acc_trajectory(4, dt)
    #     plt.show()

    def update(frame):
        plt.figure("LIPM").clf()

        vis.lipm.x_speed = x_speed
        vis.lipm.y_speed = y_speed
        vis.lipm.a_speed = a_speed

        vis.update_lipm(dt)
        vis.project_3d_gait(1)
        vis.visualize_com_vel_trajectory(2, dt)
        vis.project_walk_pattern(3)
        vis.visualize_com_acc_trajectory(4, dt)

        # system("clear")
        # vis.lipm.print_info()

    ani = FuncAnimation(plt.figure("LIPM"), update, frames=range(100), repeat=True)
    plt.show()


if __name__ == "__main__":
    main()
