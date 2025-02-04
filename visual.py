import math
from decimal import Decimal as D

import matplotlib.pyplot as plt
import numpy as np

from LIPM_3D import LIPM3D


class LIPM3D_Visual:
    def __init__(self, lipm: LIPM3D, nrow=2, ncol=1):
        self.lipm = lipm
        self.nrow = nrow
        self.ncol = ncol

    def project_walk_pattern(self, x_t_history, index):
        lipm = self.lipm

        fig = plt.figure("LIPM")

        ax = fig.add_subplot(self.nrow, self.ncol, index)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        (mod_p,) = ax.plot(lipm.mod_p_x, lipm.mod_p_y, "o", color="gray")
        (left_foot_pos,) = ax.plot(
            lipm.left_foot_pos[0],
            lipm.left_foot_pos[1],
            "o",
            color="blue" if lipm.support_leg == "left" else "black"
        )
        (right_foot_pos,) = ax.plot(
            lipm.right_foot_pos[0],
            lipm.right_foot_pos[1],
            "o",
            color="blue" if lipm.support_leg == "right" else "black"
        )
        (com_init,) = ax.plot(
            lipm.x_i,
            lipm.y_i,
            "o",
            color="brown",
            label=str(float(lipm.t))
        )
        x_t_history.append((lipm.x_t, lipm.y_t))
        # for (x_t, y_t) in x_t_history:
        #     bx.plot(
        #         x_t,
        #         y_t,
        #         "o",
        #         color="red",
        #     )
        # print((lipm.mod_p_y - lipm.start_swing_foot[1]))
        (com_pos,) = ax.plot(
            lipm.x_t,
            lipm.y_t,
            "o",
            color="red",
            label=str(float(lipm.t))
        )

        dist = math.sqrt(float(lipm.right_foot_pos[0]-lipm.left_foot_pos[0])**2 + float(lipm.right_foot_pos[1] - lipm.left_foot_pos[1])**2)
        print(dist)

        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.5, 0.5)

        plt.legend()
        plt.show()
