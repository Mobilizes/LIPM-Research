import matplotlib.pyplot as plt
import numpy as np

from LIPM_3D import LIPM3D


class LIPM3D_Visual:
    def __init__(self, lipm: LIPM3D, nrow=2, ncol=1):
        self.com_history = []
        self.lipm = lipm
        self.nrow = nrow
        self.ncol = ncol

    def project_3d_gait(self, index):
        lipm = self.lipm
        com_history = self.com_history

        fig = plt.figure("LIPM")

        ax = fig.add_subplot(self.nrow, self.ncol, index, projection="3d")

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)

        (left_leg,) = ax.plot(
            [float(lipm.x_t), float(lipm.left_foot_pos[0])],
            [float(lipm.y_t), float(lipm.left_foot_pos[1])],
            zs=[float(lipm.z_c), float(lipm.left_foot_pos[2])],
            color="cyan",
            linewidth=2.0
        )

        (right_leg,) = ax.plot(
            [float(lipm.x_t), float(lipm.right_foot_pos[0])],
            [float(lipm.y_t), float(lipm.right_foot_pos[1])],
            zs=[float(lipm.z_c), float(lipm.right_foot_pos[2])],
            color="blue",
            linewidth=2.0
        )

        (com,) = ax.plot(
            float(lipm.x_t),
            float(lipm.y_t),
            "o",
            color="red",
            zs=float(lipm.z_c),
        )

        (com_traj,) = ax.plot(
            [float(row[0]) for row in com_history],
            [float(row[1]) for row in com_history],
            zs=0,
            color="red"
        )

    def project_walk_pattern(self, index):
        lipm = self.lipm
        com_history = self.com_history

        fig = plt.figure("LIPM")

        ax = fig.add_subplot(self.nrow, self.ncol, index)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # com_history.append((lipm.x_t, lipm.y_t))
        # for (x_t, y_t) in x_t_history:
        #     ax.plot(
        #         x_t,
        #         y_t,
        #         "o",
        #         color="red",
        #     )

        (com_init,) = ax.plot(
            lipm.x_i,
            lipm.y_i,
            "o",
            color="brown",
            label=str(float(lipm.t))
        )
        (com_pos,) = ax.plot(
            lipm.x_t,
            lipm.y_t,
            "o",
            color="red",
            label=str(float(lipm.t))
        )
        (com_traj,) = ax.plot(
            [float(row[0]) for row in com_history],
            [float(row[1]) for row in com_history],
            color="red"
        )

        (mod_p,) = ax.plot(lipm.mod_p_x, lipm.mod_p_y, "o", color="gray")
        (left_foot_pos,) = ax.plot(
            lipm.left_foot_pos[0],
            lipm.left_foot_pos[1],
            "o",
            color="aqua" if lipm.support_leg == "left" else "black"
        )
        (right_foot_pos,) = ax.plot(
            lipm.right_foot_pos[0],
            lipm.right_foot_pos[1],
            "o",
            color="blue" if lipm.support_leg == "right" else "black"
        )
        # print((lipm.mod_p_y - lipm.start_swing_foot[1]))

        # dist = math.sqrt(float(lipm.right_foot_pos[0]-lipm.left_foot_pos[0])**2 + float(lipm.right_foot_pos[1] - lipm.left_foot_pos[1])**2)
        # print(f"{dist}, {lipm.s_theta_1}, {lipm.s_theta_1 / D(np.pi) * D("180.0")}")

        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.5, 0.5)

        plt.legend()
