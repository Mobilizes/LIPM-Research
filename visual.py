import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from decimal import Decimal as D

from LIPM_3D import LIPM3D


class LIPM3D_Visual:
    def __init__(self, lipm: LIPM3D, nrow=2, ncol=2):
        self.com_history = []
        self.lipm = lipm
        self.nrow = nrow
        self.ncol = ncol

        self.curr_azim = -60
        self.curr_elev = 30

    def project_3d_gait(self, index):
        lipm = self.lipm
        com_history = self.com_history

        fig = plt.figure("LIPM")

        ax = fig.add_subplot(self.nrow, self.ncol, index, projection="3d")

        def on_release(event):
            """Print previous and new angles after rotation."""
            self.curr_azim, self.curr_elev = ax.azim, ax.elev

        ax.azim = self.curr_azim
        ax.elev = self.curr_elev

        fig.canvas.mpl_connect('button_release_event', on_release)

        if lipm.support_leg == "left":
            ax.set_xlim(float(lipm.get_support_leg()[0] - lipm.x_speed), float(lipm.get_swing_leg()[0] + lipm.x_speed))
            ax.set_ylim(float(lipm.get_support_leg()[1] - lipm.y_speed), float(lipm.get_swing_leg()[1] + lipm.y_speed))
        else:
            ax.set_xlim(float(lipm.get_swing_leg()[0] - lipm.x_speed), float(lipm.get_support_leg()[0] + lipm.x_speed))
            ax.set_ylim(float(lipm.get_swing_leg()[1] - lipm.y_speed), float(lipm.get_support_leg()[1] + lipm.y_speed))

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
            [float(row[0][0]) for row in com_history],
            [float(row[0][1]) for row in com_history],
            zs=0.0,
            color="red"
        )

    def visualize_com_vel_trajectory(self, index, dt):
        com_history = self.com_history

        fig = plt.figure("LIPM")

        ax = fig.add_subplot(self.nrow, self.ncol, index)

        ax.set_title("Velocity")

        t = np.arange(stop=dt*len(com_history), step=dt)
        x = [float(row[1][0]) for row in com_history]
        y = [float(row[1][1]) for row in com_history]

        (vel_traj_x,) = ax.plot(
            t,
            x,
            label="x"
        )

        (vel_traj_y,) = ax.plot(
            t,
            y,
            label="y"
        )

        cons = 2
        left_lim = max(0, t[-1] - cons)
        ax.set_xlim(left_lim, left_lim + 2*cons)

        plt.legend()

    def visualize_com_acc_trajectory(self, index, dt):
        com_history = self.com_history

        fig = plt.figure("LIPM")

        ax = fig.add_subplot(self.nrow, self.ncol, index)

        ax.set_title("Acceleration")

        t = np.arange(stop=dt*len(com_history), step=dt)
        x = [float(row[2][0]) for row in com_history]
        y = [float(row[2][1]) for row in com_history]

        (acc_traj_x,) = ax.plot(
            t,
            x,
            label="x"
        )

        (acc_traj_y,) = ax.plot(
            t,
            y,
            label="y"
        )

        cons = 2
        left_lim = max(0, t[-1] - cons)
        ax.set_xlim(left_lim, left_lim + 2*cons)

        plt.legend()

    def project_walk_pattern(self, index):
        lipm = self.lipm
        com_history = self.com_history

        fig = plt.figure("LIPM")

        ax = fig.add_subplot(self.nrow, self.ncol, index)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        (com_pos,) = ax.plot(
            lipm.x_t,
            lipm.y_t,
            "o",
            color="red",
            label=str(float(lipm.t + lipm.t_s))
        )
        (com_traj,) = ax.plot(
            [float(row[0][0]) for row in com_history],
            [float(row[0][1]) for row in com_history],
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

        # cons = D("0.5")
        #
        # x_min = min(lipm.x_t, lipm.left_foot_pos[0], lipm.right_foot_pos[0])
        # x_max = max(lipm.x_t, lipm.left_foot_pos[0], lipm.right_foot_pos[0])
        #
        # y_min = min(lipm.y_t, lipm.left_foot_pos[1], lipm.right_foot_pos[1])
        # y_max = max(lipm.y_t, lipm.left_foot_pos[1], lipm.right_foot_pos[1])
        #
        # ax.set_xlim(x_min - cons, x_max + cons)
        # ax.set_xlim(y_min - cons, y_max + cons)

        plt.legend()

    def update_lipm(self, dt):
        self.lipm.step(dt)

        lipm = self.lipm
        self.com_history.append(
            ((lipm.x_t, lipm.y_t), (lipm.vx_t, lipm.vy_t), (lipm.ax_t, lipm.ay_t)))
