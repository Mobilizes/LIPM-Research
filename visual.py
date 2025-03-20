import warnings

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from decimal import Decimal as D

from LIPM_3D import LIPM3D


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, colors=c, **default_kwargs)
    # lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def colored_line_3d(x, y, z, c, ax, **lc_kwargs):
    """
    Plot a 3D line with color specified along the line by a fourth value.

    Parameters
    ----------
    x, y, z : array-like
        The 3D coordinates of the data points
    c : array-like
        The color values, same size as x, y, z
    ax : Axes3D
        3D axis object for plotting
    **lc_kwargs
        Additional arguments for Line3DCollection

    Returns
    -------
    Line3DCollection
        The 3D line collection object
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Verify we have a 3D axis
    if not hasattr(ax, 'zaxis'):
        raise ValueError("ax must be a 3D axis")

    # Default parameters
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Calculate midpoints for all three dimensions
    x_mid = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_mid = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
    z_mid = np.hstack((z[0], 0.5 * (z[1:] + z[:-1]), z[-1]))

    # Create 3D segments
    coord_start = np.column_stack((x_mid[:-1], y_mid[:-1], z_mid[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y, z))[:, np.newaxis, :]
    coord_end = np.column_stack((x_mid[1:], y_mid[1:], z_mid[1:]))[:, np.newaxis, :]
    
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    # Create 3D line collection
    lc = Line3DCollection(segments, colors=c, **default_kwargs)
    
    # Add to axis and set color mapping
    ax.add_collection(lc)
    return lc

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

        min_x = float(min(lipm.get_support_leg()[0], lipm.get_swing_leg()[0], lipm.x_t[0]))
        max_x = float(max(lipm.get_support_leg()[0], lipm.get_swing_leg()[0], lipm.x_t[0]))
        min_x -= min_x * 0.3
        max_x += max_x * 0.3

        min_y = float(min(lipm.get_support_leg()[1], lipm.get_swing_leg()[1], lipm.y_t[0]))
        max_y = float(max(lipm.get_support_leg()[1], lipm.get_swing_leg()[1], lipm.y_t[0]))
        min_y += min_y * 0.3
        max_y += max_y * 0.3

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # if lipm.support_leg == "right":
        #     ax.set_xlim(float(lipm.get_support_leg()[0] - lipm.x_speed), float(lipm.get_swing_leg()[0] + lipm.x_speed))
        #     ax.set_ylim(float(lipm.get_support_leg()[1] - lipm.y_speed), float(lipm.get_swing_leg()[1] + lipm.y_speed))
        # else:
        #     ax.set_xlim(float(lipm.get_swing_leg()[0] - lipm.x_speed), float(lipm.get_support_leg()[0] + lipm.x_speed))
        #     ax.set_ylim(float(lipm.get_swing_leg()[1] - lipm.y_speed), float(lipm.get_support_leg()[1] + lipm.y_speed))

        (left_leg,) = ax.plot(
            [float(lipm.x_t[0]), float(lipm.left_foot_pos[0])],
            [float(lipm.y_t[0]), float(lipm.left_foot_pos[1])],
            zs=[float(lipm.z_c), float(lipm.left_foot_pos[2])],
            color="cyan",
            linewidth=2.0
        )

        (right_leg,) = ax.plot(
            [float(lipm.x_t[0]), float(lipm.right_foot_pos[0])],
            [float(lipm.y_t[0]), float(lipm.right_foot_pos[1])],
            zs=[float(lipm.z_c), float(lipm.right_foot_pos[2])],
            color="blue",
            linewidth=2.0
        )

        (com,) = ax.plot(
            float(lipm.x_t[0]),
            float(lipm.y_t[0]),
            "o",
            color="red",
            zs=float(lipm.z_c),
        )

        com_traj = colored_line_3d(
            [float(row[0][0]) for row in com_history],
            [float(row[0][1]) for row in com_history],
            [0.0] * len(com_history),
            ["red" if row[-1] else "cyan" for row in com_history],
            ax
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
            lipm.x_t[0],
            lipm.y_t[0],
            "o",
            color="red"
        )

        com_traj = colored_line(
            [float(row[0][0]) for row in com_history],
            [float(row[0][1]) for row in com_history],
            ["red" if row[-1] else "cyan" for row in com_history],
            ax
        )

        (mod_p,) = ax.plot(lipm.mod_p_x[0], lipm.mod_p_y[0], "o", color="gray")
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
        # x_min = min(lipm.x_t[0], lipm.left_foot_pos[0], lipm.right_foot_pos[0])
        # x_max = max(lipm.x_t[0], lipm.left_foot_pos[0], lipm.right_foot_pos[0])
        #
        # y_min = min(lipm.y_t[0], lipm.left_foot_pos[1], lipm.right_foot_pos[1])
        # y_max = max(lipm.y_t[0], lipm.left_foot_pos[1], lipm.right_foot_pos[1])
        #
        # ax.set_xlim(x_min - cons, x_max + cons)
        # ax.set_xlim(y_min - cons, y_max + cons)

        plt.legend(
            [com_pos, mod_p], ["t : " + str(float(lipm.t)), "t_s : " + str(float(lipm.t_s))]
        )

    def update_lipm(self, dt):
        self.lipm.step(dt)

        lipm = self.lipm
        self.com_history.append(
            (
                (lipm.x_t[0], lipm.y_t[0]),
                (lipm.vx_t[0], lipm.vy_t[0]),
                (lipm.ax_t[0], lipm.ay_t[0]),
                lipm.support_leg != "both",
            )
        )
