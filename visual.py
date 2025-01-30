import matplotlib.pyplot as plt

from LIPM_3D import LIPM3D


class LIPM3D_Visual:
    def __init__(self, lipm: LIPM3D):
        self.lipm = lipm

    def project_walk_pattern(self):
        lipm = self.lipm

        fig = plt.figure()
        ax = fig.add_subplot(211, projection="3d")
        (test,) = ax.plot(0.0, 0.0, 0.0, "o")

        bx = fig.add_subplot(212)

        bx.set_xlabel("x (m)")
        bx.set_ylabel("y (m)")

        (mod_p,) = bx.plot(lipm.mod_p_x, lipm.mod_p_y, "o", color="gray")
        (left_foot_pos,) = bx.plot(
            lipm.left_foot_pos[0],
            lipm.left_foot_pos[1],
            "o",
            color="blue" if lipm.support_leg == "left" else "black"
        )
        (right_foot_pos,) = bx.plot(
            lipm.right_foot_pos[0],
            lipm.right_foot_pos[1],
            "o",
            color="blue" if lipm.support_leg == "right" else "black"
        )
        (com_init,) = bx.plot(
            lipm.x_i,
            lipm.y_i,
            "o",
            color="brown",
        )
        (com_pos,) = bx.plot(
            lipm.x_t,
            lipm.y_t,
            "o",
            color="red",
            label=str(float(lipm.t))
        )

        bx.set_xlim(-0.5, 2.0)
        bx.set_ylim(-0.5, 0.5)

        plt.legend()
        plt.show()
