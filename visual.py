import matplotlib.pyplot as plt

from LIPM_3D import LIPM3D


class LIPM3D_Visual:
    def __init__(self, lipm: LIPM3D):
        self.lipm = lipm

    def project_walk_pattern(self):
        lipm = self.lipm

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        (left_foot_pos,) = ax.plot(
            lipm.left_foot_pos[0], lipm.left_foot_pos[1], "o", color="blue"
        )
        (right_foot_pos,) = ax.plot(
            lipm.right_foot_pos[0], lipm.right_foot_pos[1], "o", color="blue"
        )
        (com_pos,) = ax.plot(
            lipm.x_f,
            lipm.y_f,
            "o",
            color="red",
        )

        plt.show()
