import numpy as np


class LIPM:
    def __init__(
        self,
        left_foot_pos: np.ndarray,
        right_foot_pos: np.ndarray,
        com_pos: np.ndarray,
        support_leg="left",
        t_sup=1.0,
    ) -> None:
        self.left_foot_pos = left_foot_pos
        self.right_foot_pos = right_foot_pos
        self.com_pos = com_pos
        self.support_leg = support_leg  # left / both / right

        self.t_sup = t_sup

        self.s_x = 0.0
        self.s_y = 0.0

        self.mod_p_x = (left_foot_pos if support_leg == "left" else right_foot_pos)[0]
        self.mod_p_y = (left_foot_pos if support_leg == "left" else right_foot_pos)[1]

        self.p_x = (left_foot_pos if support_leg == "left" else right_foot_pos)[0]
        self.p_y = (left_foot_pos if support_leg == "left" else right_foot_pos)[1]

        self.t = 0.0

        self.z_c = self.com_pos[2]
        self.t_c = np.sqrt(self.z_c / 9.81)
        self.c = np.cosh(self.t_sup / self.t_c)
        self.s = np.sinh(self.t_sup / self.t_c)

        self.walk_x = 0.0
        self.walk_y = 0.0
        self.walk_vx = 0.0
        self.walk_vy = 0.0

        self.x_i = 0.0
        self.vx_i = 0.0
        self.y_i = 0.0
        self.vy_i = 0.0

        self.x_t = 0.0
        self.vx_t = 0.0
        self.y_t = 0.0
        self.vy_t = 0.0

        self.x_d = 0.0
        self.vx_d = 0.0
        self.y_d = 0.0
        self.vy_d = 0.0

    def update_t_sup(self, t_sup) -> None:
        self.t_sup = t_sup
        t_c = self.t_c

        self.c = np.cosh(t_sup / t_c)
        self.s = np.cosh(t_sup / t_c)

    def update_z_c(self, z_c) -> None:
        self.z_c = z_c
        self.t_c = np.sqrt(z_c / 9.81)
        t_c = self.t_c
        t_sup = self.t_sup

        self.c = np.cosh(t_sup / t_c)
        self.s = np.cosh(t_sup / t_c)

    # Step 3
    def calculate_next_com_state(self):
        t, t_c = self.t, self.t_c
        x_i, y_i = self.x_i, self.y_i
        vx_i, vy_i = self.vx_i, self.vy_i
        mod_p_x, mod_p_y = self.mod_p_x, self.mod_p_y

        self.x_t = (
            (x_i - mod_p_x) * np.cosh(t / t_c) + t_c * vx_i * np.sinh(t / t_c) + mod_p_x
        )
        self.vx_t = ((x_i - mod_p_x) / t_c) * np.sinh(t / t_c) + x_i * np.cosh(t / t_c)

        self.y_t = (
            (y_i - mod_p_y) * np.cosh(t / t_c) + t_c * vy_i * np.sinh(t / t_c) + mod_p_y
        )
        self.vy_t = ((y_i - mod_p_y) / t_c) * np.sinh(t / t_c) + y_i * np.cosh(t / t_c)

    # Step 5
    def calculate_new_foot_place(self, s_x, s_y, s_theta) -> None:
        if self.support_leg == "right":
            self.p_x += np.cos(s_theta) * s_x - np.sin(s_theta) * s_y
            self.p_y += np.sin(s_theta) * s_x + np.cos(s_theta) * s_y
        elif self.support_leg == "left":
            self.p_x += np.cos(s_theta) * s_x + np.sin(s_theta) * s_y
            self.p_y += np.sin(s_theta) * s_x - np.cos(s_theta) * s_y

    # Step 6
    def set_new_walk_primitive(self, s_x, s_y, s_theta) -> None:
        """
        Make sure the s parameters are a step ahead
        from new foot place
        """
        c = self.c
        s = self.s
        t_c = self.t_c

        if self.support_leg == "right":
            self.walk_x = np.cos(s_theta) * s_x / 2.0 + np.sin(s_theta) * s_y / 2.0
            self.walk_y = np.sin(s_theta) * s_x / 2.0 - np.cos(s_theta) * s_y / 2.0
        elif self.support_leg == "left":
            self.walk_x = np.cos(s_theta) * s_x / 2.0 - np.sin(s_theta) * s_y / 2.0
            self.walk_y = np.sin(s_theta) * s_x / 2.0 + np.cos(s_theta) * s_y / 2.0

        walk_x = self.walk_x
        walk_y = self.walk_y

        a = (1 + c) / (t_c * s) * walk_x
        b = (c - 1) / (t_c * s) * walk_y

        self.walk_vx = np.cos(s_theta) * a - np.sin(s_theta) * b
        self.walk_vy = np.sin(s_theta) * a + np.cos(s_theta) * b

    # Step 7
    def calculate_target_state(self) -> None:
        p_x, p_y = self.p_x, self.p_y
        walk_x, walk_y = self.walk_x, self.walk_y
        walk_vx, walk_vy = self.walk_vx, self.walk_vy

        self.x_d = p_x + walk_x
        self.vx_d = walk_vx

        self.y_d = p_y + walk_y
        self.vy_d = walk_vy

    # Step 8
    def calculate_modified_foot_placement(self, a, b) -> None:
        c, s = self.c, self.s
        t_c = self.t_c
        x_i, y_i = self.x_i, self.y_i
        vx_i, vy_i = self.vx_i, self.vy_i
        x_d, y_d = self.x_d, self.y_d

        d = a * (c - 1) ** 2 + b * (s / t_c) ** 2

        self.mod_p_x = (-(a * (c - 1) / d) * (x_d - c * x_i - t_c * s * vx_i)) - (
            -((b * s) / (t_c * d)) * (x_d - (s / t_c) * x_i - c * vx_i)
        )
        self.mod_p_y = (-(a * (c - 1) / d) * (y_d - c * y_i - t_c * s * vy_i)) - (
            -((b * s) / (t_c * d)) * (y_d - (s / t_c) * y_i - c * vy_i)
        )

    def walk_pattern_gen(
        self, s_x, s_y, s_theta, s_x_1, s_y_1, s_theta_1, a, b
    ) -> None:
        self.calculate_next_com_state()
        self.calculate_new_foot_place(s_x, s_y, s_theta)
        self.set_new_walk_primitive(s_x_1, s_x_1, s_theta_1)
        self.calculate_target_state()
        self.calculate_modified_foot_placement(a, b)
