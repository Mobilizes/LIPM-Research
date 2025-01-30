from decimal import Decimal as D

import numpy as np


class LIPM3D:
    def __init__(
        self,
        left_foot_pos: np.ndarray,
        right_foot_pos: np.ndarray,
        z_c: D,
        a=D("1.0"),
        b=D("1.0"),
        y_offset=D("0.0"),
        support_leg="right",
        t_sup=D("0.5"),
    ) -> None:
        self.left_foot_pos = left_foot_pos
        self.right_foot_pos = right_foot_pos
        self.y_offset = y_offset
        self.support_leg = support_leg  # left / both / right

        # Swing leg position at start of step
        self.start_swing_foot = self.get_swing_leg()

        # Time for non support leg be in swing state
        self.t_sup = t_sup

        # Foot location at end of step
        self.mod_p_x = self.get_support_leg()[0]
        self.mod_p_y = self.get_support_leg()[1]

        # Foot location at start of step
        self.p_x = self.get_support_leg()[0]
        self.p_y = self.get_support_leg()[1]

        # Weight for mod_p evaluation function
        self.a = a
        self.b = b

        self.t = D("0.0")

        self.z_c = z_c
        self.t_c = np.sqrt(self.z_c / D("9.81"))
        self.c = D(np.cosh(float(self.t_sup / self.t_c)))
        self.s = D(np.sinh(float(self.t_sup / self.t_c)))

        self.walk_x = D("0.0")
        self.walk_y = D("0.0")
        self.walk_vx = D("0.0")
        self.walk_vy = D("0.0")

        # COM state at start of step
        self.x_i = self.get_support_leg()[0]
        self.vx_i = D("0.0")
        self.y_i = self.get_support_leg()[1]
        self.vy_i = D("0.0")

        # COM state in real time
        self.x_t = self.get_support_leg()[0]
        self.vx_t = D("0.0")
        self.y_t = self.get_support_leg()[1]
        self.vy_t = D("0.0")

        # COM target state
        self.x_d = D("0.0")
        self.vx_d = D("0.0")
        self.y_d = D("0.0")
        self.vy_d = D("0.0")

        # Walk parameters
        self.s_x = D("0.0")
        self.s_y = D("0.0")
        self.s_theta = D("0.0")

        self.s_x_1 = D("0.0")
        self.s_y_1 = D("0.0")
        self.s_theta_1 = D("0.0")

        self.n = D("0.0")

    def get_support_leg(self) -> np.ndarray:
        return self.left_foot_pos if self.support_leg == "left" else self.right_foot_pos

    def get_swing_leg(self) -> np.ndarray:
        return self.right_foot_pos if self.support_leg == "left" else self.left_foot_pos

    def set_swing_leg(self, value: np.ndarray) -> None:
        if self.support_leg == "left":
            self.right_foot_pos = value
        elif self.support_leg == "right":
            self.left_foot_pos = value

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
    def calculate_next_com_state(self) -> None:
        x_t, y_t = self.x_t, self.y_t
        vx_t, vy_t = self.vx_t, self.vy_t

        self.x_i = x_t
        self.vx_i = vx_t

        self.y_i = y_t
        self.vy_f = vy_t

    # Step 5
    def calculate_new_foot_place(self) -> None:
        s_x, s_y, s_theta = self.s_x, self.s_y, self.s_theta
        s_y += self.y_offset

        theta_c = np.cos(float(s_theta))
        theta_s = np.sin(float(s_theta))
        if self.support_leg == "left":
            self.p_x += D(theta_c) * s_x - D(theta_s) * s_y
            self.p_y += D(theta_s) * s_x + D(theta_c) * s_y
        elif self.support_leg == "right":
            self.p_x += D(theta_c) * s_x + D(theta_s) * s_y
            self.p_y += D(theta_s) * s_x - D(theta_c) * s_y

    # Step 6
    def set_new_walk_primitive(self) -> None:
        """
        Make sure the step length paramters are a step ahead
        from new foot place
        """
        c = self.c
        s = self.s
        t_c = self.t_c

        s_x_1, s_y_1, s_theta_1 = self.s_x_1, self.s_y_1, self.s_theta_1
        s_y_1 += self.y_offset

        theta_c = np.cos(float(s_theta_1))
        theta_s = np.sin(float(s_theta_1))
        if self.support_leg == "left":
            self.walk_x = D(theta_c) * s_x_1 / D("2.0") + \
                D(theta_s) * s_y_1 / D("2.0")
            self.walk_y = D(theta_s) * s_x_1 / D("2.0") - \
                D(theta_c) * s_y_1 / D("2.0")
        elif self.support_leg == "right":
            self.walk_x = D(theta_c) * s_x_1 / D("2.0") - \
                D(theta_s) * s_y_1 / D("2.0")
            self.walk_y = D(theta_s) * s_x_1 / D("2.0") + \
                D(theta_c) * s_y_1 / D("2.0")

        walk_x = self.walk_x
        walk_y = self.walk_y

        a = (1 + c) / (t_c * s) * walk_x
        b = (c - 1) / (t_c * s) * walk_y

        self.walk_vx = D(theta_c) * a - D(theta_s) * b
        self.walk_vy = D(theta_s) * a + D(theta_c) * b

    # Step 7
    def calculate_target_com_state(self) -> None:
        p_x, p_y = self.p_x, self.p_y
        walk_x, walk_y = self.walk_x, self.walk_y
        walk_vx, walk_vy = self.walk_vx, self.walk_vy

        self.x_d = p_x + walk_x
        self.vx_d = walk_vx

        self.y_d = p_y + walk_y
        self.vy_d = walk_vy

    # Step 8
    def calculate_modified_foot_placement(self) -> None:
        c, s = self.c, self.s
        t_c = self.t_c
        x_i, y_i = self.x_i, self.y_i
        vx_i, vy_i = self.vx_i, self.vy_i
        x_d, y_d = self.x_d, self.y_d
        vx_d, vy_d = self.vx_d, self.vy_d
        a, b = self.a, self.b

        d = a * (c - 1) ** 2 + b * (s / t_c) ** 2

        self.mod_p_x = (-(a * (c - 1) / d) * (x_d - c * x_i - t_c * s * vx_i)) - (
            ((b * s) / (t_c * d)) * (vx_d - (s / t_c) * x_i - c * vx_i)
        )

        self.mod_p_y = (-(a * (c - 1) / d) * (y_d - c * y_i - t_c * s * vy_i)) - (
            ((b * s) / (t_c * d)) * (vy_d - (s / t_c) * y_i - c * vy_i)
        )

    def switch_support_leg(self) -> None:
        # x_i, y_i = self.x_i, self.y_i
        # right_foot_pos, left_foot_pos = self.right_foot_pos, self.left_foot_pos

        if self.support_leg == "right":
            self.support_leg = "left"
            # self.x_f = x_i + left_foot_pos[0] - right_foot_pos[0]
            # self.y_f = y_i + left_foot_pos[1] - right_foot_pos[1]
        elif self.support_leg == "left":
            self.support_leg = "right"
        #     self.x_f = x_i + right_foot_pos[0] - left_foot_pos[0]
        #     self.y_f = y_i + right_foot_pos[1] - left_foot_pos[1]
        #
        # self.vx_i = self.vx_f
        # self.vy_i = self.vy_f

    def walk_pattern_gen(self) -> None:
        self.calculate_next_com_state()
        self.calculate_new_foot_place()
        self.set_new_walk_primitive()
        self.calculate_target_com_state()
        self.calculate_modified_foot_placement()

    def calculate_real_time_com_state(self) -> None:
        t, t_sup, t_c = self.t, self.t_sup, self.t_c
        x_i, y_i = self.x_i, self.y_i
        vx_i, vy_i = self.vx_i, self.vy_i
        mod_p_x, mod_p_y = self.mod_p_x, self.mod_p_y
        n = self.n

        t %= t_sup

        if self.t >= t_sup * n:
            t += t_sup

        self.x_t = (
            (x_i - mod_p_x) * D(np.cosh(float(t) / float(t_c)))
            + t_c * vx_i * D(np.sinh(float(t) / float(t_c)))
            + mod_p_x
        )
        self.vx_t = ((x_i - mod_p_x) / t_c) * D(
            np.sinh(float(t) / float(t_c))
        ) + vx_i * D(np.cosh(float(t) / float(t_c)))

        self.y_t = (
            (y_i - mod_p_y) * D(np.cosh(float(t) / float(t_c)))
            + t_c * vy_i * D(np.sinh(float(t) / float(t_c)))
            + mod_p_y
        )
        self.vy_t = ((y_i - mod_p_y) / t_c) * D(
            np.sinh(float(t) / float(t_c))
        ) + vy_i * D(np.cosh(float(t) / float(t_c)))

        print("From calculate_real_time_com_state")
        print(self.x_t)
        print(self.y_t)

    def move_swing_leg(self) -> None:
        start_swing_leg = self.start_swing_foot
        mod_p_x, mod_p_y = self.mod_p_x, self.mod_p_y
        t = self.t
        t_sup, n = self.t_sup, self.n
        z_c = self.z_c

        progress_t = 1 + (t - t_sup * n) / t_sup

        update_z = z_c - abs(progress_t / D(2.0) - z_c)
        update_swing = np.array([mod_p_x, mod_p_y, update_z]) - start_swing_leg
        update_swing = [val * progress_t for val in update_swing]

        self.set_swing_leg(start_swing_leg + update_swing)
        self.print_info()

    def set_walk_parameter(self, input) -> None:
        s_x_1, s_y_1, s_theta_1 = self.s_x_1, self.s_y_1, self.s_theta_1

        self.s_x, self.s_y, self.s_theta = s_x_1, s_y_1, s_theta_1
        self.s_x_1, self.s_y_1, self.s_theta_1 = input

    def step(self, dt) -> None:
        t = self.t
        t_sup, n = self.t_sup, self.n

        self.t += dt

        print(abs(t_sup * n - t))
        print(abs(t_sup * n - t) <= D("0.01"))
        if t >= t_sup * n:
            print("HELLO")
            self.n += D("1.0")

            print("From step")
            print(self.x_t)
            print(self.y_t)

            self.walk_pattern_gen()
            self.switch_support_leg()

            self.start_swing_foot = self.get_swing_leg()

        self.calculate_real_time_com_state()
        self.move_swing_leg()

    def print_info(self) -> None:
        print("------------------------------------------------------------")
        print(f"Time : {self.t}")
        print(f"Step end : {self.t_sup * self.n}")
        print(f"Iteration : {self.n}")
        print(f"Support leg : {self.support_leg}")
        print(f"Left foot pos : {self.left_foot_pos}")
        print(f"Right foot pos : {self.right_foot_pos}")
        print(f"Current foot placement : {np.array([self.p_x, self.p_y])}")
        print(f"Next foot placement : {
              np.array([self.mod_p_x, self.mod_p_y])}")
        print(f"Walk primitive : {np.array([self.walk_x, self.walk_y])}")
        print(f"Walk primitive velocity : {
              np.array([self.walk_vx, self.walk_vy])}")
        print(
            f"Initial COM state : {
                np.array([self.x_i, self.y_i, self.vx_i, self.vy_i])}"
        )
        print(
            f"Current COM state : {
                np.array([self.x_t, self.y_t, self.vx_t, self.vy_t])}"
        )
        print(
            f"Desired COM state : {
                np.array([self.x_d, self.y_d, self.vx_d, self.vy_d])}"
        )
        print(f"C : {self.c}")
        print(f"S : {self.s}")
        print(f"Tc : {self.t_c}")
        print(f"Progress t : {
              1 + (self.t - self.t_sup * self.n) / self.t_sup}")
        print("------------------------------------------------------------")
