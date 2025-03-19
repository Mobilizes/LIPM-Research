import math

from decimal import Decimal as D

import numpy as np


def wrap(val, min, max):
    min_val = val - min
    min_max = max - min

    return min + ((min_max + (min_val % min_max)) % min_max)

# def compute_velocity_coefficients(v0, a0, vt, at, t_dbl):
#     delta_v = vt - v0
#
#     a = (-D("2") * delta_v + t_dbl * (a0 + at)) / t_dbl**D("3")
#     b = (D("3") * delta_v - t_dbl * (D("2") * a0 + at)) / t_dbl**D("2")
#     c = a0
#     d = v0
#
#     return (a, b, c, d)

def compute_velocity_coefficients(p0, v0, a0, pt, vt, at, t_dbl):
    f = D(p0)
    e = D(v0)
    d = D(a0) / D("2.0")

    r1 = D(pt) - D(p0) - D(v0) * D(t_dbl) - (D(a0) / D("2.0")) * D(t_dbl)**2
    r2 = D(vt) - D(v0) - D(a0) * D(t_dbl)
    r3 = D(at) - D(a0)

    a = (D("6.0") * r1) / D(t_dbl)**5 - (D("3.0") * r2) / D(t_dbl)**4 + r3 / (D("2.0") * D(t_dbl)**3)
    b = (D("-15.0") * r1) / D(t_dbl)**4 + (D("7.0") * r2) / D(t_dbl)**3 - r3 / D(t_dbl)**2
    c = (D("10.0") * r1) / D(t_dbl)**3 - (D("4.0") * r2) / D(t_dbl)**2 + r3 / (D("2.0") * D(t_dbl))

    # Return coefficients [a, b, c, d, e, f]
    return [a, b, c, d, e, f]


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
        t_dbl=D("0.0")
    ):
        self.left_foot_pos = left_foot_pos
        self.right_foot_pos = right_foot_pos
        self.y_offset = y_offset
        self.support_leg = support_leg  # left / both / right
        self.previous_support_leg = "both"

        # Swing leg position at start of step
        self.start_swing_foot = self.get_swing_leg()

        # Time for non support leg be in swing state
        self.t_sup = t_sup

        # Time for COM to transfer support leg in double support phase
        # 0 means double support disabled
        self.t_dbl = t_dbl

        # Foot location at start of step
        self.p_x = self.get_support_leg()[0]
        self.p_y = self.get_support_leg()[1]

        # Foot location at end of step
        self.mod_p_x = self.p_x
        self.mod_p_y = self.p_y

        # Weight for mod_p evaluation function
        self.a = a
        self.b = b

        self.t = D("0.0")
        self.t_s = D("0.0")

        self.g = D("9.81")
        self.z_c = z_c
        self.t_c = D(np.sqrt(float(self.z_c / self.g)))
        self.c = D(np.cosh(float(self.t_sup / self.t_c)))
        self.s = D(np.sinh(float(self.t_sup / self.t_c)))

        self.walk_x = D("0.0")
        self.walk_y = D("0.0")
        self.walk_vx = D("0.0")
        self.walk_vy = D("0.0")

        # COM state at start of step
        self.x_i = self.p_x
        self.vx_i = D("0.0")
        self.ax_i = D("0.0")
        self.y_i = self.p_y
        self.vy_i = D("0.0")
        self.ay_i = D("0.0")

        # COM state in real time
        self.x_t = self.p_x
        self.vx_t = D("0.0")
        self.ax_t = D("0.0")
        self.y_t = self.p_y
        self.vy_t = D("0.0")
        self.ay_t = D("0.0")

        # COM state at end of step
        self.x_f = self.p_x
        self.vx_f = D("0.0")
        self.ax_f = D("0.0")
        self.y_f = self.p_y
        self.vy_f = D("0.0")
        self.ay_f = D("0.0")

        # COM target state
        self.x_d = self.p_x
        self.vx_d = D("0.0")
        self.y_d = self.p_y
        self.vy_d = D("0.0")

        # Walk parameters
        self.s_x = D("0.0")
        self.s_y = D("0.0")
        self.s_theta = D("0.0")

        self.s_x_1 = D("0.0")
        self.s_y_1 = D("0.0")
        self.s_theta_1 = D("0.0")

        self.s_x_2 = D("0.0")
        self.s_y_2 = D("0.0")
        self.s_theta_2 = D("0.0")

        self.n = D("0.0")

        if y_offset == D("0.0"):
            self.y_offset = abs(left_foot_pos[1] - right_foot_pos[1])

        # Walk speed (will modify next walk parameter)
        self.x_speed = D("0.0")
        self.y_speed = D("0.0")
        self.a_speed = D("0.0")

        # Current velocity coefficients
        self.coeff_calc_this_step = False

        self.cxa = D("0.0")
        self.cxb = D("0.0")
        self.cxc = D("0.0")
        self.cxd = D("0.0")
        self.cxe = D("0.0")
        self.cxf = D("0.0")

        self.cya = D("0.0")
        self.cyb = D("0.0")
        self.cyc = D("0.0")
        self.cyd = D("0.0")
        self.cye = D("0.0")
        self.cyf = D("0.0")

        self.initial = True
        self.initial_ssp = True
        self.initial_dsp = True

    def get_support_leg(self) -> np.ndarray:
        return self.left_foot_pos if self.support_leg == "left" else self.right_foot_pos

    def get_swing_leg(self) -> np.ndarray:
        return self.right_foot_pos if self.support_leg == "left" else self.left_foot_pos

    def set_support_leg(self, value: np.ndarray):
        if self.support_leg == "right":
            self.right_foot_pos = value
        elif self.support_leg == "left":
            self.left_foot_pos = value

    def set_swing_leg(self, value: np.ndarray):
        if self.support_leg == "left":
            self.right_foot_pos = value
        elif self.support_leg == "right":
            self.left_foot_pos = value

    def update_t_sup(self, t_sup):
        self.t_sup = t_sup
        t_c = self.t_c

        self.c = np.cosh(t_sup / t_c)
        self.s = np.cosh(t_sup / t_c)

    def update_z_c(self, z_c):
        self.z_c = z_c
        self.t_c = np.sqrt(z_c / 9.81)
        t_c = self.t_c
        t_sup = self.t_sup

        self.c = np.cosh(t_sup / t_c)
        self.s = np.cosh(t_sup / t_c)

    def get_distance_between_legs(self) -> D:
        return D(math.sqrt(
            float(self.right_foot_pos[0] - self.left_foot_pos[0])**2 +
            float(self.right_foot_pos[1] - self.left_foot_pos[1])**2))

    # Step 3
    def calculate_next_com_state(self):
        x_t, y_t = self.x_t, self.y_t
        vx_t, vy_t = self.vx_t, self.vy_t
        ax_t, ay_t = self.ax_t, self.ay_t

        self.x_i = x_t
        self.vx_i = vx_t
        self.ax_i = ax_t

        self.y_i = y_t
        self.vy_i = vy_t
        self.ay_i = ay_t

    # Step 5
    def calculate_new_foot_place(self):
        s_x, s_y, s_theta = self.s_x, self.s_y, self.s_theta

        # TODO: this formula cannot stepping turn in place, find a new formula
        theta_c = np.cos(float(s_theta))
        theta_s = np.sin(float(s_theta))
        if self.support_leg == "right":
            self.p_x += D(theta_c) * s_x - D(theta_s) * s_y
            self.p_y += D(theta_s) * s_x + D(theta_c) * s_y
        elif self.support_leg == "left":
            self.p_x += D(theta_c) * s_x + D(theta_s) * s_y
            self.p_y += D(theta_s) * s_x - D(theta_c) * s_y

    # Step 6
    def set_new_walk_primitive(self):
        """
        Make sure the step length parameters are a step ahead
        from new foot place
        """
        c = self.c
        s = self.s
        t_c = self.t_c

        s_x_1, s_y_1, s_theta_1 = self.s_x_1, self.s_y_1, self.s_theta_1

        theta_c = np.cos(float(s_theta_1))
        theta_s = np.sin(float(s_theta_1))
        if self.support_leg == "right":
            self.walk_x = D(theta_c) * s_x_1 / D("2.0") + D(theta_s) * s_y_1 / D("2.0")
            self.walk_y = D(theta_s) * s_x_1 / D("2.0") - D(theta_c) * s_y_1 / D("2.0")
        elif self.support_leg == "left":
            self.walk_x = D(theta_c) * s_x_1 / D("2.0") - D(theta_s) * s_y_1 / D("2.0")
            self.walk_y = D(theta_s) * s_x_1 / D("2.0") + D(theta_c) * s_y_1 / D("2.0")

        walk_x = self.walk_x
        walk_y = self.walk_y

        a = (1 + c) / (t_c * s) * walk_x
        b = (c - 1) / (t_c * s) * walk_y

        self.walk_vx = D(theta_c) * a - D(theta_s) * b
        self.walk_vy = D(theta_s) * a + D(theta_c) * b

    # Step 7
    def calculate_target_com_state(self):
        p_x, p_y = self.p_x, self.p_y
        walk_x, walk_y = self.walk_x, self.walk_y
        walk_vx, walk_vy = self.walk_vx, self.walk_vy

        self.x_d = p_x + walk_x
        self.vx_d = walk_vx

        self.y_d = p_y + walk_y
        self.vy_d = walk_vy

    # Step 8
    def calculate_modified_foot_placement(self, init_state=None, init_vel=None, target_state=None, target_vel=None):
        if init_state is None:
            init_state = self.x_i, self.y_i

        if init_vel is None:
            init_vel = self.vx_i, self.vy_i

        if target_state is None:
            target_state = self.x_d, self.y_d

        if target_vel is None:
            target_vel = self.vx_d, self.vy_d

        c, s = self.c, self.s
        t_c = self.t_c
        x_i, y_i = init_state
        vx_i, vy_i = init_vel
        x_d, y_d = target_state
        vx_d, vy_d = target_vel
        a, b = self.a, self.b

        d = a * (c - 1) ** 2 + b * (s / t_c) ** 2

        mod_p_x = (-(a * (c - 1) / d) * (x_d - c * x_i - t_c * s * vx_i)) - (
            ((b * s) / (t_c * d)) * (vx_d - (s / t_c) * x_i - c * vx_i)
        )

        mod_p_y = (-(a * (c - 1) / d) * (y_d - c * y_i - t_c * s * vy_i)) - (
            ((b * s) / (t_c * d)) * (vy_d - (s / t_c) * y_i - c * vy_i)
        )

        return (mod_p_x, mod_p_y)

    def switch_support_leg(self):
        support_leg = self.support_leg
        previous_support_leg = self.previous_support_leg

        if self.t_dbl == D("0.0"):
            self.support_leg = "left" if support_leg == "right" else "right"
        else:
            self.previous_support_leg = support_leg

            if support_leg != "both":
                self.support_leg = "both"
            else:
                self.support_leg = "left" if previous_support_leg == "right" else "right"

        self.start_swing_foot = self.get_swing_leg()

    def walk_pattern_gen(self):
        self.calculate_next_com_state()
        self.calculate_new_foot_place()
        self.set_new_walk_primitive()
        self.calculate_target_com_state()
        mod_p = self.calculate_modified_foot_placement()

        self.mod_p_x, self.mod_p_y = mod_p

    def analytical_real_time_com_state(self, t, init_state=None, init_vel=None, mod_p=None):
        if init_state is None:
            init_state = self.x_i, self.y_i

        if init_vel is None:
            init_vel = self.vx_i, self.vy_i

        if mod_p is None:
            mod_p = self.mod_p_x, self.mod_p_y

        t_c = self.t_c
        x_i, y_i = init_state
        vx_i, vy_i = init_vel
        mod_p_x, mod_p_y = mod_p
        g, z_c = self.g, self.z_c

        x_t = (
            (x_i - mod_p_x) * D(np.cosh(float(t) / float(t_c)))
            + t_c * vx_i * D(np.sinh(float(t) / float(t_c)))
            + mod_p_x
        )
        vx_t = ((x_i - mod_p_x) / t_c) * D(
            np.sinh(float(t) / float(t_c))
        ) + vx_i * D(np.cosh(float(t) / float(t_c)))
        ax_t = g / z_c * (x_t - mod_p_x)

        y_t = (
            (y_i - mod_p_y) * D(np.cosh(float(t) / float(t_c)))
            + t_c * vy_i * D(np.sinh(float(t) / float(t_c)))
            + mod_p_y
        )
        vy_t = ((y_i - mod_p_y) / t_c) * D(
            np.sinh(float(t) / float(t_c))
        ) + vy_i * D(np.cosh(float(t) / float(t_c)))
        ay_t = g / z_c * (y_t - mod_p_y)

        return ((x_t, y_t), (vx_t, vy_t), (ax_t, ay_t))

    def double_support_phase_com_state(self, t_s):
        t_sup, t_dbl = self.t_sup, self.t_dbl

        if not self.coeff_calc_this_step:
            com_i = self.analytical_real_time_com_state(t_sup)

            x_i, y_i = com_i[0]
            vx_i, vy_i = com_i[1]
            ax_i, ay_i = com_i[2]

            s_x_1, s_y_1, s_theta_1 = self.s_x_1, self.s_y_1, self.s_theta_1
            s_x_2, s_y_2, s_theta_2 = self.s_x_2, self.s_y_2, self.s_theta_2

            theta_c = np.cos(float(s_theta_1))
            theta_s = np.sin(float(s_theta_1))
            if self.previous_support_leg == "left":
                p_x = D(theta_c) * s_x_1 - D(theta_s) * s_y_1 + self.p_x
                p_y = D(theta_s) * s_x_1 + D(theta_c) * s_y_1 + self.p_y
            else:
                p_x = D(theta_c) * s_x_1 + D(theta_s) * s_y_1 + self.p_x
                p_y = D(theta_s) * s_x_1 - D(theta_c) * s_y_1 + self.p_y

            theta_c = np.cos(float(s_theta_2))
            theta_s = np.sin(float(s_theta_2))
            if self.previous_support_leg == "left":
                walk_x = D(theta_c) * s_x_2 / D("2.0") + D(theta_s) * s_y_2 / D("2.0")
                walk_y = D(theta_s) * s_x_2 / D("2.0") - D(theta_c) * s_y_2 / D("2.0")
            else:
                walk_x = D(theta_c) * s_x_2 / D("2.0") - D(theta_s) * s_y_2 / D("2.0")
                walk_y = D(theta_s) * s_x_2 / D("2.0") + D(theta_c) * s_y_2 / D("2.0")

            t_c = self.t_c
            c, s = self.c, self.s

            a = (1 + c) / (t_c * s) * walk_x
            b = (c - 1) / (t_c * s) * walk_y

            walk_vx = D(theta_c) * a - D(theta_s) * b
            walk_vy = D(theta_s) * a + D(theta_c) * b

            x_d = p_x + walk_x
            vx_d = walk_vx

            y_d = p_y + walk_y
            vy_d = walk_vy

            mod_p = self.calculate_modified_foot_placement(
                (x_i, y_i), (vx_i, vy_i), (x_d, y_d), (vx_d, vy_d)
            )

            com_t = self.analytical_real_time_com_state(
                t_dbl, (x_i, y_i), (vx_i, vy_i), mod_p
            )

            x_f, y_f = com_t[0]
            vx_f, vy_f = com_t[1]
            ax_f, ay_f = com_t[2]

            print("Init state : " + str(x_i) + ", " + str(y_i))
            print("Init vel : " + str(vx_i) + ", " + str(vy_i))
            print("Init acc : " + str(ax_i) + ", " + str(ay_i))
            print("Target state : " + str(x_f) + ", " + str(y_f))
            print("Target vel : " + str(vx_f) + ", " + str(vy_f))
            print("Target acc : " + str(ax_f) + ", " + str(ay_f))
            print()

            self.cxa, self.cxb, self.cxc, self.cxd, self.cxe, self.cxf = compute_velocity_coefficients(
                x_i, vx_i, ax_i, x_f, vx_f, ax_f, t_dbl
            )
            self.cya, self.cyb, self.cyc, self.cyd, self.cye, self.cyf = compute_velocity_coefficients(
                y_i, vy_i, ay_i, y_f, vy_f, ay_f, t_dbl
            )

            self.coeff_calc_this_step = True

        cxa, cxb, cxc, cxd, cxe, cxf = self.cxa, self.cxb, self.cxc, self.cxd, self.cxe, self.cxf
        cya, cyb, cyc, cyd, cye, cyf = self.cya, self.cyb, self.cyc, self.cyd, self.cye, self.cyf

        x_t = cxa * t_s**D("5") + cxb * t_s**D("4") + cxc * t_s**D("3") + cxd * t_s**D("2") + cxe * t_s + cxf
        y_t = cya * t_s**D("5") + cyb * t_s**D("4") + cyc * t_s**D("3") + cyd * t_s**D("2") + cye * t_s + cyf

        vx_t = D("5") * cxa * t_s**D("4") + D("4") * cxb * t_s**D("3") + D("3") * cxc * t_s**D("2") + D("2") * cxd * t_s + cxe
        vy_t = D("5") * cya * t_s**D("4") + D("4") * cyb * t_s**D("3") + D("3") * cyc * t_s**D("2") + D("2") * cyd * t_s + cye

        ax_t = D("20") * cxa * t_s**D("3") + D("12") * cxb * t_s**D("2") + D("6") * cxc * t_s + D("2") * cxd
        ay_t = D("20") * cya * t_s**D("3") + D("12") * cyb * t_s**D("2") + D("6") * cyc * t_s + D("2") * cyd

        return ((x_t, y_t), (vx_t, vy_t), (ax_t, ay_t))

    def calculate_real_time_com_state(self):
        t, t_s = self.t, self.t_s
        t_sup, t_dbl = self.t_sup, self.t_dbl
        n = self.n

        if self.support_leg != "both":
            t -= t_sup * (n - D("1.0"))
            com = self.analytical_real_time_com_state(t)
        else:
            t_s -= t_dbl * (n - D("1.0"))
            com = self.double_support_phase_com_state(t_s)

        self.x_t, self.y_t = com[0]
        self.vx_t, self.vy_t = com[1]
        self.ax_t, self.ay_t = com[2]

    def move_swing_leg(self):
        start_swing_foot = self.start_swing_foot
        mod_p_x, mod_p_y = self.mod_p_x, self.mod_p_y
        t = self.t
        t_sup, n = self.t_sup, self.n

        progress_t = (t - t_sup * (n - D("1.0"))) / t_sup

        update_swing = np.array([mod_p_x, mod_p_y, D("0.0")]) - start_swing_foot
        update_swing = [pos * progress_t for pos in update_swing]
        update_swing[2] = D("0.5") * D(np.sin(float(progress_t) * np.pi))

        self.set_swing_leg(start_swing_foot + update_swing)

    def set_walk_parameter(self, input):
        s_x_1, s_y_1, s_theta_1 = self.s_x_1, self.s_y_1, self.s_theta_1

        self.s_x, self.s_y, self.s_theta = s_x_1, s_y_1, s_theta_1
        self.s_x_1, self.s_y_1, self.s_theta_1 = input

    def update_walk_parameter(self):
        """
        Run only at start of new step, before walk pattern generation
        """
        x_speed, y_speed, a_speed = self.x_speed, self.y_speed, self.a_speed
        s_x_1, s_y_1, s_theta_1 = self.s_x_1, self.s_y_1, self.s_theta_1
        s_x_2, s_y_2, s_theta_2 = self.s_x_2, self.s_y_2, self.s_theta_2
        y_offset = self.y_offset

        # Update new x walk parameter
        self.s_x_2 = x_speed

        # Update new a (theta) walk parameter
        self.s_theta_2 = wrap(s_theta_2 + a_speed, -D(np.pi), D(np.pi))

        # Update new y walk parameter
        self.s_y_2 = y_offset * D("2.0")
        if s_y_2 < y_offset * D("2.0"):
            self.s_y_2 += y_speed
        else:
            self.s_y_2 -= y_speed

        # Update walk parameter iteration
        self.s_x_1 = s_x_2
        self.s_y_1 = s_y_2
        self.s_theta_1 = s_theta_2

        self.s_x = s_x_1
        self.s_y = s_y_1
        self.s_theta = s_theta_1

    def update_end_com_state(self):
        t_sup = self.t_sup

        com = self.analytical_real_time_com_state(t_sup)

        self.x_f, self.y_f = com[0]
        self.vx_f, self.vy_f = com[1]
        self.ax_f, self.ay_f = com[2]

    def single_support_phase_step(self, dt):
        t = self.t
        t_sup, n = self.t_sup, self.n

        if self.initial_ssp:
            self.initial_ssp = False

            self.print_info()

        self.t += dt
        if t >= t_sup * n and self.support_leg != "both":
            self.n += D("1.0")

            self.update_walk_parameter()
            self.walk_pattern_gen()
            self.update_end_com_state()
            self.switch_support_leg()

            self.print_info()
            self.initial_ssp = True

        self.move_swing_leg()

    def double_support_phase_step(self, dt):
        t_s = self.t_s
        t_dbl = self.t_dbl
        n = self.n

        if self.initial_dsp:
            self.initial_dsp = False

        self.t_s += dt
        if t_s >= t_dbl * n:
            self.switch_support_leg()
            self.coeff_calc_this_step = False

    def step(self, dt):
        if self.initial:
            self.initial = False

            self.update_walk_parameter()

        self.calculate_real_time_com_state()

        if self.support_leg == "both":
            self.double_support_phase_step(dt)
        else:
            self.single_support_phase_step(dt)

        reset_support_leg = self.get_support_leg()
        reset_support_leg[2] = D("0.0")
        self.set_support_leg(reset_support_leg)

    def print_info(self):
        print("------------------------------------------------------------")
        print(f"Time : {self.t}")
        print(f"At : {"SSP" if self.support_leg != "both" else "DSP"}")
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
                np.array([self.x_i, self.y_i, self.vx_i, self.vy_i, self.ax_i, self.ay_i])}"
        )
        print(
            f"Current COM state : {
                np.array([self.x_t, self.y_t, self.vx_t, self.vy_t, self.ax_t, self.ay_t])}"
        )
        print(
            f"Final COM state : {
                np.array([self.x_f, self.y_f, self.vx_f, self.vy_f, self.ax_f, self.ay_f])}"
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
