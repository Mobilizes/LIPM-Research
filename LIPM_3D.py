import math

from decimal import Decimal as D
from os import walk

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


# def compute_velocity_coefficients(p0, v0, a0, pt, vt, at, t_dbl):
#     f = D(p0)
#     e = D(v0)
#     d = D(a0) / D("2.0")
#
#     r1 = D(pt) - D(p0) - D(v0) * D(t_dbl) - (D(a0) / D("2.0")) * D(t_dbl) ** 2
#     r2 = D(vt) - D(v0) - D(a0) * D(t_dbl)
#     r3 = D(at) - D(a0)
#
#     a = (
#         (D("6.0") * r1) / D(t_dbl) ** 5
#         - (D("3.0") * r2) / D(t_dbl) ** 4
#         + r3 / (D("2.0") * D(t_dbl) ** 3)
#     )
#     b = (
#         (D("-15.0") * r1) / D(t_dbl) ** 4
#         + (D("7.0") * r2) / D(t_dbl) ** 3
#         - r3 / D(t_dbl) ** 2
#     )
#     c = (
#         (D("10.0") * r1) / D(t_dbl) ** 3
#         - (D("4.0") * r2) / D(t_dbl) ** 2
#         + r3 / (D("2.0") * D(t_dbl))
#     )
#
#     # Return coefficients [a, b, c, d, e, f]
#     return [a, b, c, d, e, f]

def compute_velocity_coefficients(p0, v0, a0, pt, vt, t_dbl):
    a = v0
    b = a0
    c = (12 * (pt - p0) - 3 * vt * t_dbl - 9 * v0 * t_dbl - 3 * a0 * t_dbl**2) / t_dbl**3
    d = (vt - (a + b * t_dbl + c * t_dbl**2)) / t_dbl**3

    return (a, b, c, d)


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
        t_dbl=D("0.0"),
    ):
        self.left_foot_pos = left_foot_pos
        self.right_foot_pos = right_foot_pos
        self.y_offset = y_offset
        self.support_leg = support_leg  # left / right
        self.previous_support_leg = "left" if support_leg == "right" else support_leg

        # Swing leg position at start of step
        self.start_swing_foot = self.get_swing_leg()

        # Time for non support leg be in swing state
        self.t_sup = t_sup

        # Time for COM to transfer support leg in double support phase
        # 0 means double support disabled
        self.t_dbl = t_dbl

        # Foot location at start of step
        self.p_x = [self.get_swing_leg()[0]] * 2
        self.p_y = [self.get_swing_leg()[1]] * 2

        # Foot location at end of step
        self.mod_p_x = [self.p_x[0]] * 2
        self.mod_p_y = [self.p_y[0]] * 2

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

        self.walk_x = [D("0.0")] * 2
        self.walk_y = [D("0.0")] * 2
        self.walk_vx = [D("0.0")] * 2
        self.walk_vy = [D("0.0")] * 2

        # COM state at start of step
        self.x_i = [self.p_x[0]] * 2
        self.vx_i = [D("0.0")] * 2
        self.ax_i = [D("0.0")] * 2
        self.y_i = [self.p_y[0]] * 2
        self.vy_i = [D("0.0")] * 2
        self.ay_i = [D("0.0")] * 2

        # COM state in real time
        self.x_t = [self.p_x[0]] * 50
        self.vx_t = [D("0.0")] * 50
        self.ax_t = [D("0.0")] * 50
        self.y_t = [self.p_y[0]] * 50
        self.vy_t = [D("0.0")] * 50
        self.ay_t = [D("0.0")] * 50

        # COM state at end of step
        self.x_f = [self.p_x[0]] * 2
        self.vx_f = [D("0.0")] * 2
        self.ax_f = [D("0.0")] * 2
        self.y_f = [self.p_y[0]] * 2
        self.vy_f = [D("0.0")] * 2
        self.ay_f = [D("0.0")] * 2

        # COM target state
        self.x_d = [self.p_x[0]] * 2
        self.vx_d = [D("0.0")] * 2
        self.y_d = [self.p_y[0]] * 2
        self.vy_d = [D("0.0")] * 2

        # Walk parameters
        self.s_x = [D("0.0")] * 3
        self.s_y = [D("0.0")] * 3
        self.s_theta = [D("0.0")] * 3

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

        self.state_double_support_phase = False

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
        return D(
            math.sqrt(
                float(self.right_foot_pos[0] - self.left_foot_pos[0]) ** 2
                + float(self.right_foot_pos[1] - self.left_foot_pos[1]) ** 2
            )
        )

    # Step 3
    def update_next_com(self, real_time_com=None):
        if real_time_com is None:
            real_time_com = (
                (self.x_t[0], self.y_t[0]),
                (self.vx_t[0], self.vy_t[0]),
                (self.ax_t[0], self.ay_t[0]),
            )

        x_t, y_t = real_time_com[0]
        vx_t, vy_t = real_time_com[1]
        ax_t, ay_t = real_time_com[2]

        self.x_i = [x_t, self.x_i[0]]
        self.vx_i = [vx_t, self.vx_i[0]]
        self.ax_i = [ax_t, self.ax_i[0]]

        self.y_i = [y_t, self.y_i[0]]
        self.vy_i = [vy_t, self.vy_i[0]]
        self.ay_i = [ay_t, self.ay_i[0]]

    # Step 5
    def calculate_new_foot_place(self, p=None, s_x=None, s_y=None, s_theta=None):
        if p is None:
            p = [self.p_x[0], self.p_y[0]]

        if s_x is None:
            s_x = self.s_x[0]

        if s_y is None:
            s_y = self.s_y[0]

        if s_theta is None:
            s_theta = self.s_theta[0]

        # TODO: this formula cannot stepping turn in place, find a new formula
        theta_c = np.cos(float(s_theta))
        theta_s = np.sin(float(s_theta))
        if self.support_leg == "right":
            p[0] += D(theta_c) * s_x - D(theta_s) * s_y
            p[1] += D(theta_s) * s_x + D(theta_c) * s_y
        else:
            p[0] += D(theta_c) * s_x + D(theta_s) * s_y
            p[1] += D(theta_s) * s_x - D(theta_c) * s_y

        return p

    def update_new_foot_place(self, p):
        self.p_x = [p[0], self.p_x[0]]
        self.p_y = [p[1], self.p_y[0]]

    # Step 6
    def calculate_new_walk_primitive(self, s_x=None, s_y=None, s_theta=None):
        """
        Make sure the step length parameters are a step ahead
        from new foot place
        """
        if s_x is None:
            s_x = self.s_x[1]

        if s_y is None:
            s_y = self.s_y[1]

        if s_theta is None:
            s_theta = self.s_theta[1]

        c = self.c
        s = self.s
        t_c = self.t_c

        theta_c = np.cos(float(s_theta))
        theta_s = np.sin(float(s_theta))
        if self.support_leg == "right":
            walk_x = D(theta_c) * s_x / D("2.0") + D(theta_s) * s_y / D("2.0")
            walk_y = D(theta_s) * s_x / D("2.0") - D(theta_c) * s_y / D("2.0")
        else:
            walk_x = D(theta_c) * s_x / D("2.0") - D(theta_s) * s_y / D("2.0")
            walk_y = D(theta_s) * s_x / D("2.0") + D(theta_c) * s_y / D("2.0")

        a = (1 + c) / (t_c * s) * walk_x
        b = (c - 1) / (t_c * s) * walk_y

        walk_vx = D(theta_c) * a - D(theta_s) * b
        walk_vy = D(theta_s) * a + D(theta_c) * b

        return ((walk_x, walk_y), (walk_vx, walk_vy))

    def update_walk_primitive(self, walk_primitive):
        walk_x, walk_y = walk_primitive[0]
        walk_vx, walk_vy = walk_primitive[1]

        self.walk_x = [walk_x, self.walk_x[0]]
        self.walk_y = [walk_y, self.walk_y[0]]
        self.walk_vx = [walk_vx, self.walk_vx[0]]
        self.walk_vy = [walk_vy, self.walk_vy[0]]

    # Step 7
    def calculate_target_com(self, p=None, walk_primitive=None):
        if p is None:
            p = self.p_x[0], self.p_y[0]

        if walk_primitive is None:
            walk_primitive = (
                (self.walk_x[0], self.walk_y[0]),
                (self.walk_vx[0], self.walk_vy[0]),
            )

        p_x, p_y = p
        walk_x, walk_y = walk_primitive[0]
        walk_vx, walk_vy = walk_primitive[1]

        x_d = p_x + walk_x
        y_d = p_y + walk_y

        vx_d = walk_vx
        vy_d = walk_vy

        return ((x_d, y_d), (vx_d, vy_d))

    def update_target_com(self, target_com):
        x_d, y_d = target_com[0]
        vx_d, vy_d = target_com[1]

        self.x_d = [x_d, self.x_d[0]]
        self.y_d = [y_d, self.y_d[0]]

        self.vx_d = [vx_d, self.vx_d[0]]
        self.vy_d = [vy_d, self.vy_d[0]]

    # Step 8
    def calculate_modified_foot_placement(
        self, init_state=None, init_vel=None, target_state=None, target_vel=None
    ):
        if init_state is None:
            init_state = self.x_i[0], self.y_i[0]

        if init_vel is None:
            init_vel = self.vx_i[0], self.vy_i[0]

        if target_state is None:
            target_state = self.x_d[0], self.y_d[0]

        if target_vel is None:
            target_vel = self.vx_d[0], self.vy_d[0]

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

    def update_modified_foot_placement(self, mod_p):
        self.mod_p_x = [mod_p[0], self.mod_p_x[0]]
        self.mod_p_y = [mod_p[1], self.mod_p_y[0]]

    def switch_support_leg(self):
        support_leg = self.support_leg

        self.previous_support_leg = support_leg
        self.support_leg = "left" if support_leg == "right" else "right"

        self.start_swing_foot = self.get_swing_leg()

    def walk_pattern_gen(self):
        self.update_next_com()

        p = self.calculate_new_foot_place()
        self.update_new_foot_place(p)
        print(self.p_x)

        walk_primitive = self.calculate_new_walk_primitive()
        self.update_walk_primitive(walk_primitive)
        print(self.walk_x)

        target_com = self.calculate_target_com()
        self.update_target_com(target_com)
        print(self.x_d)

        mod_p = self.calculate_modified_foot_placement()
        self.update_modified_foot_placement(mod_p)
        print(self.mod_p_x)

    def analytical_real_time_com_state(
        self, t, init_state=None, init_vel=None, mod_p=None
    ):
        if init_state is None:
            init_state = self.x_i[0], self.y_i[0]

        if init_vel is None:
            init_vel = self.vx_i[0], self.vy_i[0]

        if mod_p is None:
            mod_p = self.mod_p_x[0], self.mod_p_y[0]

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
        vx_t = ((x_i - mod_p_x) / t_c) * D(np.sinh(float(t) / float(t_c))) + vx_i * D(
            np.cosh(float(t) / float(t_c))
        )
        ax_t = g / z_c * (x_t - mod_p_x)

        y_t = (
            (y_i - mod_p_y) * D(np.cosh(float(t) / float(t_c)))
            + t_c * vy_i * D(np.sinh(float(t) / float(t_c)))
            + mod_p_y
        )
        vy_t = ((y_i - mod_p_y) / t_c) * D(np.sinh(float(t) / float(t_c))) + vy_i * D(
            np.cosh(float(t) / float(t_c))
        )
        ay_t = g / z_c * (y_t - mod_p_y)

        return ((x_t, y_t), (vx_t, vy_t), (ax_t, ay_t))

    def double_support_phase_com_state(self, t_s):
        t_sup, t_dbl = self.t_sup, self.t_dbl

        if not self.coeff_calc_this_step:

            self.coeff_calc_this_step = True

        cxa, cxb, cxc, cxd = (
            self.cxa,
            self.cxb,
            self.cxc,
            self.cxd,
        )
        cya, cyb, cyc, cyd = (
            self.cya,
            self.cyb,
            self.cyc,
            self.cyd,
        )

        x_i, y_i = D("0.0"), D("0.0")

        x_t = (
            cxa * t_s ** D("4") / D("4")
            + cxb * t_s ** D("3") / D("3")
            + cxc * t_s ** D("2") / D("2")
            + cxd
            + x_i
        )
        y_t = (
            cya * t_s ** D("4") / D("4")
            + cyb * t_s ** D("3") / D("3")
            + cyc * t_s ** D("2") / D("2")
            + cyd
            + y_i
        )

        vx_t = cxa * t_s ** D("3") + cxb * t_s ** D("2") + cxc * t_s + cxc
        vy_t = cya * t_s ** D("3") + cyb * t_s ** D("2") + cyc * t_s + cyc

        ax_t = D("3") * cxa * t_s ** D("2") + D("2") * cxb * t_s + cxc
        ay_t = D("3") * cya * t_s ** D("2") + D("2") * cyb * t_s + cyc

        return ((x_t, y_t), (vx_t, vy_t), (ax_t, ay_t))

    def calculate_real_time_com_state(self):
        t, t_s = self.t, self.t_s
        t_sup, t_dbl = self.t_sup, self.t_dbl
        n = self.n

        if self.state_double_support_phase:
            t_s -= t_dbl * (n - D("1.0"))
            com = self.double_support_phase_com_state(t_s)
        else:
            t -= t_sup * (n - D("1.0"))
            com = self.analytical_real_time_com_state(t)

        self.x_t = [com[0][0], self.x_t[0:]]
        self.vx_t = [com[1][0], self.vx_t[0:]]
        self.ax_t = [com[2][0], self.ax_t[0:]]

        self.y_t = [com[0][1], self.y_t[0:]]
        self.vy_t = [com[1][1], self.vy_t[0:]]
        self.ay_t = [com[2][1], self.ay_t[0:]]

    def move_swing_leg(self):
        start_swing_foot = self.start_swing_foot
        mod_p_x, mod_p_y = self.mod_p_x[0], self.mod_p_y[0]
        t = self.t
        t_sup, n = self.t_sup, self.n

        progress_t = (t - t_sup * (n - D("1.0"))) / t_sup

        update_swing = np.array([mod_p_x, mod_p_y, D("0.0")]) - start_swing_foot
        update_swing = [pos * progress_t for pos in update_swing]
        update_swing[2] = D("0.5") * D(np.sin(float(progress_t) * np.pi))

        self.set_swing_leg(start_swing_foot + update_swing)

    def update_walk_parameter(self):
        """
        Run only at start of new step, before walk pattern generation
        """
        x_speed, y_speed, a_speed = self.x_speed, self.y_speed, self.a_speed
        s_theta = self.s_theta
        y_offset = self.y_offset

        # Update new x walk parameter
        s_x = x_speed

        # Update new a (theta) walk parameter
        s_theta = wrap(s_theta[2] + a_speed, -D(np.pi), D(np.pi))

        # Update new y walk parameter
        s_y = y_offset * D("2.0")
        if s_y < y_offset * D("2.0"):
            s_y += y_speed
        else:
            s_y -= y_speed

        # Update walk parameter iteration
        self.s_x.pop(0)
        self.s_x.append(s_x)

        self.s_y.pop(0)
        self.s_y.append(s_y)

        self.s_theta.pop(0)
        self.s_theta.append(s_theta)

    def update_end_com_state(self):
        t_sup = self.t_sup

        com = self.analytical_real_time_com_state(t_sup)

        self.x_f = [com[0][0], self.x_f[0]]
        self.vx_f = [com[1][0], self.vx_f[0]]
        self.ax_f = [com[2][0], self.ax_f[0]]

        self.y_f = [com[0][1], self.y_f[0]]
        self.vy_f = [com[1][1], self.vy_f[0]]
        self.ay_f = [com[2][1], self.ay_f[0]]

    def single_support_phase_step(self, dt):
        if self.initial_ssp:
            self.initial_ssp = False

            self.update_walk_parameter()
            self.walk_pattern_gen()
            print()
            self.walk_pattern_gen()
            self.update_end_com_state()

            self.n += D("1.0")
            # self.print_info()

        self.t += dt

        self.move_swing_leg()

        if self.t >= self.t_sup * self.n:
            self.switch_support_leg()

            self.update_walk_parameter()
            self.walk_pattern_gen()
            self.update_end_com_state()

            if self.t_dbl != D("0.0"):
                self.state_double_support_phase = True

            self.n += D("1.0")

    def double_support_phase_step(self, dt):
        if self.initial_dsp:
            self.initial_dsp = False

        self.t_s += dt
        if self.t_s >= self.t_dbl * self.n:
            self.state_double_support_phase = False
            self.coeff_calc_this_step = False

    def step(self, dt):
        if self.initial:
            self.initial = False

            self.update_walk_parameter()

        if self.state_double_support_phase:
            self.double_support_phase_step(dt)
        else:
            self.single_support_phase_step(dt)

        self.calculate_real_time_com_state()

        reset_support_leg = self.get_support_leg()
        reset_support_leg[2] = D("0.0")
        self.set_support_leg(reset_support_leg)

    def print_info(self):
        print("------------------------------------------------------------")
        print(f"Time : {self.t}")
        print(f"At : {'SSP' if self.support_leg != 'both' else 'DSP'}")
        print(f"Step end : {self.t_sup * self.n}")
        print(f"Iteration : {self.n}")
        print(f"Support leg : {self.support_leg}")
        print(f"Left foot pos : {self.left_foot_pos}")
        print(f"Right foot pos : {self.right_foot_pos}")
        print(f"Current foot placement : {np.array([self.p_x[0], self.p_y[0]])}")
        print(f"Next foot placement : {np.array([self.mod_p_x[0], self.mod_p_y[0]])}")
        print(f"Walk primitive : {np.array([self.walk_x[0], self.walk_y[0]])}")
        print(
            f"Walk primitive velocity : {np.array([self.walk_vx[0], self.walk_vy[0]])}"
        )
        print(
            f"Initial COM state : {
                np.array(
                    [
                        self.x_i[0],
                        self.y_i[0],
                        self.vx_i[0],
                        self.vy_i[0],
                        self.ax_i[0],
                        self.ay_i[0],
                    ]
                )
            }"
        )
        print(
            f"Current COM state : {
                np.array(
                    [
                        self.x_t[0],
                        self.y_t[0],
                        self.vx_t[0],
                        self.vy_t[0],
                        self.ax_t[0],
                        self.ay_t[0],
                    ]
                )
            }"
        )
        print(
            f"Final COM state : {
                np.array(
                    [
                        self.x_f[0],
                        self.y_f[0],
                        self.vx_f[0],
                        self.vy_f[0],
                        self.ax_f[0],
                        self.ay_f[0],
                    ]
                )
            }"
        )
        print(
            f"Desired COM state : {
                np.array([self.x_d, self.y_d, self.vx_d, self.vy_d])
            }"
        )
        print(f"C : {self.c}")
        print(f"S : {self.s}")
        print(f"Tc : {self.t_c}")
        print(f"Progress t : {1 + (self.t - self.t_sup * self.n) / self.t_sup}")
        print("------------------------------------------------------------")
