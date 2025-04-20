import numpy as np
import placo


class Walk:
    def __init__(
        self,
        opt={
            "dt": 30 * 0.001,
            "t_sup": 0.7,
            "t_dbl": 0.0,
            "t_mpc": 90 * 0.001,
            "left_foot": np.array([0.0, 0.2, 0.0, 0.0]),
            "right_foot": np.array([0.0, -0.2, 0.0, 0.0]),
            "foot_width": 0.25,
            "foot_height": 0.1,
            "z_c": 1.2,
            "preview_steps": 10,
            "support_leg": "right",
            "x_speed": 2.0,
            "y_speed": 0.0,
            "a_speed": 0.0,
        },
    ) -> None:
        self.dt = opt["dt"]
        self.t_sup = opt["t_sup"]
        self.t_dbl = opt["t_dbl"]
        self.t_mpc = opt["t_mpc"]
        self.t_phase = 0.0
        self.t_preview = 0.0

        self.foot_width = opt["foot_width"]
        self.foot_height = opt["foot_height"]

        self.z_c = opt["z_c"]

        self.A = np.array(
            [[1.0, self.dt, self.dt**2 / 2.0], [0.0, 1.0, self.dt], [0.0, 0.0, 1.0]]
        )
        self.b = np.array([self.dt**3 / 3.0, self.dt**2 / 2.0, self.dt])[np.newaxis].T
        self.c = np.array([1.0, 0.0, -self.z_c / 9.81])[np.newaxis]

        self.left_foot = opt["left_foot"]
        self.right_foot = opt["right_foot"]
        self.support_leg = opt["support_leg"]

        self.start_swing_foot = (
            self.left_foot if self.support_leg == "right" else self.right_foot
        )

        self.preview_steps = opt["preview_steps"]

        self.state = "dsp"  # dsp | ssp

        self.speed = [opt["x_speed"], opt["y_speed"], opt["a_speed"]]

        self.lipm_init = True
        self.update_p_ref()

    def get_support_foot(self):
        return self.left_foot if self.support_leg == "left" else self.right_foot

    def get_swing_foot(self):
        return self.left_foot if self.support_leg == "right" else self.right_foot

    def set_swing_foot(self, foot):
        if self.support_leg == "right":
            self.left_foot = foot
        elif self.support_leg == "left":
            self.right_foot = foot
        else:
            raise Exception("Unknown support leg : ", self.support_leg)

    def update_p_ref(self):
        curr_p_ref = []
        for i in range(1, 1001):
            curr_p_ref.append([0.2 * i, 0.2 * (-1 if i % 2 == 0 else 1), 0, 0])

        self.p_ref = np.array(curr_p_ref)

    def update_mpc(self):
        problem = placo.Problem()
        if self.lipm_init:
            self.lipm = placo.LIPM(
                problem,
                self.dt,
                self.preview_steps,
                0.0,
                np.array([
                    (self.left_foot[0] + self.right_foot[0]) / 2.0,
                    (self.left_foot[1] + self.right_foot[1]) / 2.0,
                ]),
                np.array([0., 0.]),
                np.array([0., 0.]),
            )
        else:
            self.lipm = placo.LIPM.build_LIPM_from_previous(problem, self.dt, self.preview_steps, 0., self.lipm)

        support_foot = self.get_support_foot()
        support_polygon = np.array([
            [support_foot[0] - self.foot_width / 2., support_foot[1] + self.foot_height / 2.],
            [support_foot[0] + self.foot_width / 2., support_foot[1] + self.foot_height / 2.],
            [support_foot[0] + self.foot_width / 2., support_foot[1] - self.foot_height / 2.],
            [support_foot[0] - self.foot_width / 2., support_foot[1] - self.foot_height / 2.],
        ])

        self.t_preview = 0.

    def run_ssp(self):
        progress = min(1.0, self.t_phase / self.t_sup)

        curr_swing_foot = (1 - progress) * self.start_swing_foot + progress * self.p_ref[0]
        if progress <= 0.5:
            curr_swing_foot[2] = (progress / 0.5) * 0.2
        else:
            curr_swing_foot[2] = (1 - progress) / 0.5 * 0.2

        self.set_swing_foot(curr_swing_foot)

        self.t_phase += self.dt
        if self.t_phase >= self.t_sup:
            self.state = "dsp"
            self.t_phase = 0.
            self.support_leg = "left" if self.support_leg == "right" else "right"
            self.start_swing_foot = self.get_swing_foot()
            self.p_ref = np.delete(self.p_ref, 0, 0)
            # self.update_p_ref()

    def run_dsp(self):
        self.t_phase += self.dt
        if self.t_phase >= self.t_sup:
            self.state = "ssp"
            self.t_phase = 0.

    def step(self):
        self.t_preview += self.dt
        if self.t_preview >= self.t_mpc:
            self.update_mpc()

        if self.state == "ssp":
            self.run_ssp()
        elif self.state == "dsp":
            self.run_dsp()
        else:
            raise Exception("Unknown state : ", self.state)

