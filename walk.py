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
            "support_state": "right",
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

        self.left_foot = [opt["left_foot"]]
        self.right_foot = [opt["right_foot"]]
        self.support_state = [opt["support_state"]]

        self.start_swing_foot = self.get_swing_foot()

        self.preview_steps = opt["preview_steps"]

        self.state = "dsp"  # dsp | ssp

        self.speed = [opt["x_speed"], opt["y_speed"], opt["a_speed"]]

        self.initial = True
        self.initial_lipm = True
        self.update_p_ref()

    def get_support_foot(self):
        return self.left_foot[-1] if self.support_state[-1] == "left" else self.right_foot[-1]

    def get_swing_foot(self):
        return self.left_foot[-1] if self.support_state[-1] == "right" else self.right_foot[-1]

    def update_foot(self, swing_foot, pop=True, switch=False):
        if switch:
            self.support_state.append("left" if self.support_state[-1] == "right" else "right")
        else:
            self.support_state.append(self.support_state[-1])
        self.support_state.pop(0) if pop else None

        if self.support_state[-1] == "right":
            self.left_foot.append(swing_foot)
            self.left_foot.pop(0) if pop else None

            self.right_foot.append(self.get_support_foot())
            self.right_foot.pop(0) if pop else None
        elif self.support_state[-1] == "left":
            self.right_foot.append(swing_foot)
            self.right_foot.pop(0) if pop else None

            self.left_foot.append(self.get_support_foot())
            self.left_foot.pop(0) if pop else None
        else:
            raise Exception("Unknown support leg : ", self.support_state[-1])

    def update_p_ref(self):
        curr_p_ref = []
        for i in range(1, 1001):
            curr_p_ref.append([0.2 * i, 0.2 * (-1 if i % 2 == 0 else 1), 0, 0])

        self.p_ref = np.array(curr_p_ref)

    def update_mpc(self):
        problem = placo.Problem()
        # if self.initial_lipm:
        #     self.lipm = placo.LIPM(
        #         problem,
        #         self.dt,
        #         self.preview_steps,
        #         0.0,
        #         np.array([
        #             (self.left_foot[-1][0] + self.right_foot[-1][0]) / 2.0,
        #             (self.left_foot[-1][1] + self.right_foot[-1][1]) / 2.0,
        #         ]).reshape(2, 1),
        #         np.array([0., 0.]).reshape(2, 1),
        #         np.array([0., 0.]).reshape(2, 1),
        #     )
        #     self.initial_lipm = False
        # else: self.lipm = placo.LIPM.build_LIPM_from_previous(problem, self.dt, self.preview_steps, 0., self.lipm)
        #
        # support_foot = self.get_support_foot()
        # support_polygon = np.array([
        #     [support_foot[0] - self.foot_width / 2., support_foot[1] + self.foot_height / 2.],
        #     [support_foot[0] + self.foot_width / 2., support_foot[1] + self.foot_height / 2.],
        #     [support_foot[0] + self.foot_width / 2., support_foot[1] - self.foot_height / 2.],
        #     [support_foot[0] - self.foot_width / 2., support_foot[1] - self.foot_height / 2.],
        # ])

    def run_ssp(self):
        if self.t_phase >= self.t_sup:
            self.state = "dsp"
            self.t_phase = 0.
            self.start_swing_foot = self.get_swing_foot()
            self.p_ref = np.delete(self.p_ref, 0, 0)
            # self.update_p_ref()
            return

        progress = min(1.0, self.t_phase / self.t_sup)

        curr_swing_foot = (1 - progress) * self.start_swing_foot + progress * self.p_ref[0]
        if progress <= 0.5:
            curr_swing_foot[2] = (progress / 0.5) * 0.2
        else:
            curr_swing_foot[2] = (1 - progress) / 0.5 * 0.2

        self.update_foot(
            curr_swing_foot, not self.initial, self.t_phase + self.dt >= self.t_sup
        )

        self.t_phase += self.dt

    def run_dsp(self):
        if self.t_phase >= self.t_dbl:
            self.state = "ssp"
            self.t_phase = 0.
            return

        self.update_foot(self.get_swing_foot(), not self.initial)

        self.t_phase += self.dt

    def step(self):
        if (
            len(self.left_foot) >= self.preview_steps
            and len(self.right_foot) >= self.preview_steps
        ):
            self.initial = False

        self.t_preview += self.dt * (1 if not self.initial else 0)
        if self.t_preview >= self.t_mpc:
            self.update_mpc()
            self.t_preview = 0.

        if self.state == "dsp":
            self.run_dsp()
        if self.state == "ssp":
            self.run_ssp()
        if not any(state in self.state for state in ["ssp", "dsp"]):
            raise Exception("Unknown state : ", self.state)

