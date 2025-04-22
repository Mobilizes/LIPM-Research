import numpy as np
import placo


class Walk:
    def __init__(
        self,
        opt={
            "dt": 30 * 0.001,
            "t_sup": 0.7,
            "t_dbl": 0.1,
            "t_mpc": 90 * 0.001,
            "left_foot": np.array([0.0, 0.2, 0.0, 0.0]),
            "right_foot": np.array([0.0, -0.2, 0.0, 0.0]),
            "foot_width": 0.25,
            "foot_height": 0.1,
            "z_c": 1.2,
            "horizon": 50,
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

        self.left_foot = [opt["left_foot"]]
        self.right_foot = [opt["right_foot"]]
        self.support_step = [opt["support_state"]]
        self.start_swing_foot = self.get_swing_foot()

        self.horizon = opt["horizon"]

        self.state_history = ["dsp"]
        self.state = "dsp"  # dsp | ssp

        self.speed = [opt["x_speed"], opt["y_speed"], opt["a_speed"]]

        self.com = [
            [np.array([(self.left_foot[0][0] + self.right_foot[0][1]) / 2.0, 0.0, 0.0])],
            [np.array([(self.left_foot[0][0] + self.right_foot[0][1]) / 2.0, 0.0, 0.0])],
        ]

        self.zmp = [
            [self.com[0][0][0] + self.z_c / 9.81 * self.com[0][0][2]],
            [self.com[1][0][0] + self.z_c / 9.81 * self.com[1][0][2]],
        ]

        self.initial = True
        self.initial_lipm = True

        self.update_p_ref()
        for _ in range(self.horizon):
            self.step()

    def get_support_foot(self, timestep=-1):
        return self.left_foot[timestep] if self.support_step[timestep] == "left" else self.right_foot[timestep]

    def get_swing_foot(self, timestep=-1):
        return self.left_foot[timestep] if self.support_step[timestep] == "right" else self.right_foot[timestep]

    def update_foot(self, swing_foot, pop=True, switch=False, timestep=-1):
        if self.support_step[timestep] == "right":
            self.left_foot.append(swing_foot)
            self.right_foot.append(self.get_support_foot())
        elif self.support_step[timestep] == "left":
            self.right_foot.append(swing_foot)
            self.left_foot.append(self.get_support_foot())
        else:
            raise Exception("Unknown support leg : ", self.support_step[timestep])

        if switch:
            self.support_step.append("left" if self.support_step[timestep] == "right" else "right")
        else:
            self.support_step.append(self.support_step[timestep])

        if pop:
            self.right_foot.pop(0)
            self.left_foot.pop(0)
            self.support_step.pop(0) 

    def update_p_ref(self):
        curr_p_ref = []
        for i in range(1, 1001):
            curr_p_ref.append([0.2 * i, 0.2 * (-1 if i % 2 == 0 else 1), 0, 0])

        self.p_ref = np.array(curr_p_ref)
        self.p_ref_step = [self.p_ref[0]]

    def update_mpc(self):
        problem = placo.Problem()
        lipm = placo.LIPM(
            problem,
            self.dt,
            self.horizon,
            0.0,
            np.array([self.com[0][0][0], self.com[1][0][0]]).reshape(2, 1),
            np.array([self.com[0][0][1], self.com[1][0][1]]).reshape(2, 1),
            np.array([self.com[0][0][2], self.com[1][0][2]]).reshape(2, 1)
        )

        for i in range(1, self.horizon + 1):
            support_foot = self.get_support_foot(i - 1)
            support_polygon = [
                np.array([support_foot[0] - self.foot_width / 2., support_foot[1] + self.foot_height / 2.]),
                np.array([support_foot[0] + self.foot_width / 2., support_foot[1] + self.foot_height / 2.]),
                np.array([support_foot[0] + self.foot_width / 2., support_foot[1] - self.foot_height / 2.]),
                np.array([support_foot[0] - self.foot_width / 2., support_foot[1] - self.foot_height / 2.]),
            ]

            # problem.add_constraint(
            #     placo.PolygonConstraint.in_polygon_xy(
            #         lipm.zmp(i, (9.81 / self.z_c) ** 2),
            #         support_polygon,
            #         0.05
            #     )
            # )

            problem.add_constraint(
                lipm.zmp(i, (9.81 / self.z_c) ** 2) == support_foot[:2]
            ).configure("soft", 1.)

        problem.solve()

        trajectory = lipm.get_trajectory()
        ts = np.linspace(0, self.dt * self.horizon, self.horizon)

        self.com = [[], []]
        self.zmp = [[], []]
        for t in ts:
            pos = trajectory.pos(t)
            vel = trajectory.vel(t)
            acc = trajectory.acc(t)
            self.com[0].append(np.array([pos[0], vel[0], acc[0]]))
            self.com[1].append(np.array([pos[1], vel[1], acc[1]]))

            zmp = trajectory.zmp(t, (9.81 / self.z_c) ** 2)
            self.zmp[0].append(zmp[0])
            self.zmp[1].append(zmp[1])

    def run_mpc(self):
        self.com[0].pop(0)
        self.com[1].pop(0)

        self.zmp[0].pop(0)
        self.zmp[1].pop(0)

    def run_ssp(self):
        if self.t_phase >= self.t_sup:
            self.state = "dsp"
            self.t_phase = 0.
            self.start_swing_foot = self.get_swing_foot()
            self.p_ref = np.delete(self.p_ref, 0, 0)
            # self.update_p_ref()
            return

        progress = min(1.0, self.t_phase / self.t_sup)

        curr_swing_foot = self.start_swing_foot + progress * (self.p_ref[0] - self.start_swing_foot)
        if progress <= 0.5:
            curr_swing_foot[2] = (progress / 0.5) * 0.2
        else:
            curr_swing_foot[2] = (1 - progress) / 0.5 * 0.2

        self.update_foot(
            curr_swing_foot, not self.initial, self.t_phase + self.dt >= self.t_sup
        )

        self.p_ref_step.append(self.p_ref[0])
        self.p_ref_step.pop(0) if not self.initial else None

        if not self.initial:
            self.run_mpc()

        self.t_phase += self.dt

    def run_dsp(self):
        if self.t_phase >= self.t_dbl:
            self.state = "ssp"
            self.t_phase = 0.
            return

        self.update_foot(self.get_swing_foot(), not self.initial)

        self.p_ref_step.append(self.p_ref[0])
        self.p_ref_step.pop(0) if not self.initial else None

        if not self.initial:
            self.run_mpc()

        self.t_phase += self.dt

    def step(self):
        if (
            len(self.left_foot) >= self.horizon
            and len(self.right_foot) >= self.horizon
            and self.initial
        ):
            self.initial = False

        self.t_preview += self.dt
        if self.t_preview >= self.t_mpc and not self.initial:
            self.update_mpc()
            self.t_preview = 0.

        if self.state == "dsp":
            self.run_dsp()
        if self.state == "ssp":
            self.run_ssp()
        if not any(state in self.state for state in ["ssp", "dsp"]):
            raise Exception("Unknown state : ", self.state)

