import copy
import math
import itertools

from decimal import Decimal as D

import numpy as np

from LIPM_3D import LIPM3D
from visual import LIPM3D_Visual


def main():
    left_foot = np.array([D('0.0'), D('-0.1'), D('0.0')])
    right_foot = np.array([D('0.0'), D('0.1'), D('0.0')])
    t_sup = D('0.5')
    z_c = D('0.8')
    x_speed = D('0.1')
    y_speed = D('0.1')
    a_speed = D('0.1')
    a = D('1.0')
    b = D('1.0')

    lipm = LIPM3D(
        left_foot,
        right_foot,
        t_sup=t_sup,
        z_c=z_c,
        y_offset=D('0.0'),
        support_leg="left",
        a=a,
        b=b)
    visual = LIPM3D_Visual(lipm)

    def cycle():
        dist = math.sqrt(float(lipm.right_foot_pos[0]-lipm.left_foot_pos[0])**2 + float(lipm.right_foot_pos[1] - lipm.left_foot_pos[1])**2)
        constant = D('1.0')
        test = [-y_speed * constant, y_speed * constant]
        for i in itertools.cycle(test):
            yield i

    lipm_history = []
    dt = D('0.1')
    max_steps = 50
    cycle_gen = cycle()
    lipm.set_walk_parameter(
        (x_speed, lipm.y_offset * 2 + next(cycle_gen), a_speed))
    while lipm.n < max_steps:
        n = lipm.n
        lipm.step(dt)
        # lipm_history.append(copy.deepcopy(lipm))
        # if lipm.t % lipm.t_sup == D('0.0'):
        if lipm.n != n:
            lipm.set_walk_parameter(
                (x_speed, lipm.y_offset * 2 + next(cycle_gen), a_speed))
        lipm_history.append(copy.deepcopy(lipm))

    x_t_history = []
    for i in lipm_history:
        LIPM3D_Visual(i).project_walk_pattern(x_t_history, 2)


if __name__ == "__main__":
    main()
