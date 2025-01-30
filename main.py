import copy

from decimal import Decimal as D

import numpy as np

from LIPM_3D import LIPM3D
from visual import LIPM3D_Visual


def main():
    left_foot = np.array([D('0.0'), D('0.3'), D('0.0')])
    right_foot = np.array([D('0.0'), D('-0.3'), D('0.0')])
    z_c = D('1.0')
    x_speed = D('0.2')
    y_speed = D('1.2')
    a_speed = D('0.0')

    lipm = LIPM3D(left_foot, right_foot, z_c=z_c)
    visual = LIPM3D_Visual(lipm)

    lipm_history = []
    dt = D('0.1')
    for i in range(int(D('10.0') / dt)):
        lipm.set_walk_parameter((x_speed, y_speed, a_speed))
        lipm.step(dt)
        # lipm_history.append(copy.deepcopy(lipm))
        # if lipm.t % lipm.t_sup == D('0.0'):
        lipm_history.append(copy.deepcopy(lipm))

    for i in lipm_history:
        LIPM3D_Visual(i).project_walk_pattern()


if __name__ == "__main__":
    main()
