import os
import time
from decimal import Decimal as D

import numpy as np

from LIPM_3D import LIPM3D
from visual import LIPM3D_Visual


def main():
    left_foot = np.array([D('0.0'), D('0.2'), D('0.0')])
    right_foot = np.array([D('0.0'), D('0.0'), D('0.0')])
    z_c = D('1.0')

    lipm = LIPM3D(left_foot, right_foot, z_c=z_c)
    visual = LIPM3D_Visual(lipm)

    lipm.walk_pattern_gen()
    visual.project_walk_pattern()


if __name__ == "__main__":
    main()
