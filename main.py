import os
import time
import pygame
from pygame.locals import K_w, K_a, K_s, K_d, K_q, K_e, KEYDOWN

import numpy as np
import matplotlib as plt

from LIPM_3D import LIPM3D


def get_input(input):
    input = [0.0, 0.0, 0.0]
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            exit()
        if e.type == KEYDOWN:
            if e.key == K_w:
                input[0] += 0.2
            if e.key == K_s:
                input[0] -= 0.2
            if e.key == K_a:
                input[1] += 0.2
            if e.key == K_d:
                input[1] -= 0.2
            if e.key == K_q:
                input[2] += np.pi / 4.0
            if e.key == K_e:
                input[2] -= np.pi / 4.0
    return input


def main():
    pygame.init()
    size = width, height = 740, 480
    screen = pygame.display.set_mode(size)

    input_queue = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    left_foot_pos = np.array([-0.2, 0.2, 0.0])
    right_foot_pos = np.array([0.2, -0.2, 0.0])

    lipm_model = LIPM3D(left_foot_pos, right_foot_pos, z_c=1.0)
    lipm_model.print_info()

    while True:
        new_input = get_input(input_queue)
        if new_input != [0.0, 0.0, 0.0]:
            input_queue[0] = input_queue[1]
            input_queue[1] = new_input

            lipm_model.walk_pattern_gen(
                input_queue[0][0],
                input_queue[0][1],
                input_queue[0][2],
                input_queue[1][0],
                input_queue[1][1],
                input_queue[1][2],
            )
            print(input_queue)
            lipm_model.print_info()


if __name__ == "__main__":
    main()
