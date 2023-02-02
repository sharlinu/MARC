"""
2D rendering of the level based foraging domain
"""

import math
import os
import sys

import numpy as np
import math
#import six
from gym import error
import pygame
import matplotlib.pyplot as plt
# # Define some colours


colours = {
    1: [255,255,255],  # gem colour
    2: [128,128,117],
    3: [255, 250, 200],
    4: [255, 216, 177],
      5: [250, 190, 190],
      6: [240, 50, 230],
      7: [145, 30, 180],
      8: [67, 99, 216],
      9: [66, 212, 244],
      10: [60, 180, 75],
      11: [191, 239, 69],
      12: [255, 255, 25],
      13: [245, 130, 49],
      14: [230, 25, 75],
      15: [128, 0, 0],
      16: [154, 99, 36],
      17: [128, 128, 0],
      18: [70, 153, 144],
    19: [230, 190, 255],
}


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        #display = get_display(None) # TODO that needs to be removed - just one display?
        self.rows, self.cols = world_size
        self.grid_size = 50

        self.width = self.cols * self.grid_size + 1
        self.height = self.rows * self.grid_size + 1

        pygame.init()  # initialize pygame
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.isopen = True

        script_dir = os.path.dirname(__file__)

        self.img_agent = pygame.image.load("../mnt/Images/agent.png")
        self.img_agent = pygame.transform.scale(self.img_agent, (self.grid_size, self.grid_size))

    def close(self):
        # TODO needs to be changed to take in right arguments
        for event in pygame.event.get():
            # common for GUI, if we find a quit event we exit the system
            if event.type == pygame.QUIT:
                sys.exit()

        #self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        ) # TODO this needs to be changed bc idk what Transform is (probs from pyglet)

    def colour_map(self, idx):
        return colours[idx]

    def render(self, env, return_rgb_array=False, cnt=0):

        self.screen.fill((220, 220, 220))  # fill background with colour
        #self._draw_grid()
        self.drawCells(env)
        self.drawPlayers(env)
        #self.drawFood(env)

        if return_rgb_array:
            # TODO get array in right shape
            #buffer = self.screen.tostring()
            pygame.image.save(self.screen, "../test.png")

        #return arr if return_rgb_array else self.isopen
        return self.isopen

    def drawCells(self, env):
        idxes = list(zip(*env.field.nonzero()))
        #print(idxes) # gives me index in right order for field
        for row, col in idxes:
            idx = env.field[row,col]
            colour = self.colour_map(idx)
            rect = pygame.Rect(self.grid_size * col, self.grid_size * row ,
                               self.grid_size, self.grid_size)
            pygame.draw.rect(self.screen, color=colour, rect=rect)
        pygame.display.update()


    def drawPlayers(self, env):
        players = []

        for player in env.players:
            row, col = player.position
            players.append(
                (self.img_agent,
                 (self.grid_size * col, self.grid_size * row ),
                )
            )

        for p in players:
            pygame.transform.scale(p[0],size=(5, 5))
            #p.update(scale=self.grid_size / p.width)
        self.screen.blits(players)
        pygame.display.update()  # update window
