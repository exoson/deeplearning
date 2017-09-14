import numpy as np
import cv2
from collections import deque


class Flappy:
    COLORS = {
        'bird': (1, 1, 1),
        'pillar': (1, 1, 1),
        'ground': (1, 1, 1),
    }
    BIRD_SIZE = 15
    BIRD_START = 0.5
    PILLAR_HOLE = 100
    PILLAR_WIDTH = 25
    PILLAR_DEL = 50
    JUMP_SPEED = -10
    ACC = 1
    VEL = 3

    def __init__(self, screen_size=(300, 400)):
        self.screen_size = screen_size
        self.reset()

    def reset(self):
        """Reset game."""
        self.bird_pos = [50, int(self.screen_size[0] * self.BIRD_START)]
        self.pillars = deque()
        self.pillar_time = 0
        self.vel = self.JUMP_SPEED

    def _add_pillar(self):
        """Add a new pillar with random height."""
        offset = self.screen_size[0] / 10
        pos = [self.screen_size[1],
               np.random.random_integers(offset, self.screen_size[0] - self.PILLAR_HOLE - offset)]
        self.pillars.append(pos)

    def update(self, pin):
        """
        Update game with certain input.

        Args:
          pin: player in (true if player jumped)

        Returns: True if bird died
        """
        self.vel += self.ACC
        self.bird_pos[1] += self.vel

        self.pillar_time += 1
        if self.pillar_time >  self.PILLAR_DEL:
            self._add_pillar()
            self.pillar_time = 0

        if pin:
            self.vel = self.JUMP_SPEED

        if self.bird_pos[1] + self.BIRD_SIZE > self.screen_size[0]:
            self.reset()

        bx, by = self.bird_pos
        for pillar in self.pillars:
            pillar[0] -= self.VEL
            px, py = pillar
            collided = (by < py or (by + self.BIRD_SIZE) > (py + self.PILLAR_HOLE)) and \
                (bx + self.BIRD_SIZE) > px and bx < (px + self.PILLAR_WIDTH)
            if collided:
                self.reset()

        if len(self.pillars) > 0 and self.pillars[0][0] + self.PILLAR_WIDTH < 0:
            self.pillars.popleft()

    def _draw_square(self, x, y, width, height, color=(1, 1, 1)):
        """
        Draw a square in the image.

        Args:
          x: x coordinate
          y: y coordinate
          width: width of square
          height: height of square
        """
        cv2.rectangle(self.img, (x, y), (x + width, y + height), color=color, thickness=-1)

    def render(self, on_screen=False):
        """
        Render game and return player input.

        Args:
          on_screen: whether to show the rendered image on screen

        Returns: True if space was pressed and image is shown on screen
        """
        self.img = np.zeros(self.screen_size)

        for pillar in self.pillars:
            self._draw_square(pillar[0],
                              0,
                              self.PILLAR_WIDTH,
                              pillar[1],
                              self.COLORS['pillar'])
            self._draw_square(pillar[0],
                              pillar[1] + self.PILLAR_HOLE,
                              self.PILLAR_WIDTH,
                              self.screen_size[1],
                              self.COLORS['pillar'])

        self._draw_square(self.bird_pos[0], self.bird_pos[1], self.BIRD_SIZE, self.BIRD_SIZE)

        if on_screen:
            cv2.imshow('Flappy bird', self.img)
            key = cv2.waitKey(16)
            if key == 32:  # Space
                return True

if __name__ == '__main__':
    game = Flappy()
    while True:
        jumped = game.render(True)
        game.update(jumped)
