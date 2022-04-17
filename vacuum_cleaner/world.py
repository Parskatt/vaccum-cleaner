import numpy as np
from .defs import clean, dirty, obstacle


class World:
    def __init__(self, state: np.ndarray) -> None:
        self.state = state
        self.H, self.W = state.shape

    def has_dirt(self):
        return (self.state == dirty).any()

    def no_obstacle(self, pos):
        if self.inside_bounds(pos):
            return not self.state[pos[0], pos[1]] == obstacle
        else:
            return False

    def inside_bounds(self, pos):
        return (0 <= pos[0] < self.H) & (0 <= pos[1] < self.W)

    def __getitem__(self, idx1, idx2):
        return self.state[idx1, idx2]

    def __setitem__(self, *args, **kwargs):
        self.state.__setitem__(*args)

    def __eq__(self, o):
        return self.state == o

    def render(self, agent):
        rendition = np.zeros((self.H, self.W, 3))
        rendition[self.state == clean] = (248 / 255, 200 / 255, 220 / 255)
        rendition[self.state == dirty] = (0.6, 0.6, 0.6)
        rendition[self.state == obstacle] = (137 / 255, 207 / 255, 240 / 255)
        rendition[agent.pos[0], agent.pos[1]] = 0.3 * rendition[
            agent.pos[0], agent.pos[1]
        ] + 0.7 * np.array((0.5, 0.9, 0.5))
        return rendition

    def __str__(self):
        return self.state.__str__()
