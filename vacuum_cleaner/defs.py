import numpy as np

left, right, up, down = (
    np.array((0, -1)),
    np.array((0, 1)),
    np.array((-1, 0)),
    np.array((1, 0)),
)
clean, dirty, obstacle = 0, 1, 2
