"""Pong preprocessing (verbatim from the assignment), frame differencing and action mapping."""
import numpy as np

RIGHT, LEFT = 2, 3


def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float32).ravel(), [80,80])


def encode(raw: np.ndarray, prev: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Network input = current frame - previous frame (zeros at episode start), so motion is visible."""
    # preprocess writes into slices of its argument; copy so the env's observation is left intact
    cur = preprocess(raw.copy())
    x = cur - (prev if prev is not None else np.zeros_like(cur))
    return x, cur


def to_env_action(a: int) -> int:
    """Policy output 1 -> RIGHT (2), 0 -> LEFT (3)."""
    return RIGHT if a == 1 else LEFT
