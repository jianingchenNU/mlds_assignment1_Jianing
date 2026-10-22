import ale_py
import gymnasium as gym
import numpy as np
import pytest

from pg.pong_utils import encode, preprocess, to_env_action

gym.register_envs(ale_py)


@pytest.fixture(scope="module")
def env():
    e = gym.make("ALE/Pong-v5")
    yield e
    e.close()


@pytest.fixture(scope="module")
def frames(env):
    raw, _ = env.reset(seed=0)
    out = [raw]
    for _ in range(30):
        raw, *_ = env.step(0)
        out.append(raw)
    return out


def test_preprocess_shape_and_values(frames):
    x = preprocess(frames[-1].copy())
    assert x.shape == (80, 80) and x.dtype == np.float32
    assert set(np.unique(x)) <= {0.0, 1.0}
    assert 0 < x.sum() < 0.1 * x.size  # paddles/ball survive, background is erased


def test_preprocess_synthetic():
    img = np.full((210, 160, 3), 144, dtype=np.uint8)
    img[101, 80, 0] = 236              # inside the crop, on a kept (even) row after cropping
    img[10, 10, 0] = 236               # score area, cropped away
    x = preprocess(img)
    assert x.sum() == 1 and x[(101 - 35) // 2, 40] == 1


def test_encode_does_not_mutate_obs(frames):
    raw = frames[-1]
    before = raw.copy()
    encode(raw, None)
    np.testing.assert_array_equal(raw, before)


def test_frame_difference(frames):
    a, b = frames[-2], frames[-1]
    x0, prev = encode(a, None)
    np.testing.assert_array_equal(x0, preprocess(a.copy()))
    x1, _ = encode(b, prev)
    np.testing.assert_array_equal(x1, preprocess(b.copy()) - preprocess(a.copy()))
    assert set(np.unique(x1)) <= {-1.0, 0.0, 1.0}


def test_action_mapping(env):
    names = env.unwrapped.get_action_meanings()
    assert names[to_env_action(1)] == "RIGHT"
    assert names[to_env_action(0)] == "LEFT"
    assert env.action_space.n == 6
