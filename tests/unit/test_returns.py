import numpy as np
import torch

from pg.returns import discounted_returns, normalize


def test_hand_example():
    G = discounted_returns(np.array([1.0, 1.0, 1.0]), 0.5)
    np.testing.assert_allclose(G, [1.75, 1.5, 1.0])


def test_gamma_zero_and_one():
    rew = np.array([3.0, -1.0, 2.0])
    np.testing.assert_allclose(discounted_returns(rew, 0.0), rew)
    np.testing.assert_allclose(discounted_returns(rew, 1.0), [4.0, 1.0, 2.0])


def test_cartpole_constant_reward():
    # 1 + .95 + .95^2 + ... over n steps
    G = discounted_returns(np.ones(10), 0.95)
    np.testing.assert_allclose(G[0], (1 - 0.95**10) / 0.05)
    assert np.all(np.diff(G) < 0)


def test_pong_reset_on_score():
    rew = np.array([0.0, 0.0, 1.0, 0.0, -1.0])
    G = discounted_returns(rew, 0.5, reset_on_nonzero=True)
    np.testing.assert_allclose(G, [0.25, 0.5, 1.0, -0.5, -1.0])
    # without reset the later -1 leaks into the first rally
    G_no = discounted_returns(rew, 0.5)
    np.testing.assert_allclose(G_no, [0.1875, 0.375, 0.75, -0.5, -1.0])


def test_pong_rally_sign():
    # every step of a lost rally gets negative credit, every step of a won rally positive
    rew = np.array([0, 0, 0, -1, 0, 0, 1], dtype=float)
    G = discounted_returns(rew, 0.99, reset_on_nonzero=True)
    assert np.all(G[:4] < 0) and np.all(G[4:] > 0)


def test_single_step_and_empty():
    np.testing.assert_allclose(discounted_returns(np.array([5.0]), 0.9), [5.0])
    assert discounted_returns(np.array([]), 0.9).shape == (0,)


def test_normalize():
    x = normalize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert abs(x.mean()) < 1e-12 and abs(x.std() - 1) < 1e-6
    # constant returns (e.g. length-1 episode): no NaN
    np.testing.assert_allclose(normalize(np.ones(5)), 0.0)
    t = normalize(torch.tensor([1.0, 1.0]))
    assert torch.isfinite(t).all()
