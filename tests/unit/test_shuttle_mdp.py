import numpy as np

from mdp.shuttle_mdp import ShuttleMDP


def test_shapes():
    m = ShuttleMDP()
    assert m.cost.shape == (201, 2)
    assert m.P.shape == (2, 201, 201)


def test_rows_sum_to_one():
    m = ShuttleMDP()
    np.testing.assert_allclose(m.P.sum(-1), 1.0)
    assert (m.P >= 0).all()


def test_dispatch_cost_at_empty_station():
    m = ShuttleMDP()
    assert m.cost[0, 1] == 100
    assert m.cost[0, 0] == 0


def test_cost_formula():
    m = ShuttleMDP()
    assert m.cost[10, 0] == 20
    assert m.cost[10, 1] == 100  # everyone fits, no holding cost
    assert m.cost[20, 1] == 100 + 2 * 5
    assert m.cost[200, 0] == 400


def test_truncation_at_s_max():
    m = ShuttleMDP()
    # no dispatch from 200: every arrival overflows to 200
    assert m.P[0, 200, 200] == 1.0
    # from 198 arrivals 2..5 all land on 200
    np.testing.assert_allclose(m.P[0, 198, 199], 0.2)
    np.testing.assert_allclose(m.P[0, 198, 200], 0.8)
    # dispatch from 200 leaves 185 -> 186..190 uniformly
    np.testing.assert_allclose(m.P[1, 200, 186:191], 0.2)


def test_dispatch_transition():
    m = ShuttleMDP()
    np.testing.assert_allclose(m.P[1, 7, 1:6], 0.2)
    np.testing.assert_allclose(m.P[0, 7, 8:13], 0.2)
    assert m.P[0, 0, 0] == 0  # at least one person always arrives


def test_deterministic_arrivals():
    m = ShuttleMDP(arrivals={3: 1.0})
    assert m.P[0, 5, 8] == 1.0
    assert m.P[1, 20, 8] == 1.0


def test_q_values_matches_loop():
    m = ShuttleMDP()
    V = np.random.default_rng(0).normal(size=m.S)
    Q = m.q_values(V)
    for s in (0, 14, 15, 100, 200):
        for a in (0, 1):
            assert np.isclose(Q[s, a], m.cost[s, a] + m.gamma * m.P[a, s] @ V)
