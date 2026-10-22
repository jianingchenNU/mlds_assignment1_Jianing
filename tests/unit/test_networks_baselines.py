import torch

from pg.baselines import MeanBaseline, NoBaseline, ValueBaseline, advantage
from pg.networks import CartPolePolicy, PongCNN, PongPolicy

CPU = torch.device("cpu")


def test_cartpole_policy():
    d = CartPolePolicy()(torch.randn(7, 4))
    assert d.probs.shape == (7, 2)
    torch.testing.assert_close(d.probs.sum(-1), torch.ones(7))
    assert d.sample().shape == (7,)


def test_pong_policies():
    x = torch.randn(3, 80, 80)
    for net in (PongPolicy(), PongCNN()):
        d = net(x)
        assert d.probs.shape == (3,)
        assert ((d.probs > 0) & (d.probs < 1)).all()
        assert set(d.sample().tolist()) <= {0.0, 1.0}


def test_no_baseline():
    G = torch.randn(5)
    assert (NoBaseline()(None, G) == 0).all()


def test_mean_baseline_uses_past_only():
    b = MeanBaseline(M=2)
    assert (b(None, torch.ones(3)) == 0).all()
    b.update(None, torch.tensor([1.0, 3.0]))          # mean 2
    b.update(None, torch.tensor([4.0, 4.0, 4.0, 4.0]))  # pooled: 20 / 6
    torch.testing.assert_close(b(None, torch.zeros(2)), torch.full((2,), 20 / 6))
    b.update(None, torch.tensor([0.0]))  # oldest episode drops out: 16 / 5
    torch.testing.assert_close(b(None, torch.zeros(1)), torch.full((1,), 16 / 5))


def test_value_baseline_detached():
    torch.manual_seed(0)
    pol = CartPolePolicy()
    vb = ValueBaseline(4, 32, 1e-2, CPU)
    obs, G = torch.randn(10, 4), torch.randn(10)
    act = pol(obs).sample()
    adv, _ = advantage(vb, obs, G, norm=True)
    loss = -(pol(obs).log_prob(act) * adv).sum()
    loss.backward()
    assert all(p.grad is None for p in vb.net.parameters())
    assert all(p.grad is not None for p in pol.parameters())


def test_value_baseline_fits():
    torch.manual_seed(0)
    vb = ValueBaseline(4, 32, 1e-2, CPU)
    obs = torch.randn(64, 4)
    G = obs[:, 0] * 3 + 1
    first = vb.update(obs, G)
    for _ in range(300):
        last = vb.update(obs, G)
    assert last < 0.05 * first


def test_mean_baseline_cancelled_by_normalization():
    b = MeanBaseline()
    b.update(None, torch.tensor([10.0, 20.0]))
    G = torch.tensor([1.0, 5.0, 2.0])
    torch.testing.assert_close(advantage(b, None, G, True)[0], advantage(NoBaseline(), None, G, True)[0])
