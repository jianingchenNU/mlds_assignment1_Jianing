"""Device selection shared by all training/eval scripts."""
import torch


def get_device(device: str | None = None, env: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    # the CartPole MLP is tiny; kernel launch overhead makes GPU slower than CPU
    if env == "cartpole":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
