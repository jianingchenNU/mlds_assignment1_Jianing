"""Global seeding for python, numpy, torch, and (optionally) a gymnasium env."""
import random

import numpy as np
import torch


def set_seed(seed: int, env=None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if env is not None:
        # gymnasium seeds the env RNG only through reset(seed=...)
        env.reset(seed=seed)
        env.action_space.seed(seed)
