"""Append-only per-episode CSV log (append mode keeps --resume runs in one file)."""
import csv
from pathlib import Path

COLUMNS = ["episode", "reward", "length", "loss", "entropy", "grad_norm", "adv_var", "value_loss"]


def log_episode(path: str | Path, row: dict) -> None:
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})
