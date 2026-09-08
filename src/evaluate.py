"""
    Score every saved checkpoint on the AUTSL test split.

    Run from the project root:
        python -m src.evaluate
"""

import csv
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import AUTSLDataset
from src.x3d import X3D


# Report order, smallest variant first
ORDER = {"XS": 0, "S": 1, "M": 2}


def main(
    checkpoint_dir = "checkpoints",
    results_dir = "results",
    batch_size = 8,
    num_workers = 4,
):
    """
        Evaluate each checkpoint and write results/evaluation.csv
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders = {}

    def test_loader(model_size):
        # One loader per variant, since each expects its own clip geometry
        if model_size not in loaders:
            num_frames, crop_size = X3D.geometry(model_size)

            loaders[model_size] = DataLoader(
                AUTSLDataset(
                    "data/test_labels.csv",
                    "data/test",
                    num_frames = num_frames,
                    crop_size = crop_size,
                ),
                batch_size = batch_size,
                shuffle = False,
                num_workers = num_workers,
                persistent_workers = True,
            )

        return loaders[model_size]

    rows = []

    for path in sorted(Path(checkpoint_dir).glob("x3d_*_freeze*.pt")):
        size, frozen = re.match(r"x3d_(.+)_freeze(\d+)", path.stem).groups()
        size = size.upper()

        # pretrained = False, since load() overwrites every weight anyway
        clf = X3D(model_size = size, pretrained = False, num_classes = 226, device = device)
        clf.load(path)

        clf.freeze(int(frozen))

        metrics = clf.evaluate(test_loader(size))
        rows.append({"model": size, "frozen": int(frozen), **metrics})

        print(f"{size:>2} freeze{frozen}  top1 {metrics['top1']:.4f}", flush = True)

    if not rows:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    rows.sort(key = lambda row: (ORDER[row["model"]], row["frozen"]))

    results = Path(results_dir) / "evaluation.csv"
    results.parent.mkdir(parents = True, exist_ok = True)

    with open(results, "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {results}")

    return rows


if __name__ == "__main__":
    main(
        checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints"),
        results_dir = os.environ.get("RESULTS_DIR", "results"),
        batch_size = int(os.environ.get("BATCH_SIZE", 8)),
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
    )
