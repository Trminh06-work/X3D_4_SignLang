"""
    Fine-tune one X3D variant on AUTSL, then save its weights and training curve.

    Run from the project root:
        python -m src.main
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import AUTSLDataset
from src.x3d import X3D


def main(
    model_size = "XS",
    frozen = 5,
    epochs = 5,
    batch_size = 8,
    learning_rate = 1e-3,
    subset = None,
    num_workers = 4,
    cache_dir = None,
    checkpoint_dir = "checkpoints",
    figure_dir = "figures",
):
    """
        Train one variant and write its state_dict and training curve

        Returns the fitted X3D so a notebook can keep using it.
    """
    # The variant dictates the clip geometry, and raises if it is unknown
    num_frames, crop_size = X3D.geometry(model_size)

    torch.manual_seed(0)

    def build(split, train):
        data = AUTSLDataset(
            f"data/{split}_labels.csv",
            f"data/{split}",
            num_frames = num_frames,
            crop_size = crop_size,
            train = train,
            subset = subset,
            cache_dir = cache_dir,
            num_workers = num_workers,
        )

        # Reading the cache is the bottleneck, so give it every allocated CPU
        return DataLoader(
            data,
            batch_size = batch_size,
            shuffle = train,
            num_workers = num_workers,
            persistent_workers = True,
        )

    loader = build("train", train = True)
    val_loader = build("val", train = False)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(
        f"X3D-{model_size} | {frozen} blocks frozen | "
        f"{num_frames} frames at {crop_size}px | "
        f"{len(loader.dataset)} train / {len(val_loader.dataset)} val "
        f"clips on {device}",
        flush = True,
    )

    clf = X3D(
        model_size = model_size,
        pretrained = True,
        num_classes = 226,
        device = device,
    )
    clf.freeze(frozen)

    name = f"x3d_{model_size.lower()}_freeze{frozen}"

    checkpoint = Path(checkpoint_dir) / f"{name}.pt"
    figure = Path(figure_dir) / f"{name}.png"

    checkpoint.parent.mkdir(parents = True, exist_ok = True)
    figure.parent.mkdir(parents = True, exist_ok = True)

    # Validation each epoch, so the saved curve can show overfitting
    clf.fit(
        loader,
        epochs = epochs,
        lr = learning_rate,
        val_loader = val_loader,
    )

    # Over 100 epochs the final weights are rarely the best ones
    clf.save(checkpoint, best = True)
    clf.plot(figure)

    print(f"saved {checkpoint} and {figure}")

    return clf


if __name__ == "__main__":
    # Compute nodes have no display, so render straight to file
    import matplotlib
    matplotlib.use("Agg")

    subset = os.environ.get("SUBSET", "")

    main(
        model_size = os.environ.get("SIZE", "XS"),
        frozen = int(os.environ.get("FROZEN", 5)),
        epochs = int(os.environ.get("EPOCHS", 5)),
        batch_size = int(os.environ.get("BATCH_SIZE", 8)),
        learning_rate = float(os.environ.get("LEARNING_RATE", 1e-3)),
        subset = int(subset) if subset else None,
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
        cache_dir = os.environ.get("CACHE_DIR") or None,
        checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints"),
        figure_dir = os.environ.get("FIGURE_DIR", "figures"),
    )
