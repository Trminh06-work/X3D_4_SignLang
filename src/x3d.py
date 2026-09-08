from copy import deepcopy

import torch
from torch import nn
from pytorchvideo.models.hub import x3d_xs, x3d_s, x3d_m

class X3D():
    # Builder, clip length and crop size each variant was pretrained on.
    VARIANTS = {
        "XS": (x3d_xs, 4, 160),
        "S": (x3d_s, 13, 160),
        "M": (x3d_m, 16, 224),
    }


    @classmethod
    def geometry(cls, model_size: str):
        """
            Clip length and crop size the variant expects, as (frames, crop)
        """
        if model_size not in cls.VARIANTS:
            raise ValueError(
                f"The model size is invalid, must be {list(cls.VARIANTS)}"
            )

        _, num_frames, crop_size = cls.VARIANTS[model_size]

        return num_frames, crop_size

    def __init__(
        self,
        model_size: str = "S",
        pretrained: bool = None,
        num_classes: int = 226,
        device: str = "cpu",
    ):
        if pretrained is None:
            raise ValueError("pretrained is not specified")

        # Raises if the size is unknown
        self.num_frames, self.crop_size = self.geometry(model_size)

        self.device = device

        # Backbone carries Kinetics-400 weights when pretrained = True
        builder, _, _ = self.VARIANTS[model_size]
        self.model = builder(pretrained = pretrained)

        # Replace the Kinetics-400 head with one sized for our classes
        head = self.model.blocks[-1].proj
        self.model.blocks[-1].proj = nn.Linear(head.in_features, num_classes)

        self.model.to(device)

        # Blocks held fixed by freeze(), consulted in fit()
        self.frozen = []

        # Per-epoch curves, filled by fit() and drawn by plot()
        self.history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

        # Highest-scoring epoch on the validation split, kept by fit()
        self.best = {"acc": -1.0, "epoch": 0, "state": None}


    def freeze(self, num_blocks: int):
        """
            Freeze the first num_blocks blocks, leaving the rest trainable

            X3D has 6 blocks: 0 is the stem, 1-4 the residual stages, 5 the head.
            freeze(5) trains the new head alone, freeze(0) trains everything.
        """
        self.frozen = self.model.blocks[:num_blocks]

        for block in self.frozen:
            block.requires_grad_(False)


    def fit(
        self,
        loader,
        epochs: int = 5,
        lr: float = 1e-3,
        val_loader = None,
    ):
        """
            Fit the current model using the given training data
        """
        # Standard multi-class classification setup
        criterion = nn.CrossEntropyLoss()

        # Hand the optimiser only what freeze() left trainable
        optimiser = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr = lr
        )

        self.model.train()

        # Frozen blocks stay in eval mode so their BatchNorm stats do not drift
        for block in self.frozen:
            block.eval()

        for epoch in range(epochs):
            # Running totals for the epoch summary
            total_loss, correct, seen = 0.0, 0, 0

            for videos, labels in loader:
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(videos)
                loss = criterion(logits, labels)

                # Backward pass and weight update
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Weight by batch size so the averages are per sample, not per batch
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim = 1) == labels).sum().item()
                seen += labels.size(0)

            self.history["loss"].append(total_loss / seen)
            self.history["acc"].append(correct / seen)

            summary = (
                f"epoch {epoch + 1}/{epochs}  "
                f"loss {total_loss / seen:.4f}  "
                f"acc {correct / seen:.4f}"
            )

            if val_loader is not None:
                val_loss, val_acc = self._score(val_loader, criterion)

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                summary += f"  val loss {val_loss:.4f}  val acc {val_acc:.4f}"

                # A long run drifts past its best epoch, so keep that one
                if val_acc > self.best["acc"]:
                    self.best = {
                        "acc": val_acc,
                        "epoch": epoch + 1,
                        "state": deepcopy(self.model.state_dict()),
                    }

            print(summary, flush = True)


    @torch.no_grad()
    def _score(self, loader, criterion):
        """
            Average loss and accuracy over a split, without updating weights
        """
        self.model.eval()

        total_loss, correct, seen = 0.0, 0, 0

        for videos, labels in loader:
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(videos)

            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim = 1) == labels).sum().item()
            seen += labels.size(0)

        # Hand the model back the way fit() had it
        self.model.train()

        for block in self.frozen:
            block.eval()

        return total_loss / seen, correct / seen


    def plot(self, path = None):
        """
            Draw the per-epoch loss and accuracy recorded by fit()
        """
        # Imported here so training does not depend on matplotlib
        import matplotlib.pyplot as plt

        if not self.history["loss"]:
            raise ValueError("Nothing to plot, call fit() first")

        ink, muted, grid = "#0b0b0b", "#52514e", "#e8e7e3"

        epochs = list(range(1, len(self.history["loss"]) + 1))

        # A tick per epoch is unreadable much past a dozen, and the grid follows
        # the ticks, so thin them to a round step and keep the first and last
        step = max(1, round(len(epochs) / 10))
        ticks = [e for e in epochs if e == 1 or e % step == 0]

        # Marking every point turns a long run into a caterpillar
        marker = "o" if len(epochs) <= 30 else None

        figure, axes = plt.subplots(1, 2, figsize = (10, 4))
        figure.patch.set_facecolor("#fcfcfb")

        # Colour names the split, so it means the same thing in both panels
        panels = (
            (axes[0], "loss", "loss", "val_loss"),
            (axes[1], "accuracy", "acc", "val_acc"),
        )

        for panel, title, train_key, val_key in panels:
            lines = [("train", self.history[train_key], "#2a78d6")]

            if self.history[val_key]:
                lines.append(("validation", self.history[val_key], "#eb6834"))

            # Label the higher endpoint above and the other below, so two
            # values at the same epoch never land on top of each other
            highest = max(range(len(lines)), key = lambda i: lines[i][1][-1])

            for index, (label, values, colour) in enumerate(lines):
                panel.plot(
                    epochs, values,
                    label = label,
                    color = colour,
                    linewidth = 2,
                    solid_capstyle = "round",
                    marker = marker,
                    markersize = 4.5,
                )

                # Endpoints only, in ink rather than the mark colour
                panel.annotate(
                    f"{values[-1]:.3f}",
                    (epochs[-1], values[-1]),
                    textcoords = "offset points",
                    xytext = (0, 10 if index == highest else -18),
                    ha = "center",
                    color = ink,
                    fontsize = 9,
                )

            # Headroom so the endpoint labels are never clipped by the panel edge
            panel.margins(x = 0.12, y = 0.2)

            panel.set_title(title, color = ink, fontsize = 11, loc = "left")
            panel.set_xlabel("epoch", color = muted, fontsize = 9)
            panel.set_xticks(ticks)

            # Recessive chrome: solid hairline grid, no box around the plot
            panel.grid(True, color = grid, linewidth = 0.8)
            panel.set_axisbelow(True)
            panel.tick_params(colors = muted, labelsize = 9)

            for side in ("top", "right"):
                panel.spines[side].set_visible(False)

            for side in ("left", "bottom"):
                panel.spines[side].set_color(grid)

        # Two series need a legend; a lone one is named by the panel title
        if self.history["val_loss"]:
            axes[0].legend(frameon = False, fontsize = 9, labelcolor = ink)

        figure.tight_layout()

        if path:
            figure.savefig(path, dpi = 150, facecolor = figure.get_facecolor())

        return figure


    @torch.no_grad()
    def predict(self, loader):
        """
            Give prediction
        """
        self.model.eval()

        # Highest scoring class per clip, collected batch by batch
        preds = [
            self.model(videos.to(self.device)).argmax(dim=1).cpu()
            for videos, _ in loader
        ]

        return torch.cat(preds)


    @torch.no_grad()
    def evaluate(self, loader):
        """
            Produce the performance metrics of the model on a split

            Returns top-1, top-3 and top-5 accuracy, the parameter counts,
            and the cost of one clip. All measured in a single pass.
        """
        self.model.eval()

        correct_1, correct_3, correct_5, seen = 0, 0, 0, 0

        # Kept to measure cost against the clip shape actually being fed in
        clip = None

        for videos, labels in loader:
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(videos)

            # topk returns classes best first, so column 0 is the top-1 guess
            ranked = logits.topk(k = 5, dim = 1).indices
            hits = ranked == labels[:, None]

            correct_1 += hits[:, 0].sum().item()
            correct_3 += hits[:, :3].any(dim = 1).sum().item()
            correct_5 += hits.any(dim = 1).sum().item()
            seen += labels.size(0)

            if clip is None:
                clip = videos[:1]

        return {
            "samples": seen,
            "top1": correct_1 / seen,
            "top3": correct_3 / seen,
            "top5": correct_5 / seen,
            "params": sum(p.numel() for p in self.model.parameters()),
            "trainable": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "gflops": self._gflops(clip),
        }


    def _gflops(self, clip):
        """
            Cost of one forward pass over a single clip
        """
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError:
            return None

        counter = FlopCountAnalysis(self.model, clip)

        # The pooling and activation layers it cannot trace cost nothing anyway
        counter.unsupported_ops_warnings(False)
        counter.uncalled_modules_warnings(False)

        return counter.total() / 1e9



    def restore_best(self):
        """
            Load the best validation epoch back into the model itself

            fit() keeps those weights but leaves the model on the last epoch,
            so call this before predict() or evaluate() to score the same
            network that save(best = True) writes.
        """
        if self.best["state"] is None:
            raise ValueError("No validated epoch to restore, fit() needs a val_loader")

        self.model.load_state_dict(self.best["state"])

        return self.best["epoch"]


    def save(self, path, best: bool = False):
        """
            Write the trained weights to disk

            Pass best = True to write the epoch that scored highest on the
            validation split instead of the last one.
            Falls back to the final weights when fit() was given no validation split.
        """
        if best and self.best["state"] is not None:
            torch.save(self.best["state"], path)
        else:
            torch.save(self.model.state_dict(), path)


    def load(self, path):
        """
            Restore weights written by save()

            The X3D instance must be built with the same model_size and
            num_classes, otherwise the shapes will not line up.
        """
        self.model.load_state_dict(
            torch.load(path, map_location = self.device)
        )
