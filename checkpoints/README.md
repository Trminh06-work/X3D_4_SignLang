# Checkpoints

Each file is a `state_dict` for a complete X3D network, i.e. the Kinetics-400 pretrained backbone and a classifier head resized to the 226 AUTSL classes.

Every run also writes its training curve to `figures/<same name>.png`, showing per-epoch loss and accuracy.

## Naming

    x3d_<size>_freeze<n>.pt

`<size>` is the X3D variant; `<n>` is how many of the network's 6 blocks were held fixed during training. Block 0 is the stem, blocks 1-4 the residual stages, block 5 the head. Some examples are:

| n | what trains |
|---|-------------|
| 0 | the whole network |
| 4 | last residual stage and head |
| 5 | the new head alone |

## Variants

| size | clip length | crop |
|------|-------------|------|
| XS | 4 frames | 160 px |
| S | 13 frames | 160 px |
| M | 16 frames | 224 px |

XS, S and M are the same architecture and differ only in the input clip they were pretrained on. A checkpoint is therefore only meaningful together with its clip geometry, hence `AUTSLDataset` must be built with the matching `num_frames` and `crop_size`, otherwise the weights load but the model sees inputs it was never trained for.

<!-- ## Runs configured in `script/train.conf`

| file | variant | frozen | notes |
|------|---------|-------:|-------|
| `x3d_xs_freeze5.pt` | XS | 5 | head only, cheapest run |
| `x3d_xs_freeze0.pt` | XS | 0 | full fine-tune |
| `x3d_s_freeze5.pt` | S | 5 | head only, longer clips |
| `x3d_s_freeze0.pt` | S | 0 | full fine-tune, longer clips | -->

## How they were trained

- **Data**: AUTSL train split, `data/train_labels.csv`: 28,142 clips over 226 classes, unless `SUBSET` is set in the config.

- **Preprocessing**: uniform temporal subsampling to the variant's clip length, resize to its crop size, random horizontal flip, scaled to [0, 1] and normalised with mean 0.45 and std 0.225 per channel.

- **Initialisation**: Kinetics-400 weights for the backbone, a fresh `nn.Linear` for the head.

- **Optimisation**: cross-entropy loss, Adam. Frozen blocks are kept out of the optimiser and held in eval mode, so their BatchNorm running statistics do not drift during training.

- **Hyperparameters**: epochs, batch size and learning rate come from `script/train.conf`; each job echoes its settings into `logs/`.


## Loading

```python
from src.x3d import X3D

clf = X3D(model_size = "XS", pretrained = False, num_classes = 226)
clf.load("checkpoints/x3d_xs_freeze5.pt")

preds = clf.predict(test_loader)
```

`model_size` and `num_classes` must match the values used at training time, or the shapes will not line up. `pretrained = False` avoids downloading Kinetics weights that the checkpoint immediately overwrites.
