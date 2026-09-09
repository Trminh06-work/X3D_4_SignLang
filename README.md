# X3D on AUTSL

Fine-tuning [X3D](https://arxiv.org/abs/2004.04730) video networks for isolated sign language recognition on [AUTSL](https://www.kaggle.com/datasets/baominhtran06/sit332-autsl4x3d), a 226-class Turkish Sign Language dataset.

We answer two questions in this repo:

- How different fine-tuning approaches affect different model sizes?

- How capable the X3D is to learn isolated sign language?

Three X3D variants are each trained at six freeze depths, giving 18 runs over one grid.

## Layout

    src/dataset.py       AUTSL clips
    src/x3d.py           The model
    src/main.py          train and score on train/test sets
    src/evaluate.py      score every model checkpoint on test set

    script/train.sh      Slurm array job
    script/train.conf    hyperparameters and paths
    script/evaluate.sh   Slurm job for evaluation

    checkpoints/         18 state_dicts
    report/              LaTeX source and figures
    notebooks/lab.ipynb  exploratory work
    my_data/             hand-crafted clips

## Setup

```bash
conda create -n videoCV python=3.11 && conda activate videoCV
pip install -r requirements.txt
```

The clips are not in this repository. Download the RGB-only AUTSL release from
[Kaggle](https://www.kaggle.com/datasets/sttaseen/autsl) and unpack it.

## Running

One variant locally, configured entirely through the environment:

```bash
SIZE=XS FROZEN=1 EPOCHS=100 BATCH_SIZE=16 python -m src.main
```

The full sweep on Slurm

```bash
sbatch script/train.sh      # trains, writes checkpoints/ and figures/
sbatch script/evaluate.sh   # scores every checkpoint into results/
```
