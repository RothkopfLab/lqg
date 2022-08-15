# Inverse optimal control for continuous psychophysics

![Experimenter-actor-loop](https://raw.githubusercontent.com/RothkopfLab/lqg/main/img/experimenter-actor-loop.png)


This repository contains the official implementation of the inverse optimal control method presented in the paper:

> Straub, D., & Rothkopf, C. A. (2021). Putting perception into action: Inverse optimal control for continuous psychophysics. [bioRxiv.](https://www.biorxiv.org/content/10.1101/2021.12.23.473976v1.abstract)

## Installation
The package can be installed via `pip`

```bash
python -m pip install lqg
```

although I recommend cloning the repository to get the most recent version and installing locally with a virtual environment

```bash
python -m venv env
source env/bin/activate
python -m pip install -e .
```

## Usage examples
- `main.py` shows how to simulate data and infer parameters using the LQG model of the tracking task.

- [`notebooks/01-HowTo.ipynb`](https://github.com/RothkopfLab/lqg/blob/main/notebooks/01-HowTo.ipynb) explains the model and its parameters in more detail, including the extension to subjective internal models.

- [`notebooks/02-Data.ipynb`](https://github.com/RothkopfLab/lqg/blob/main/notebooks/02-Data.ipynb) fits the ideal observer and subjective actor model to the [data](https://github.com/kbonnen/BonnenEtAl2015_KalmanFilterCode) from [Bonnen et al. (2015)](https://jov.arvojournals.org/article.aspx?articleid=2301260) to reproduce Fig. 4A from our paper.

## Citation
If you use our method in your research, please cite our preprint:

```
@article{straub2021putting,
  title={Putting perception into action: Inverse optimal control for continuous psychophysics},
  author={Straub, Dominik and Rothkopf, Constantin A},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
