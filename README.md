# Inverse optimal control for continuous psychophysics

![Experimenter-actor-loop](https://raw.githubusercontent.com/RothkopfLab/lqg/main/docs/source/_static/experimenter-actor-loop.png)


This repository contains the official [JAX](https://github.com/google/jax) implementation of the inverse optimal control method presented in the paper:

> [Straub, D., & Rothkopf, C. A. (2022). Putting perception into action with inverse optimal control for continuous psychophysics. eLife, 11, e76635.](https://elifesciences.org/articles/76635)

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

- [`notebooks/02-Data.ipynb`](https://github.com/RothkopfLab/lqg/blob/main/notebooks/02-Data.ipynb) fits the ideal observer and bounded actor model to the [data](https://github.com/kbonnen/BonnenEtAl2015_KalmanFilterCode) from [Bonnen et al. (2015)](https://jov.arvojournals.org/article.aspx?articleid=2301260) to reproduce Fig. 4A from our paper.

## Citation
If you use our method or code in your research, please cite our paper:

```
@article{straub2022putting,
  title={Putting perception into action with inverse optimal control for continuous psychophysics},
  author={Straub, Dominik and Rothkopf, Constantin A},
  journal={eLife},
  volume={11},
  pages={e76635},
  year={2022},
  publisher={eLife Sciences Publications Limited}
}
```

## Signal-dependent noise
This implementation supports the basic LQG framework. For the extension to signal-dependent noise [(Todorov, 2005)](https://direct.mit.edu/neco/article-abstract/17/5/1084/6949/Stochastic-Optimal-Control-and-Estimation-Methods), please see [our NeurIPS 2021 paper](https://proceedings.neurips.cc/paper/2021/hash/4e55139e019a58e0084f194f758ffdea-Abstract.html) and [its implementation](https://github.com/RothkopfLab/inverse-optimal-control).
