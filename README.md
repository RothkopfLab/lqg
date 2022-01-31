# Inverse optimal control for linear-quadratic Gaussian systems

![Experimenter-actor-loop](https://raw.githubusercontent.com/RothkopfLab/lqg/main/img/experimenter-actor-loop.png)


This repository contains code for the paper
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
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage examples
The script `main.py` shows how to simulate data and infer parameters using the LQG model of the tracking task.
