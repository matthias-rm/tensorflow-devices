# Tensorflow on Apple silicon

Comparing CNN training times on apple silicon CPU vs GPU using CIFAR10.

### Install

Using default python version (might fail):
```sh
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install tensorflow tensorflow-macos tensorflow-metal
```

Or, use poetry with python 3.10:
```sh
poetry env use 3.10
poetry install
poetry shell
```

### Run

Check if devices work:
```sh
python devices.py
```

Train and compare CNN running times:
```sh
python train.py
```

One-epoch training times on a MBP-M4 are:
- GPU: ~78s
- CPU: ~38s
