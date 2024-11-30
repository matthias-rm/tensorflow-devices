# Tensorflow on Apple silicon

Comparing CNN training times on apple silicon CPU vs GPU using CIFAR10.

### Install

Try to use your default python version:
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

### 

Check if devices work:
```sh
python devices.py
```


```sh
python train.py
```

On my MBP-M4 the one-epoch training times are:
- GPU: ~78s
- CPU: ~38s
