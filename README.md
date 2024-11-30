# Tensorflow on Apple silicon

### Install

Either by using your default python version:
```sh
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install tensorflow tensorflow-macos tensorflow-metal
```
Or, use poetry:
```sh
poetry install
poetry shell
```

python=3.10.15
tensorflow=2.18.0
tensorflow-macos=2.
tensorflow-metal=1.1.0


tensorflow-metal 1.1.0
tensorflow-macos 2.15.1
tensorflow 2.18.0

pip 24.3.1

tensorflow                   2.16.2
tensorflow-macos             2.16.2
tensorflow-metal             1.1.0


https://stackoverflow.com/questions/59810276/why-is-my-poetry-virtualenv-using-the-system-python-instead-of-the-pyenv-python

poetry init: initializes poetry in existing project
poetry add tensorflow@^2.16.2 tensorflow-macos@2.16.2 tensorflow-metal@^1.1.0