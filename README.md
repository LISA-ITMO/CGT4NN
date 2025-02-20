# Compositional Game Theory for Neural Networks

This repository explores the application of compositional game
theory, as introduced in the paper *Compositional Game Theory* [1] to
the analysis and enhancement of neural networks. We represent neural
network components as players in open games, aiming to leverage
game-theoretic tools for improved training and understanding

Repository includes:

- cgtnnlib, a library for performing the research
- `data` directory with some of the data we use
- `doc` directory with documentation
- notebooks for running experiments

As of now, the main branch is in flux and many of the stable releases are available at https://disk.yandex.ru/d/aZozDpBlzh_z1A

## How to run

1. Create a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Open a notebook `*.ipynb` file with any .ipynb reader available to you
   and run

## cgtnnlib

The library consists of classes (with filenames beginning with capital letter)
that represent problem domain (Dataset, Report, etc.) and several procedural
modules:

- `common.py`: main functions and evaluation
- `analyze.py`: reads report JSON files and plots graphs
- `datasets.py`: dataset definitions
- `plt_extras.py`: matplotlib extensions
- `torch_device.py`: abstracts away PyTorch device selection
- `training.py`: training procedures
- etc.

The `nn` subdirectory contains PyTorch modules and functions that represent
neural architectures we evaluate

The `doc` subdirectory contains info about datasets.

## Reports

Trained models are stored in the `pth/` directory. Along with each
model, a corresponding JSON file is also created which contains
properties like:

- `started`: date of report creation
- `saved`: date of last update
- `model`: model parameters, like classname and hyperparameter value
- `dataset`: dataset info, including the type of learning task
  (regression/classification)
- `loss`: an array of loss values during each iteration of training,
  for analyzing loss curves
- `eval`: an object that contains various values of "noise_factor",
  that represents noise mixed into the input during evaluation, and
  their corresponding evaluation metrics values: "r2" and "mse" for
  regression, and "f1", "accuracy", "roc_auc" for classification
- other, experiment-specific keys

Typically a report is created during the model creation and initial
training, and then updated during evaluation. This two-step process
creates the complete report to be analyzed by `analyze.py`.

## References

1. N. Ghani, J. Hedges, V. Winschel, and P. Zahn. *Compositional game theory*.
   Mar 2016. https://arxiv.org/abs/1603.04641
