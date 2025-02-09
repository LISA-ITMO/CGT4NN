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
- `analyze.py`: reads `report.json` and plots graphs
- `datasets.py`: dataset definitions
- `plt_extras.py`: matplotlib extensions
- `torch_device.py`: abstracts away PyTorch device selection
- `training.py`: training procedures
- etc.

The `nn` subdirectory contains PyTorch modules and functions that represent
neural architectures we evaluate

The `doc` subdirectory contains info about datasets.

## References

1. N. Ghani, J. Hedges, V. Winschel, and P. Zahn. *Compositional game theory*.
   Mar 2016. https://arxiv.org/abs/1603.04641
