import torch

from cgtnnlib.nn.NaiveAugmentedReLUFunction import NaiveAugmentedReLUFunction


class NaiveAugmentedReLULayer(torch.nn.Module):
    def __init__(self, p: float):
        super(NaiveAugmentedReLULayer, self).__init__()
        self.p = p
        self.custom_relu_backward = NaiveAugmentedReLUFunction.apply

    def forward(self, x):
        return self.custom_relu_backward(x, self.p)