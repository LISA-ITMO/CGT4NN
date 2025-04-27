import torch

from cgtnnlib.nn.CustomReLUFunction import CustomReLUFunction


class GradientDropoutReLULayer(torch.nn.Module):
    """
    Used to be `CustomReLULayer`
    """

    def __init__(self, p: float):
        super(GradientDropoutReLULayer, self).__init__()
        self.p = p
        self.custom_relu_backward = CustomReLUFunction.apply

    def forward(self, x):
        return self.custom_relu_backward(x, self.p)