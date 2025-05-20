import torch

from cgtnnlib.nn.CustomReLUFunction import CustomReLUFunction


class GradientDropoutReLULayer(torch.nn.Module):
    """
    Used to be `CustomReLULayer`
    """

    def __init__(self, p: float):
        """
        Initializes the GradientDropoutReLULayer.

            Args:
                p: The dropout probability.

            Returns:
                None
        """
        super(GradientDropoutReLULayer, self).__init__()
        self.p = p
        self.custom_relu_backward = CustomReLUFunction.apply

    def forward(self, x):
        """
        Applies the custom ReLU backward pass.

          Args:
            x: The input tensor.

          Returns:
            The output tensor after applying the custom ReLU backward operation.
        """
        return self.custom_relu_backward(x, self.p)
