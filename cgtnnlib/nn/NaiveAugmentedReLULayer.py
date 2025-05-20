import torch

from cgtnnlib.nn.NaiveAugmentedReLUFunction import NaiveAugmentedReLUFunction


class NaiveAugmentedReLULayer(torch.nn.Module):
    """
    Applies an augmented ReLU activation function with a probability p."""

    def __init__(self, p: float):
        """
        Initializes the NaiveAugmentedReLULayer.

            Args:
                p: The probability of applying the augmentation.

            Returns:
                None
        """
        super(NaiveAugmentedReLULayer, self).__init__()
        self.p = p
        self.custom_relu_backward = NaiveAugmentedReLUFunction.apply

    def forward(self, x):
        """
        Applies the custom ReLU backward pass.

          Args:
            x: The input tensor.

          Returns:
            The output tensor after applying the custom ReLU backward operation.
        """
        return self.custom_relu_backward(x, self.p)
