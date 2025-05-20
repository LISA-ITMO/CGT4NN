import torch.nn as nn

from cgtnnlib.nn.GradientDropoutReLULayer import GradientDropoutReLULayer
import torch.nn.functional as F


class AugmentedReLUNetwork(nn.Module):
    """
    Модель B. Нейросеть с переопределённой функцией распространения ошибки
    для функции активации.
    """

    def __init__(
        self,
        inputs_count: int,
        outputs_count: int,
        p: float,
        softmax_output: bool = False,
    ):
        """
        Initializes the AugmentedReLUNetwork.

            Args:
                inputs_count: The number of input features.
                outputs_count: The number of output features.
                t:  A parameter used in GradientDropoutReLULayer.
                p: The dropout probability for GradientDropoutReLULayer.
                softmax_output: Whether to apply a softmax function to the output. Defaults to False.

            Returns:
                None
        """
        super(AugmentedReLUNetwork, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(inputs_count, self.inner_layer_size)
        self.custom_relu1 = GradientDropoutReLULayer(p)
        self.fc2 = nn.Linear(self.inner_layer_size, self.inner_layer_size)
        self.custom_relu2 = GradientDropoutReLULayer(p)
        self.fc3 = nn.Linear(self.inner_layer_size, outputs_count)

        self.p = p

        self.softmax_output = softmax_output

    @property
    def inputs_count(self) -> int:
        """
        Returns the size of the inner layer.

            This property calculates and returns the size of the inner layer
            based on a fixed multiplication factor.

            Returns:
                int: The size of the inner layer, which is always 96 (32 * 3).
        """
        return self.fc1.in_features

    @property
    def inner_layer_size(self) -> int:
        """
        Returns the size of the inner fully connected layer.

            Args:
                None

            Returns:
                int: The output dimension (number of neurons) of the third
                     fully connected layer (fc3).
        """
        return 32 * 32

    @property
    def outputs_count(self) -> int:
        """
        Returns the number of output features.

            This method flattens the input, passes it through a fully connected layer (fc1),
            and implicitly returns the size of the output tensor from that layer which represents the count.

            Args:
                x: The input tensor.

            Returns:
                int: The number of output features after processing with fc1.
        """
        return self.fc3.out_features

    def forward(self, x):
        """
        Performs a forward pass through the network.

            Args:
                x: The input tensor.

            Returns:
                The output tensor after passing through fully connected layers and ReLU activations.
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.custom_relu1(x)
        x = self.fc2(x)
        x = self.custom_relu2(x)
        x = self.fc3(x)

        # if self.softmax_output:
        #     x = F.softmax(x, dim=1)

        return x

    def __str__(self):
        """
        Returns a string representation of the object.

          Args:
            None

          Returns:
            str: A string containing the number of inputs and outputs for this
                 object, formatted as "s_count: {self.inputs_count}, outputs_count: {self.outputs_count}".
        """
        return f"{self.__class__.__name__}(p: {self.p}, inputs_count: {self.inputs_count}, outputs_count: {self.outputs_count})"
