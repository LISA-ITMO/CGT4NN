import torch.nn as nn

from cgtnnlib.nn.GradientDropoutReLULayer import GradientDropoutReLULayer
import torch.nn.functional as F


class AugmentedReLUNetworkMultilayer(nn.Module):
    """
    Модель B*. Нейросеть с переопределённой функцией распространения ошибки
    для функции активации и поддержкой множественных скрытых слоёв.

    По умолчанию `hidden_layers_count` равен 1.
    """

    def __init__(
        self,
        inputs_count: int,
        outputs_count: int,
        p: float,
        inner_layer_size: int = 32 * 32,
        hidden_layers_count: int = 1,
    ):
        """
        Initializes the AugmentedReLUNetworkMultilayer.

            Args:
                inputs_count: The number of input features.
                outputs_count: The number of output features.
                p: The dropout probability for GradientDropoutReLULayer.
                inner_layer_size: The size of the inner layers (default is 32 * 32).
                hidden_layers_count: The number of hidden layers (default is 1).

            Returns:
                None
        """
        super(AugmentedReLUNetworkMultilayer, self).__init__()

        self.flatten = nn.Flatten()
        self.p = p
        self.inner_layer_size = inner_layer_size
        self.hidden_layers_count = hidden_layers_count
        self.layers = nn.ModuleList()

        input_layer = nn.Linear(inputs_count, self.inner_layer_size)
        self.layers.append(input_layer)
        self.layers.append(GradientDropoutReLULayer(p))

        # Subtract 1 because we already created the first layer
        for _ in range(hidden_layers_count - 1):
            self.layers.append(nn.Linear(self.inner_layer_size, self.inner_layer_size))
            self.layers.append(GradientDropoutReLULayer(p))

        output_layer = nn.Linear(self.inner_layer_size, outputs_count)
        self.layers.append(output_layer)

    @property
    def inputs_count(self) -> int:
        """
        Returns the number of input features to the final layer.

            This corresponds to the output dimension of the last layer in the model.

            Args:
                None

            Returns:
                int: The number of input features for the final layer.
        """
        return self.layers[0].in_featuress

    @property
    def outputs_count(self) -> int:
        """
        Returns the number of output features.

            This property iterates through the layers of the model and determines
            the size of the output tensor after passing a sample input through all layers.

            Args:
                None

            Returns:
                int: The number of output features in the final layer's output.
        """
        return self.layers[-1].out_features

    def forward(self, x):
        """
        Returns a string representation of the object.

            Args:
                None

            Returns:
                str: A string containing the class name, p value, and inputs/outputs count.
        """
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)

        return x

    def __str__(self):
        """
        Returns a string representation of the object.

            Args:
                None

            Returns:
                str: A string containing the name and outputs count of the object,
                     formatted as "Name (Outputs Count)".
        """
        return f"{self.__class__.__name__}(p: {self.p}, inputs_count: {self.inputs_count}, outputs_count: {self.outputs_count})"
