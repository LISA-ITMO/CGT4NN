import torch.nn as nn

from cgtnnlib.nn.NaiveAugmentedReLULayer import NaiveAugmentedReLULayer


class NaiveAugmentedNetwork(nn.Module):
    """
    Модель D. Нейросеть, использующая аугментацию NaiveReLULayer.
    """

    def __init__(self, inputs_count: int, outputs_count: int, p: float):
        """
        Initializes the NaiveAugmentedNetwork.

            Args:
                inputs_count: The number of input features.
                outputs_count: The number of output classes.
                p: The probability for the augmented ReLU layer.

            Returns:
                None
        """
        super(NaiveAugmentedNetwork, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(inputs_count, self.inner_layer_size)
        self.fc2 = nn.Linear(self.inner_layer_size, self.inner_layer_size)
        self.fc3 = nn.Linear(self.inner_layer_size, outputs_count)

        self.custom_relu1 = NaiveAugmentedReLULayer(p)
        self.custom_relu2 = NaiveAugmentedReLULayer(p)

        self.p = p

    @property
    def inputs_count(self) -> int:
        """
        Returns the number of input features to the first fully connected layer.

            This property provides access to the size of the input layer,
            effectively indicating how many features the model expects as input.

            Parameters:
                None

            Returns:
                int: The number of input features.
        """
        return self.fc1.in_features

    @property
    def inner_layer_size(self) -> int:
        """
        Returns the size of the inner layer.

                Args:
                    None

                Returns:
                    int: The size of the inner layer, which is always 1024.
        """
        return 32 * 32

    @property
    def outputs_count(self) -> int:
        """
        Returns the number of output features.

            This property accesses the `out_features` attribute of the final fully connected layer (`fc3`)
            to determine the dimensionality of the model's output.

            Returns:
                int: The number of output features in the model.
        """
        return self.fc3.out_features

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

            Args:
                x: The input tensor.

            Returns:
                The output tensor after passing through the fully connected layers and ReLU activations.
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.custom_relu1(x)
        x = self.fc2(x)
        x = self.custom_relu2(x)
        x = self.fc3(x)
        return x

    def __str__(self):
        """
        Returns a string representation of the ELF object.

          Args:
            self: The ELF object instance.

          Returns:
            str: A formatted string containing the class name, 'p' value,
                 and input/output counts.
        """
        return f"{self.__class__.__name__}(p: {self.p}, inputs_count: {self.inputs_count}, outputs_count: {self.outputs_count})"
