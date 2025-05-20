import torch.nn as nn
import torch.nn.functional as F


class RegularNetwork(nn.Module):
    """
    Нейросеть с обычными линейными слоями. Параметр `p` игнорируется.
    """

    def __init__(self, inputs_count: int, outputs_count: int, p: float):
        """
        Initializes the RegularNetwork.

            Args:
                inputs_count: The number of input features.
                outputs_count: The number of output classes.

            Returns:
                None
        """
        super(RegularNetwork, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(inputs_count, 32 * 32)
        self.fc2 = nn.Linear(32 * 32, 32 * 32)
        self.fc3 = nn.Linear(32 * 32, outputs_count)

    @property
    def inputs_count(self):
        """
        Returns the number of output features.

                Args:
                    None

                Returns:
                    int: The number of output features in the model.
        """
        return self.fc1.in_features

    @property
    def outputs_count(self):
        """
        Returns the number of output features.

            Args:
                None

            Returns:
                int: The number of output features produced by this layer.
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
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def __str__(self):
        """
        Returns a string representation of the node.

          Args:
            None

          Returns:
            str: A string containing the node's name, input count, and output count.
                 The format is "_name__}(inputs_count: {self.inputs_count}, outputs_count: {self.outputs_count})".
        """
        return f"{self.__class__.__name__}(inputs_count: {self.inputs_count}, outputs_count: {self.outputs_count})"
