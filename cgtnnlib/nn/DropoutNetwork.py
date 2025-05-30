import torch.nn as nn


class DropoutNetwork(nn.Module):
    """
    Модель E. Нейросеть с дропаутом на двух ближних к входу слоях.
    """

    def __init__(
        self,
        inputs_count: int,
        outputs_count: int,
        p: float,
    ):
        """
        Initializes the DropoutNetwork.

            Args:
                inputs_count: The number of input features.
                outputs_count: The number of output classes.
                p: The dropout probability.

            Returns:
                None
        """
        super(DropoutNetwork, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(inputs_count, self.inner_layer_size)
        self.dropout1 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(self.inner_layer_size, self.inner_layer_size)
        self.dropout2 = nn.Dropout(p=p)
        self.fc3 = nn.Linear(self.inner_layer_size, outputs_count)

        self.p = p

    @property
    def inputs_count(self) -> int:
        """
        Returns the number of input features to the first layer.

            This property accesses the `in_features` attribute of the first fully connected layer (fc1)
            to determine the size of the input layer.

            Returns:
                int: The number of input features.
        """
        return self.fc1.in_features

    @property
    def inner_layer_size(self) -> int:
        """
        Returns the size of the inner layer.

            This represents the dimensionality of the hidden layer within the network.
            It's a fixed value determined by the architecture.

            Returns:
                int: The size of the inner layer, which is currently 1024 (32 * 32).
        """
        return 32 * 32

    @property
    def outputs_count(self) -> int:
        """
        Returns the number of output features.

            This property provides access to the number of output features
            defined in the first fully connected layer (lf.fc3) within the module.

            Returns:
                int: The number of output features from lf.fc3.out_features.
        """
        return self.fc3.out_features

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

            Args:
                x: The input tensor.

            Returns:
                The output tensor after passing through fully connected layers and dropout.
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def __str__(self):
        """
        Returns a string representation of the object.

          Args:
            p: The value of p.
            inputs_count: The number of inputs.
            outputs_count: The number of outputs.

          Returns:
            str: A formatted string containing the object's name, p value,
                 input count, and output count.
        """
        return f"{self.__class__.__name__}(p: {self.p}, inputs_count: {self.inputs_count}, outputs_count: {self.outputs_count})"
