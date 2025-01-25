import torch.nn as nn

from cgtnnlib.nn.NaiveAugmentedReLULayer import NaiveAugmentedReLULayer


class NaiveAugmentedNetwork(nn.Module):
    """
    Модель D. Нейросеть, использующая аугментацию NaiveReLULayer.
    """
    def __init__(self, inputs_count: int, outputs_count: int, p: float):
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
        return self.fc1.in_features
    
    @property
    def inner_layer_size(self) -> int:
        return 32 * 32

    @property
    def outputs_count(self) -> int:
        return self.fc3.out_features

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.custom_relu1(x)
        x = self.fc2(x)
        x = self.custom_relu2(x)
        x = self.fc3(x)
        return x

    def __str__(self):
        return f"{self.__class__.__name__}(p: {self.p}, inputs_count: {self.inputs_count}, outputs_count: {self.outputs_count})"