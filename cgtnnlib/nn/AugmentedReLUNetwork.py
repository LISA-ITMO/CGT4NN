import torch.nn as nn

from cgtnnlib.nn.CustomReLULayer import CustomReLULayer
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
        super(AugmentedReLUNetwork, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(inputs_count, self.inner_layer_size)
        self.custom_relu1 = CustomReLULayer(p)
        self.fc2 = nn.Linear(self.inner_layer_size, self.inner_layer_size)
        self.custom_relu2 = CustomReLULayer(p)
        self.fc3 = nn.Linear(self.inner_layer_size, outputs_count)

        self.p = p
        
        self.softmax_output = softmax_output

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

        # if self.softmax_output:
        #     x = F.softmax(x, dim=1)

        return x

    def __str__(self):
        return f"{self.__class__.__name__}(p: {self.p}, inputs_count: {self.inputs_count}, outputs_count: {self.outputs_count})"