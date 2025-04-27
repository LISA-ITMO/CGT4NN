import torch.nn as nn

from cgtnnlib.nn.CustomReLULayer import CustomReLULayer
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
        super(AugmentedReLUNetworkMultilayer, self).__init__()

        self.flatten = nn.Flatten()
        self.p = p
        self.hidden_layers_count = hidden_layers_count
        self.inner_layer_size = inner_layer_size

        self.hidden_layers = nn.ModuleList()

        self.hidden_layers.append(
            nn.Linear(inputs_count, self.inner_layer_size)
        )
        self.hidden_layers.append(
            CustomReLULayer(p)
        )

        # Subtract 1 because we already created the first layer
        for _ in range(hidden_layers_count - 1):
            self.hidden_layers.append(
                nn.Linear(self.inner_layer_size, self.inner_layer_size)
            )
            self.hidden_layers.append(
                CustomReLULayer(p)
            )

        self.hidden_layers.append(
            nn.Linear(self.inner_layer_size, outputs_count)
        )

    @property
    def inputs_count(self) -> int:
        return self.layers[0].in_featuress
    
    @property
    def outputs_count(self) -> int:
        return self.layers[-1].out_features

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)

        return x

    def __str__(self):
        return f"{self.__class__.__name__}(p: {self.p}, inputs_count: {self.inputs_count}, outputs_count: {self.outputs_count})"