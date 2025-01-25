from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork


class DenserAugmentedReLUNetwork(AugmentedReLUNetwork):
    """
    Модель C. `AugmentedReLUNetwork` с увеличенным количеством нейронов
    во внутреннем слое.
    """
    @property
    def inner_layer_size(self):
        return super().inner_layer_size * 2