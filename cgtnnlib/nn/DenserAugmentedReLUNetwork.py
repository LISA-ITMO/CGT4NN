from cgtnnlib.nn.AugmentedReLUNetwork import AugmentedReLUNetwork


class DenserAugmentedReLUNetwork(AugmentedReLUNetwork):
    """
    Модель C. `AugmentedReLUNetwork` с увеличенным количеством нейронов
    во внутреннем слое.
    """

    @property
    def inner_layer_size(self):
        """
        Returns the size of the inner layer.

          The inner layer size is calculated as twice the NER layer size.

          Args:
            None

          Returns:
            int: The size of the inner layer.
        """
        return super().inner_layer_size * 2
