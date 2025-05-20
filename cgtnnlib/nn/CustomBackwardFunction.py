import torch

from cgtnnlib.nn.MockCtx import MockCtx


def pprint(*args):
    """
    Prints the given arguments to standard output.

        This method is a placeholder for pretty printing functionality,
        currently it does nothing but return None.

        Args:
            *args: The arguments to be printed.  Can be any number of objects.

        Returns:
            None"""
    # print(*args)
    return


class CustomBackwardFunction(torch.autograd.Function):
    """
    Переопределённая функция для линейного слоя.
    """

    @staticmethod
    def forward(
        ctx: MockCtx,
        p: float,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: float | None = None,
    ):
        """
        Computes a linear transformation with optional bias.

            Args:
                p: A scalar value (used for saving in the context).
                input: The input tensor.
                weight: The weight tensor.
                bias: Optional bias tensor.

            Returns:
                torch.Tensor: The output tensor after applying the linear transformation.
        """
        ctx.save_for_backward(torch.scalar_tensor(p), input, weight, bias)

        output = input.mm(weight.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx: MockCtx, *grad_outputs):
        """
        Computes the gradients for the custom backward pass.

            Args:
                grad_outputs: The gradient of the loss with respect to the output.

            Returns:
                tuple: A tuple containing the gradients with respect to the input, weight, and bias.
                       The first element is always None as this operation does not have a gradient with respect to the parameter 'p'.
        """
        p, input, weight, bias = ctx.saved_tensors

        height = weight.size(0)
        bernoulli_mask = torch.bernoulli(torch.ones(height) * (1 - p.item()))

        diagonal_mask = torch.diag(bernoulli_mask) / (1 - p.item())

        grad_output = grad_outputs[0]

        grad_output = grad_output.mm(diagonal_mask)

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)

        if bias is not None:
            grad_bias = grad_output.sum(0)
        else:
            grad_bias = None

        # Yes, None
        return None, grad_input, grad_weight, grad_bias
