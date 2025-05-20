## NaiveAugmentedReLUFunction v.0.1
## Created at Thu 5 Dec 2024

import torch


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


class NaiveAugmentedReLUFunction(torch.autograd.Function):
    """
    Applies ReLU with a probability 'p' and computes its gradient."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, p: float):
        """
        Applies the ReLU function with a probability.

          This method implements a probabilistic ReLU where the ReLU activation
          is applied to the input tensor with a given probability 'p'. It also
          saves necessary tensors for backpropagation.

          Args:
            ctx: The context object for autograd.
            input: The input tensor.
            p: The probability of applying ReLU.

          Returns:
            torch.Tensor: The output tensor after applying the probabilistic ReLU,
              which is equivalent to clamping the input at zero from below.
        """
        ctx.save_for_backward(input, torch.scalar_tensor(p))
        # ReLU is happening in clamp
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient of the function with respect to its inputs.

          Args:
            ctx: The context object containing saved tensors from the forward pass.
            grad_output: The gradient of the output with respect to the loss.

          Returns:
            tuple: A tuple containing the gradients with respect to the input and p.
                   The first element is the gradient w.r.t. the input, and the second
                   element is the gradient w.r.t. p (which is always None in this case).
        """
        input, p = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input[input <= 0] = 0
        pprint("<<< grad_input[input <= 0] = 0")
        pprint("<<< grad_input")
        pprint(grad_input)

        grad_input = grad_input * p
        pprint("<<< grad_input = grad_input * p")
        pprint("<<< grad_input")
        pprint(grad_input)

        return grad_input, None
