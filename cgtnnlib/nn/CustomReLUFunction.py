## CustomReluFunction v.0.1
## Created at Sat 23 Nov 2024

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


class CustomReLUFunction(torch.autograd.Function):
    """
    Implements a probabilistic ReLU function for PyTorch autograd.

        This custom function applies the ReLU activation with a given probability 'p'.
        It's designed to be used within an autograd context for differentiable operations.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, p: float):
        """
        Applies the ReLU function with a probability.

          This method implements a probabilistic ReLU where the ReLU activation
          is applied with a given probability 'p'. It saves the input and
          probability for use in the backward pass.

          Args:
            ctx: The context object for autograd.
            input: The input tensor to apply the function to.
            p: The probability of applying ReLU.

          Returns:
            torch.Tensor: The output tensor after applying ReLU with probability p.
                           This is equivalent to clamping the input at zero.
        """
        ctx.save_for_backward(input, torch.scalar_tensor(p))
        # ReLU is happening in clamp
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient of the custom operation.

            Args:
                ctx: The context object containing saved tensors from the forward pass.
                grad_output: The gradient of the output with respect to the output.

            Returns:
                A tuple containing the gradient with respect to the input and the gradient
                with respect to the parameters (which is None in this case).
        """
        input, p = ctx.saved_tensors
        grad_input = grad_output.clone()

        # Standard ReLU
        grad_input[input <= 0] = 0
        pprint(">>> grad_input[input <= 0] = 0")
        pprint(">>> grad_input")
        pprint(grad_input)

        # У матриц ось 0 это Y (Добавляем аргумент device=grad_output.device для указания устройства для создания тензора grad_input)
        # YYY 2. grad_input.size(0) на grad_input.size(1)
        bernoulli_mask = torch.bernoulli(
            torch.ones(grad_input.size(1), device=grad_output.device) * (1 - p.item())
        )
        pprint(
            ">>> bernoulli_mask = torch.bernoulli(torch.ones(grad_input.size(1), device=grad_output.device) * (1 - p.item()))"
        )
        # bernoulli_mask = torch.bernoulli(torch.ones(grad_input.size(0), device=grad_output.device) * (1 - p.item()))
        # pprint(">>> bernoulli_mask = torch.bernoulli(torch.ones(grad_input.size(0), device=grad_output.device) * (1 - p.item()))")
        pprint(">>> bernoulli_mask")
        pprint(bernoulli_mask)

        # ??? Непонятно, влияет ли деление это на что-либо
        # diagonal_mask = torch.diag(bernoulli_mask) / (1 - p.item()+1e-5)
        diagonal_mask = torch.diag(bernoulli_mask)
        pprint(">>> diagonal_mask = torch.diag(bernoulli_mask)")
        pprint(">>> diagonal_mask")
        pprint(diagonal_mask)

        # Перемещаем diagonal_mask на Cuda
        diagonal_mask = diagonal_mask.to(grad_output.device)

        # Multiply grad_input with the diagonal matrix
        # YYY 2. Заменить на grad_input @ diagonal_mask
        # grad_input = diagonal_mask @ grad_input
        grad_input = grad_input @ diagonal_mask
        pprint(">>> grad_input = grad_input @ diagonal_mask")
        pprint(">>> grad_input")
        pprint(grad_input)

        return grad_input, None
