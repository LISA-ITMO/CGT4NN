import torch
import torch.nn.functional as F


class CustomReLUBackwardFunction(torch.autograd.Function):
    """
    Переопределённая функция для слоя ReLU.
    """

    @staticmethod
    def forward(ctx, p: float, input: torch.Tensor):
        """
        Applies the ReLU function element-wise.

            This method also saves necessary tensors for backpropagation.

            Args:
                input: The input tensor.
                p: A scalar value (used during backward pass).

            Returns:
                torch.Tensor: The output tensor after applying the ReLU function.
        """
        ctx.save_for_backward(torch.scalar_tensor(p), input)
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Computes the gradient of the custom operation.

            Args:
                grad_output: The gradient of the output with respect to the output.

            Returns:
                tuple: A tuple containing None (gradient w.r.t. input) and the
                       computed gradient with respect to the parameter p.
        """
        p, input = ctx.saved_tensors

        grad_output = grad_output * (input > 0).float()

        # У матриц ось 0 это Y
        height = grad_output.size(0)
        bernoulli_mask = torch.bernoulli(torch.ones(height) * (1 - p.item()))
        diagonal_mask = torch.diag(bernoulli_mask) / (1 - p.item())

        diagonal_mask = diagonal_mask.unsqueeze(1).expand(-1, grad_output.size(1), -1)
        diagonal_mask = diagonal_mask.permute(0, 2, 1)

        grad_output = grad_output.unsqueeze(1) * diagonal_mask
        grad_output = grad_output.sum(dim=1)

        return None, grad_output
