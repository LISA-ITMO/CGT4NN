## CustomReluFunction v.0.1
## Created at Sat 23 Nov 2024

import torch

def pprint(*args):
    # print(*args)
    return

class CustomReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, p: float):
        ctx.save_for_backward(input, torch.scalar_tensor(p))
        # ReLU is happening in clamp
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, p = ctx.saved_tensors
        grad_input = grad_output.clone()
    
        grad_input[input <= 0] = 0
        pprint(">>> grad_input[input <= 0] = 0")
        pprint(">>> grad_input")
        pprint(grad_input)

        # У матриц ось 0 это Y (Добавляем аргумент device=grad_output.device для указания устройства для создания тензора grad_input)
        # YYY 2. grad_input.size(0) на grad_input.size(1)
        bernoulli_mask = torch.bernoulli(torch.ones(grad_input.size(1), device=grad_output.device) * (1 - p.item()))
        pprint(">>> bernoulli_mask = torch.bernoulli(torch.ones(grad_input.size(1), device=grad_output.device) * (1 - p.item()))")
        #bernoulli_mask = torch.bernoulli(torch.ones(grad_input.size(0), device=grad_output.device) * (1 - p.item()))
        #pprint(">>> bernoulli_mask = torch.bernoulli(torch.ones(grad_input.size(0), device=grad_output.device) * (1 - p.item()))")
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
        #grad_input = diagonal_mask @ grad_input
        grad_input = grad_input @ diagonal_mask
        pprint(">>> grad_input = grad_input @ diagonal_mask")
        pprint(">>> grad_input")
        pprint(grad_input)

        pprint("ON THE OTHER HAND:")
        pprint(">>> grad_input @ diagonal_mask ")
        pprint(grad_input @ diagonal_mask)

        return grad_input, None
    
