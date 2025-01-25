## NaiveAugmentedReLUFunction v.0.1
## Created at Thu 5 Dec 2024

import torch

def pprint(*args):
    # print(*args)
    return

class NaiveAugmentedReLUFunction(torch.autograd.Function):
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
        pprint("<<< grad_input[input <= 0] = 0")
        pprint("<<< grad_input")
        pprint(grad_input)

        grad_input = grad_input * p
        pprint("<<< grad_input = grad_input * p")
        pprint("<<< grad_input")
        pprint(grad_input)

        return grad_input, None
    
