## Project-global torch device

import torch

if torch.cuda.is_available():
    print('TORCH_DEVICE is cuda')
    TORCH_DEVICE = 'cuda'
# elif torch.backends.mps.is_available():
#     print('TORCH_DEVICE is mps')
#     TORCH_DEVICE = 'mps'
else:
    print('TORCH_DEVICE is cpu')
    TORCH_DEVICE = 'cpu'