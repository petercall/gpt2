import torch

tens1 = torch.tensor([[1,2], [3,4]])
tens2 = torch.tensor([[5,6], [7,8]])
tens3 = torch.tensor([[9,10], [11,12]])

new_tens = torch.stack([tens1, tens2, tens3], dim = 2)
print(new_tens.shape)
print(new_tens)