import torch


class Reshape(torch.nn.Module):
    def forward(self, x):
        B = x.shape[0]
        DIM = x.shape[1]
        return x.view(B, -1, DIM)
