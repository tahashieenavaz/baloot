import torch


class Reshape(torch.nn.Module):
    def forward(self, x):
        B = x.shape[0]
        DIM = x.shape[1]
        return x.view(B, -1, DIM)


class PatchEmbedding(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        features: int,
        patch_width: int,
        patch_height: int,
        activate: bool = False,
        activation=torch.nn.functional.gelu,
    ):
        super().__init__()
        self.activate = activate
        self.activation = activation

        self.stream = torch.nn.Sequential(
            torch.nn.Conv2d(channels, features, patch_width, patch_height),
            torch.nn.GroupNorm(features // 16, features),
        )

    def forward(self, x: torch.Tensor):
        features = self.stream(x)

        if self.activate:
            return self.activation(features)

        return features
