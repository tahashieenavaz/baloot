import torch


# always add data normalized to this function (mean=0, std=1)
def ksg_mi(x, y, k=3):
    assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"

    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    N = x.shape[0]

    x = x + torch.randn_like(x) * 1e-10
    y = y + torch.randn_like(y) * 1e-10

    joint = torch.cat((x, y), dim=1)

    dist_joint = torch.cdist(joint, joint, p=float("inf"))

    radius = torch.topk(dist_joint, k + 1, largest=False).values[:, k]

    radius = radius.unsqueeze(1) - 1e-15

    dist_x = torch.cdist(x, x, p=float("inf"))
    dist_y = torch.cdist(y, y, p=float("inf"))

    nx = (dist_x < radius).sum(dim=1).float()
    ny = (dist_y < radius).sum(dim=1).float()

    psi = torch.digamma
    mi = (
        psi(torch.tensor(k)) + psi(torch.tensor(N)) - (psi(nx + 1) + psi(ny + 1)).mean()
    )
    return torch.max(mi, torch.tensor(0.0))  # Clip at 0 for interpretation
