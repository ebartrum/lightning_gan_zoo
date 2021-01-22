import torch
from torch import nn
from torch import Tensor

@torch.no_grad()
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class VerboseShapeExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, input, output: print(
                    f"{layer.__name__}: {input[0].shape} --> {output.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)   

def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=1, keepdim=True)
    p = p / z1.pow(2).sum(dim=1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2

    return z
