import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(in_channels=3, dim=256, depth=15, kernel_size=3, patch_size=8, n_classes=6, include_top=True):
    layers = [
        nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        
    ]

    if include_top:
        layers.append(nn.AdaptiveAvgPool2d((1,1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(dim, n_classes))

    return nn.Sequential(*layers)

if __name__ == "__main__":
  model = ConvMixer(depth=15, dim=256, include_top=False)
  x = torch.randn(16, 3, 224, 224)
  y = model(x)
  print(y.shape)
#   print(y)
#   print(y.softmax(dim=1).sum(dim=1))