import torch
import torch.nn as nn

from src.model.conv_mixer import ConvMixer

class InterMediateFusion(nn.Module):
  def __init__(self, use_attention=False, depth=15, dim=256, patch_size=8, n_classes=6):
    super().__init__()

    self.use_attention = use_attention

    self.mixer_r = ConvMixer(depth=depth, dim=dim, include_top=False) # ConvMixerBlock for Red Channel
    self.mixer_g = ConvMixer(depth=depth, dim=dim, include_top=False) # ConvMixerBlock for Green Channel
    self.mixer_b = ConvMixer(depth=depth, dim=dim, include_top=False) # ConvMixerBlock for Blue Channel

    self.attention_block = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(dim, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, dim),
      nn.Sigmoid()
    )

    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.flatten = nn.Flatten()
    self.fc = nn.Linear(dim, n_classes)


  def forward(self, r, g, b):
    r_feat = self.mixer_r(r)
    g_feat = self.mixer_g(g)
    b_feat = self.mixer_b(b)

    if self.use_attention:
      print(r_feat.shape, g_feat.shape, b_feat.shape)
      print((r_feat + g_feat + b_feat).shape)
      attention = self.attention_block(r_feat + g_feat + b_feat)
      print(attention.shape)
      red_attention = torch.mul(r_feat, attention)
      green_attention = torch.mul(g_feat, attention)
      blue_attention = torch.mul(b_feat, attention)

      fused = red_attention + green_attention + blue_attention

    else:
      fused = r_feat + g_feat + b_feat

    fused = self.pool(fused)
    fused = self.flatten(fused)
    fused = self.fc(fused)

    return fused
    

if __name__ == "__main__":
  model = InterMediateFusion(use_attention=True)
  x_r = torch.randn(16, 3, 224, 224)
  x_g = torch.randn(16, 3, 224, 224)
  x_b = torch.randn(16, 3, 224, 224)
  y = model(x_r, x_g, x_b)
  print(y.shape)
  print(y)
  print(y.softmax(dim=1).sum(dim=1))