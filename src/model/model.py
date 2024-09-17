import lightning as L
import torch
import torch.nn as nn
from src.model.conv_mixer import ConvMixer
# from src.model.intermediate_fusion import IntermediateFusion
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

class EarlyFusion(L.LightningModule):
  def __init__(self, mode='red', depth=15, dim=265, patch_size=8, n_classes=6):
    super().__init__()

    self.mode = mode
    if self.mode == 'color':
      self.net = ConvMixer(in_channels=9, depth=depth, dim=dim, patch_size=patch_size, n_classes=n_classes)
    else:
      self.net = ConvMixer(depth=depth, dim=dim, patch_size=patch_size, n_classes=n_classes)

    self.accuracy = MulticlassAccuracy(num_classes=6)
    self.precision = MulticlassPrecision(num_classes=6)
    self.recall = MulticlassRecall(num_classes=6)
    self.f1 = MulticlassF1Score(num_classes=6)

    self.criterion = nn.CrossEntropyLoss()

  def forward(self, x):
    return self.net(x)
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
    return optimizer
  
  def training_step(self, batch, batch_idx):
    x = batch[self.mode]
    y = batch['label']

    y_hat = self(x)
    loss = self.criterion(y_hat, y)

    self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    return loss
  
  def validation_step(self, batch, batch_idx):
    x = batch[self.mode]
    y = batch['label']

    y_hat = self(x)
    loss = self.criterion(y_hat, y)
    acc = self.accuracy(y_hat, y)
    pre = self.precision(y_hat, y)
    rec = self.recall(y_hat, y)
    f1 = self.f1(y_hat, y)

    self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
    self.log('val_pre', pre, prog_bar=True, on_step=False, on_epoch=True)
    self.log('val_rec', rec, prog_bar=True, on_step=False, on_epoch=True)
    self.log('val_f1', f1, prog_bar=True, on_step=False, on_epoch=True)

  def test_step(self, batch, batch_idx):
    x = batch[self.mode]
    y = batch['label']

    y_hat = self(x)
    loss = self.criterion(y_hat, y)

    acc = self.accuracy(y_hat, y)
    pre = self.precision(y_hat, y)
    rec = self.recall(y_hat, y)
    f1 = self.f1(y_hat, y)

    self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    self.log('test_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
    self.log('test_pre', pre, prog_bar=True, on_step=False, on_epoch=True)
    self.log('test_rec', rec, prog_bar=True, on_step=False, on_epoch=True)
    self.log('test_f1', f1, prog_bar=True, on_step=False, on_epoch=True)
    