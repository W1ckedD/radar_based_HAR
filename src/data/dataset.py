import os
from PIL import Image
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

class RadarDataset(Dataset):
  def __init__(self, root_dirs: str, transform_rgb=None, transform_gray=None):
    self.root_dirs = root_dirs
    self.samples = []
    for root_dir in root_dirs:
      for file in os.listdir(root_dir):
        self.samples.append(os.path.join(root_dir, file))

    
    if transform_rgb is None:
      self.transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
    else:
      self.transform_rgb = transform_rgb

    if transform_gray is None:
      self.transform_gray = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
      ])
    else:
      self.transform_gray = transform_gray

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, idx):
    sample_dir = self.samples[idx]
    sample = sample_dir.split('/')[-1]
    label = int(sample[0])

    range_time_rgb = Image.open(os.path.join(sample_dir, f"{sample}_RangeTimePlot.png")).convert('RGB')
    range_time = self.transform_rgb(range_time_rgb)

    range_doppler_rgb = Image.open(os.path.join(sample_dir, f"{sample}_RangeDopplerPlot.png")).convert('RGB')
    range_doppler = self.transform_rgb(range_doppler_rgb)
    
    doppler_time_rgb = Image.open(os.path.join(sample_dir, f"{sample}_DopplerTimeSpectrogram.png")).convert('RGB')
    doppler_time = self.transform_rgb(doppler_time_rgb)

    red_fused = torch.stack([range_time[0], range_doppler[0], doppler_time[0]], dim=0)
    green_fused = torch.stack([range_time[1], range_doppler[1], doppler_time[1]], dim=0)
    blue_fused = torch.stack([range_time[2], range_doppler[2], doppler_time[2]], dim=0)

    color_fused = torch.cat([red_fused, green_fused, blue_fused], dim=0)
    
    
    range_time_gray = range_time_rgb.copy().convert('L')
    range_time_gray = self.transform_gray(range_time_gray)

    range_doppler_gray = range_doppler_rgb.copy().convert('L')
    range_doppler_gray = self.transform_gray(range_doppler_gray)

    doppler_time_gray = doppler_time_rgb.copy().convert('L')
    doppler_time_gray = self.transform_gray(doppler_time_gray)

    gray_fused = torch.stack([range_time_gray.squeeze(0), range_doppler_gray.squeeze(0), doppler_time_gray.squeeze(0)], dim=0)

    return {
      'red': red_fused,
      'green': green_fused,
      'blue': blue_fused,
      'gray': gray_fused,
      'color': color_fused,
      'label': label - 1
    }

class RadarDataModule(L.LightningDataModule):
  def __init__(self, root: str, location: str = 'A', batch_size: int = 32, num_workers: int = 4):
    super().__init__()
    self.root = root
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.location = location

  def prepare_data(self):
    pass

  def load_folds(self):
    if self.location == 'A':
      val_dataset = RadarDataset(root_dirs=[os.path.join(self.root, 'location_A')])
      train_dataset = RadarDataset(root_dirs=[os.path.join(self.root, 'location_B'), os.path.join(self.root, 'location_C')])

    elif self.location == 'B':
      val_dataset = RadarDataset(root_dirs=[os.path.join(self.root, 'location_B')])
      train_dataset = RadarDataset(root_dirs=[os.path.join(self.root, 'location_A'), os.path.join(self.root, 'location_C')])

    elif self.location == 'C':
      val_dataset = RadarDataset(root_dirs=[os.path.join(self.root, 'location_C')])
      train_dataset = RadarDataset(root_dirs=[os.path.join(self.root, 'location_A'), os.path.join(self.root, 'location_B')])
    else:
      raise ValueError('Invalid location')
    
    return train_dataset, val_dataset

  def setup(self, stage=None):
    if stage == 'fit' or stage is None:
      self.train_dataset, self.val_dataset = self.load_folds()

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
  
  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
  
  def test_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == '__main__':
  dm = RadarDataModule(
    root='dataset',
    location='C',
    batch_size=32,
    num_workers=4
  )

  dm.prepare_data()
  dm.setup()

  train_loader = dm.train_dataloader()
  val_loader = dm.val_dataloader()

  for batch in train_loader:
    red = batch['red']
    green = batch['green']
    blue = batch['blue']
    gray = batch['gray']
    color = batch['color']
    label = batch['label']

    print(f'Train Red: {red.shape}')
    print(f'Train Green: {green.shape}')
    print(f'Train Blue: {blue.shape}')
    print(f'Train Gray: {gray.shape}')
    print(f'Train Color: {color.shape}')
    print(f'Train Label: {label.shape}')

    break

  for batch in val_loader:
    red = batch['red']
    green = batch['green']
    blue = batch['blue']
    gray = batch['gray']
    color = batch['color']
    label = batch['label']

    print(f'Val Red: {red.shape}')
    print(f'Val Green: {green.shape}')
    print(f'Val Blue: {blue.shape}')
    print(f'Val Gray: {gray.shape}')
    print(f'Val Color: {color.shape}')
    print(f'Val Label: {label.shape}')
    break


