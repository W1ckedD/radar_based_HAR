import torch
import wandb
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from argparse import ArgumentParser
from src.model.model import EarlyFusion
from src.data.dataset import RadarDataModule

def main(args):
  # Set up the data module
  dm = RadarDataModule(root=args.root, location=args.location, batch_size=args.batch_size, num_workers=args.num_workers)

  # Set up the model
  model = EarlyFusion(mode=args.mode)
  
  # Set up the logger
  logger = WandbLogger(project='radar', name=f"{args.name}_{args.mode}", log_model=True)

  # Set up the callbacks
  checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, dirpath='checkpoints', filename='early_fusion' + args.mode + '_{epoch:02d}_{val_loss:.2f}')
  early_stop = EarlyStopping(monitor='val_loss', patience=15)

  # Set up the trainer
  trainer = lightning.Trainer(
    logger=logger,
    callbacks=[checkpoint, early_stop],
    max_epochs=args.epochs,
    accelerator=args.accelerator,
    strategy='auto'
  )

  # Train the model
  trainer.fit(model, dm)
  trainer.test(model, datamodule=dm, ckpt_path='best')

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--root', type=str, default='dataset', help='Root directory for the dataset')
  parser.add_argument('--location', type=str, default='A', help='Location for the dataset')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
  parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
  parser.add_argument('--mode', type=str, default='red', help='Mode for the model')
  parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
  parser.add_argument('--name', type=str, default='early_fusion', help='Name of the run')
  parser.add_argument('--accelerator', type=str, default='cuda', help='Accelerator for training')
  args = parser.parse_args()

  main(args)
