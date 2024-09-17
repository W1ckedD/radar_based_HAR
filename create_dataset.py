import os
import shutil

def train_val_test_split(root_dir, output_dir, classes=[1, 2, 3, 4, 5, 6], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
  # Create the output directory
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
  os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
  os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

  class_samples = {cls : [] for cls in classes}

  for file in os.listdir(root_dir):
    class_name = file[0]
    class_samples[int(class_name)].append(file)

  for cls, samples in class_samples.items():
    import random
    random.shuffle(samples)

    train_samples = samples[:int(len(samples) * train_ratio)]
    val_samples = samples[int(len(samples) * train_ratio):int(len(samples) * (train_ratio + val_ratio))]
    test_samples = samples[int(len(samples) * (train_ratio + val_ratio)):]


    for sample in train_samples:
      # copy the entire sample directory
      shutil.copytree(os.path.join(root_dir, sample), os.path.join(output_dir, 'train', sample))
    for sample in val_samples:
      # copy the entire sample directory
      shutil.copytree(os.path.join(root_dir, sample), os.path.join(output_dir, 'val', sample))
    for sample in test_samples:
      # copy the entire sample directory
      shutil.copytree(os.path.join(root_dir, sample), os.path.join(output_dir, 'test', sample))



def main(root_dir, output_dir):
  # Create the output directory
  os.makedirs(output_dir, exist_ok=True)

  # Copy the data from the root directory to the output directory
  for file in os.listdir(root_dir):
    base_name = file.split('_')[0]
    os.makedirs(os.path.join(output_dir, base_name), exist_ok=True)
    shutil.copy(os.path.join(root_dir, file), os.path.join(output_dir, base_name, file))

def count_class_samples(root_dir):
  class_samples = {
    'class_1': 0,
    'class_2': 0,
    'class_3': 0,
    'class_4': 0,
    'class_5': 0,
    'class_6': 0
  }

  for file in os.listdir(root_dir):
    class_name = file[0]
    class_samples[f"class_{class_name}"] += 1

  return class_samples


if __name__ == '__main__':
  root_dirs_A = [
    'Dataset_848_processed/1 December 2017 Dataset',
    'Dataset_848_processed/2 March 2017 Dataset',
    'Dataset_848_processed/3 June 2017 Dataset',
    'Dataset_848_processed/5 February 2019 UoG Dataset',
  ]


  output_dir_A = 'dataset/location_A'

  root_dirs_B = [
    'Dataset_848_processed/6 February 2019 NG Homes Dataset_processed',
    'Dataset_848_processed/7 March 2019 West Cumbria Dataset_processed'
  ]

  output_dir_B = 'dataset/location_B'

  root_dirs_C = [
    'Dataset_848_processed/4 July 2018 Dataset'
  ]

  output_dir_C = 'dataset/location_C'

  for root_dir in root_dirs_A:
    main(root_dir, output_dir_A)

  for root_dir in root_dirs_B:
    main(root_dir, output_dir_B)

  for root_dir in root_dirs_C:
    main(root_dir, output_dir_C)

  

  # train_val_test_split('dataset/location_A', 'dataset_2/location_A',classes=[1, 2, 3, 4, 5, 6], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)