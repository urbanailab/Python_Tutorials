import os

import torch
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, targets, transform=None):
        self.root_dir = root_dir
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Load and preprocess image and target
        image_path = os.path.join(self.root_dir, f'{idx + 1}.jpg')
        image = Image.open(image_path).convert('RGB')
        target = self.targets[idx]
        if self.transform is not None:
          image = self.transform(image)
        return image, target