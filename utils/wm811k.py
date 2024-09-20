import os
import glob
import pathlib
from random import sample

import numpy as np
import torch
import cv2

from PIL import Image
from torch.utils.data import Dataset

class WM811K(Dataset):
    label2idx = {
        'center': 0,
        'donut': 1,
        'edge-loc': 2,
        'edge-ring': 3,
        'loc': 4,
        'random': 5,
        'scratch': 6,
        'near-full': 7,
        'none': 8,
        '-': 9,
    }
    idx2label = [k for k in label2idx.keys()]
    num_classes = len(idx2label) - 1

    def __init__(self, root, transform=None, **kwargs):
        super(WM811K, self).__init__()

        self.root = root
        self.transform = transform

        images  = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))
        labels  = [pathlib.PurePath(image).parent.name for image in images]
        targets = [self.label2idx[l] for l in labels]
        samples = list(zip(images, targets))

        self.samples = samples

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = self.load_image_cv2(path)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image_cv2(filepath: str):
        """Load image with cv2. Use with `albumentations`."""
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return np.expand_dims(img, axis=2)