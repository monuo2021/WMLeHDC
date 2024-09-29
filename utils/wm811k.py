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

    def __init__(self, root, transform=None, filter_type='gaussian', kernel_size=5, cache_dir='./cache', use_cache=True, **kwargs):
        super(WM811K, self).__init__()

        self.root = root
        self.transform = transform
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.filter_type = filter_type
        self.kernel_size = kernel_size

        images  = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))
        labels  = [pathlib.PurePath(image).parent.name for image in images]
        targets = [self.label2idx[l] for l in labels]
        samples = list(zip(images, targets))

        self.samples = samples

        # 创建缓存目录
        if self.use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            self.cache_preprocessed_images()

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        # x = self.load_image_cv2(path)

        if self.use_cache:
            # 从缓存加载图像
            x = self.load_cached_image(idx)
        else:
            # 每次调用时加载图像
            x = self.load_image_cv2(path)

        # 应用滤波处理
        x = self.apply_filter(x)  # 选择合适的滤波类型和大小

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

    def cache_preprocessed_images(self):
        """缓存预处理后的图像到磁盘"""
        print("Caching preprocessed images...")
        for idx, (path, _) in enumerate(self.samples):
            img_path = os.path.join(self.cache_dir, f'{idx}.npy')
            if not os.path.exists(img_path):
                img = self.load_image_cv2(path)
                np.save(img_path, img)

    def load_cached_image(self, idx):
        """从缓存加载图像"""
        return np.load(os.path.join(self.cache_dir, f'{idx}.npy'))

    def apply_filter(self, image):
        if self.filter_type == 'gaussian':
            return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        elif self.filter_type == 'median':
            return cv2.medianBlur(image, self.kernel_size)
        else:
            raise ValueError("Unsupported filter type. Choose 'gaussian' or 'median'.")