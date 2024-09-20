import cv2
import torch
import numpy as np
import albumentations as A

# from torch.distributions import Bernoulli
from albumentations.core.transforms_interface import BasicTransform
# from albumentations.core.transforms_interface import ImageOnlyTransform

class ToWBM(BasicTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super(ToWBM, self).__init__(always_apply, p)

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img: np.ndarray, **kwargs):
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, None]
            img = torch.from_numpy(img.transpose((2, 0, 1)))
            if isinstance(img, torch.ByteTensor):
                img = img.float().div(255)
        return torch.ceil(img * 2);

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}

class WM811KTransform(object):
    def __init__(self,
                 size: tuple = (96, 96),
                 mode: str = 'basic',
                 **kwargs):

        if isinstance(size, int):
            size = (size, size)
        defaults = dict(size=size, mode=mode)
        defaults.update(kwargs)
        self.defaults = defaults

        if mode == 'basic':
            transform = self.basic_transform(**defaults)
        else:
            raise NotImplementedError

        self.transform = A.Compose(transform)

    def __call__(self, img):
        return self.transform(image=img)['image']

    def __repr__(self):
        repr_str = self.__class__.__name__
        for k, v in self.defaults.items():
            repr_str += f"\n{k}: {v}"
        return repr_str

    @staticmethod
    def basic_transform(size: tuple, **kwargs) -> list:
        tranform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

        return tranform