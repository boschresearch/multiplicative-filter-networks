import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import numpy as np
from PIL import Image
import skimage


class CameraDataset(Dataset):
    def __init__(self, side_length=None):

        self.image = Image.fromarray(skimage.data.camera())

        if side_length is None:
            side_length = self.image.shape[-1]
        
        self.side_length = side_length

        self.transform = Compose(
            [
                Resize(side_length),
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
            ]
        )

        self.coords = torch.stack(
            torch.meshgrid(
                [
                    torch.linspace(-1.0, 1.0, side_length),
                    torch.linspace(-1.0, 1.0, side_length),
                ]
            ),
            dim=-1,
        ).view(-1, 2)
        return

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.transform(self.image).reshape(-1, 1)


class VideoDataset(Dataset):
    """
    The video argument should be a 3d numpy array with time/frame as the first dimension.
    """
    def __init__(self, video):

        self.video = torch.tensor(np.load(video))
        self.video = 2*self.video - 1. # normalize to [-1, 1]
        self.timesteps = self.video.shape[0]
        self.side_length = self.video.shape[1]
        mesh_sizes = self.video.shape[:-1]
        self.video = self.video.view(-1, 3)

        self.coords = torch.stack(
            torch.meshgrid([torch.linspace(-1.0, 1.0, s) for s in mesh_sizes]), dim=-1
        ).view(-1, 3)

        return

    def __len__(self):
        return self.video.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.video[idx]
