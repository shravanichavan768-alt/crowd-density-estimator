import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from density_map import generate_density_map

class ShanghaiTechDataset(Dataset):
    def __init__(self, data_path, part='B', split='train', sigma=15):
        self.data_path = data_path
        self.sigma = sigma

        # Build paths
        split_folder = 'train_data' if split == 'train' else 'test_data'
        self.img_dir = os.path.join(data_path, f'part_{part}', split_folder, 'images')
        self.gt_dir  = os.path.join(data_path, f'part_{part}', split_folder, 'ground_truth')

        # Get all image filenames
        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.jpg')
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        #  Load image 
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        #  Load ground truth
        gt_name = 'GT_' + img_name.replace('.jpg', '.mat')
        gt_path = os.path.join(self.gt_dir, gt_name)

        mat = loadmat(gt_path)
        points = mat['image_info'][0][0][0][0][0]  # (N, 2) head coords

        # ── Generate density map ──
        h, w = image.shape[1], image.shape[2]
        density = generate_density_map((h, w), points, sigma=self.sigma)

        # Downsample density map to 1/8th size (CSRNet output size)
        density_tensor = torch.from_numpy(density).unsqueeze(0).float()
        density_small  = torch.nn.functional.interpolate(
            density_tensor.unsqueeze(0),
            scale_factor=0.125,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Scale density so sum is preserved
        density_small = density_small * 64  # 8x8 

        return image, density_small