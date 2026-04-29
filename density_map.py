import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import os

def load_ground_truth(mat_file_path):
    """Load head coordinates from .mat annotation file"""
    mat = loadmat(mat_file_path)
    
    points = mat['image_info'][0][0][0][0][0]
    return points  

def generate_density_map(image_shape, points, sigma=15):
    
    density = np.zeros(image_shape, dtype=np.float32)

    for point in points:
        x, y = int(point[0]), int(point[1])

        
        if x >= image_shape[1] or y >= image_shape[0]:
            continue
        if x < 0 or y < 0:
            continue

        
        density[y][x] = 1

    # Spread each dot into a Gaussian blob
    density = gaussian_filter(density, sigma=sigma)

    return density

def get_count_from_density(density_map):
    
    return density_map.sum()