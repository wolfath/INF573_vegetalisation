from PIL import Image
import random
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import hex2color
import numpy as np
from pathlib import Path
import cv2
from CNN.utils_cnn import convert_to_color

def get_data_paths (path, filter):
    """
    Gets the path of the images.
    """
    for path in Path(path).rglob(filter):
         yield path.resolve().as_posix()


def read_tif_image(path):
    """Gets the numpy array of a TIF image from its path"""
    with rasterio.open(path, 'r') as f:
        imageR = f.read([1])[0]
        imageG = f.read([2])[0]
        imageB = f.read([3])[0]
        imageIR = f.read([4])[0]
        imageE = f.read([5])[0]
    image = np.stack([imageR, imageG, imageB, imageIR, imageE], axis=-1)
    return image
    
def read_tif_mask(path):
    """Gets the numpy array of a TIF mask from its path"""
    with rasterio.open(path, 'r') as f:
        mk = f.read([1])
    return mk[0]


def read_tif_mask_prediction(path):
    """Gets the numpy array of a TIF mask from its path"""
    with rasterio.open(path, 'r') as f:
        mk = f.read([1]) + 1
    return mk[0]
    