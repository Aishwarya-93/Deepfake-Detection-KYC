"""
Image Preprocessing Functions

Author: Preesh
"""

import cv2
import numpy as np


def resize_image(image, size=(224, 224)):
    """
    Resize an image.
    """
    return cv2.resize(image, size)


def normalize_image(image):
    """
    Normalize image to range 0–1.
    """
    return image.astype(np.float32) / 255.0
     

def bgr_to_rgb(image):
    """
    Convert OpenCV BGR image to RGB.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)