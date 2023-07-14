import cv2
import numpy as np
import os

#Input and Output folders
data_dir = "C:/Users/PRJWL/Documents/Facial Age Estimation - Thesis/Datasets/UTKFace/UTKFaceR"
res_dir = "C:/Users/PRJWL/Documents/Facial Age Estimation - Thesis/Datasets/UTKFace/processed/gaussian_smoothed"

# Input image size to the CNN model
input_size = (128, 128)

# Standard deviation of the Gaussian kernel
sigma = 0.5

# Set size of the Gaussian kernel
kernel_size = (3, 3)

# Traverse all images in the folder
for file_name in os.listdir(data_dir):
    if file_name.endswith(".jpg"):
        # Load the image file
        img_path = os.path.join(data_dir, file_name)
        img = cv2.imread(img_path)

        # Resize the image to the input size of the CNN model
        img = cv2.resize(img, input_size)

        # Apply Gaussian smoothing 
        img = cv2.GaussianBlur(img, kernel_size, sigma)

        # Save the processed image 
        res_path = os.path.join(res_dir, file_name)
        cv2.imwrite(res_path, img)
