import cv2
import numpy as np
import os

#Input and Output folders
data_dir = "C:/Users/PRJWL/Documents/Facial Age Estimation - Thesis/Datasets/UTKFace/UTKFaceR"
res_dir = "C:/Users/PRJWL/Documents/Facial Age Estimation - Thesis/Datasets/UTKFace/processed/histogram_equalized"

# Contrast limit for adaptive histogram equalization
clahe_limit = 1.8

# Travesr through all the images in the folder
for file_name in os.listdir(data_dir):
    if file_name.endswith(".jpg"):
        # Load the image
        img_path = os.path.join(data_dir, file_name)
        img = cv2.imread(img_path)

        # Resize image to input size of the CNN model
        img = cv2.resize(img, (224, 224))

        # Apply histogram equalization to the image
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clahe_limit)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # Save processed images 
        res_path = os.path.join(res_dir, file_name)
        cv2.imwrite(res_path, img)
