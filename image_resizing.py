from PIL import Image
import os

#Input and Output folders
data_dir = "C:/Users/PRJWL/Documents/Facial Age Estimation - Thesis/Datasets/UTKFace/UTKFaceR"
res_dir = "C:/Users/PRJWL/Documents/Facial Age Estimation - Thesis/Datasets/UTKFace/processed/resized"

# Set new size 
new_size = (128, 128)

# Travesr through all the images in the folder
for filename in os.listdir(data_dir):
    if filename.endswith(".jpg"):
        # Open the image
        image = Image.open(os.path.join(data_dir, filename))
        
        # Resize the image
        resized_image = image.resize(new_size)
        
        # Save the new image
        resized_image.save(os.path.join(res_dir, filename))