from PIL import Image
import os

#Input and Output folders
data_dir = "C:/Users/PRJWL/Documents/Facial Age Estimation - Thesis/Datasets/UTKFace/UTKFaceR"
res_dir = "C:/Users/PRJWL/Documents/Facial Age Estimation - Thesis/Datasets/UTKFace/processed/grayscaled"

for file_name in os.listdir(data_dir):
    if file_name.endswith(".jpg"):
        # Load the image file
        image = Image.open(os.path.join(data_dir, file_name))
        
        # Apply ocnversin
        gray_image = image.convert('L')

        res_path = os.path.join(res_dir, file_name)

        # Save the prcessed image
        gray_image.save(res_dir)
