import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import time

# Start timer to calcualte the execution time of the code
start_time = time.time()

# Set the path to the UTKFace dataset
dataset_path = "C:/Users/PRJWL/Documents/Facial Age Estimation - Thesis/Datasets/FGNET/images"
print("Training data obtained from: ", dataset_path)

# Load images and the corresponding age labels
images = []
ages = []
for filename in os.listdir(dataset_path):
    #age = int(filename.split("_")[1]) #Label loading for UTKFace 
    age = int(filename[4:6]) #Label loading for UTKFace 
    img = cv2.imread(os.path.join(dataset_path, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200))
    img = np.array(img) / 255.0
    images.append(img)
    ages.append(age)

# Convert the images and age labels list in to numpy array
images = np.array(images)
ages = np.array(ages)

# Split the dataset into training and validation sets
split = int(len(images) * 0.8)
x_train = images[:split]
y_train = ages[:split]
x_val = images[split:]
y_val = ages[split:]

# Defining the CNN model 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

# Compile the model
model.compile(
    optimizer='adam', 
    loss='mse', 
    metrics=['mae']
    )

# Model training
history = model.fit(x_train, 
                    y_train, 
                    epochs=20, 
                    batch_size=32, 
                    validation_data=(x_val, y_val)
                    )

# Evaluate the model's performance on the test split
x_test = images[split:]
y_test = ages[split:]
model_result = model.evaluate(x_test, y_test)
print("--- %s seconds ---" % (time.time() - start_time))