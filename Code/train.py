import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import random

image_dataset_dir = '/home/ubuntu/DL/Project/Data/'

# Set the target categories
categories = ['coast', 'coast_ship', 'detail', 'land', 'multi', 'ship', 'water'] # Replace with your actual categories

# Set the split ratios for train, validation, and test sets
train_split_ratio = 0.8
val_split_ratio = 0.10
test_split_ratio = 0.10

# Create an empty list to store the image file paths and their corresponding labels
data = []

# Iterate through the categories and their respective directories
for category in categories:
    category_dir = os.path.join(image_dataset_dir, category)

    # Get a list of image file names in the category directory
    image_names = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]

    # Randomly shuffle the list of image file names
    random.shuffle(image_names)

    # Calculate the number of images for each split
    num_train_images = int(len(image_names) * train_split_ratio)
    num_val_images = int(len(image_names) * val_split_ratio)
    num_test_images = int(len(image_names) * test_split_ratio)

    # Assign the split for each image file and add the file path and label to the data list
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(category_dir, image_name)
        if i < num_train_images:
            split = 'train'
        elif i < num_train_images + num_val_images:
            split = 'val'
        else:
            split = 'test'
        data.append([image_name, category, split])

# Convert the data list to a DataFrame
data_df = pd.DataFrame(data, columns=['image_name', 'category', 'split'])

# Save the DataFrame to an Excel file
data_df.to_excel('image_dataset.xlsx', index=False)

data_df_train = data_df[data_df["split"] == 'train']

for i, row in data_df.iterrows():
    # Load the image from disk
    img = cv2.imread(row['Image Name'])

    # Resize the image to a standard size
    img = cv2.resize(img, (512, 512))

    # Normalize the pixel values to be between 0 and 1
    img = img.astype(np.float32) / 255.0

    # Convert the color channels to match the expected input format of the CNN model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(2)
        self.drop1 = tf.keras.layers.Dropout(0.2)

        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(2)
        self.drop2 = tf.keras.layers.Dropout(0.2)

        self.flatten = tf.keras.layers.Flatten()

        self.fc = tf.keras.layers.Dense(256)
        self.out = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = self.x1(inputs)
        return self.predictions(x)

model = MyModel()