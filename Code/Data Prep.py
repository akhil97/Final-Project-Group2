import numpy as np
import os
import cv2
import pickle
from sklearn.model_selection import train_test_split

# define the categories and image dataset path
CATEGORIES = ['coast', 'coast_ship', "detail", "land", "multi", "ship", "water"]
image_dataset_path = "/home/ubuntu/Project/Data/"

data = []
width = 100
height = 100

# read and preprocess the images
for category in CATEGORIES:
    path = os.path.join(image_dataset_path, category)  # path of dataset
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)  # Getting the image path
            label = CATEGORIES.index(category)  # Assigning label to image
            arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Converting image to grey scale
            new_arr = cv2.resize(arr, (100, 100))  # Resize image
            data.append([new_arr, label])  # appedning image and label in list
        except Exception as e:
            print(str(e))

# split data into train and test sets
x = []
y = []

for features, label in data:
    x.append(features)  # Storing Images all images in X
    y.append(label)  # Storing al image label in y

x = np.array(x)  # Converting it into Numpy Array
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# save preprocessed data to pickle files
with open('x_train.pkl', 'wb') as f:
    pickle.dump(x_train, f)

with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('x_test.pkl', 'wb') as f:
    pickle.dump(x_test, f)

with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)