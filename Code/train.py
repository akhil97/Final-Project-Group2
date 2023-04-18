import numpy as np
import os
import cv2
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from sklearn.neighbors import KNeighborsClassifier

# define the categories and image dataset path
CATEGORIES = ['coast', 'coast_ship', "detail", "land", "multi", "ship", "water"]
image_dataset_path = "/home/ubuntu/Project/Data/"

data = []
width = 100
height = 100

# check if preprocessed data exists
already_preprocessed = os.path.exists('x_train.pkl') and os.path.exists('y_train.pkl') and os.path.exists(
    'x_test.pkl') and os.path.exists('y_test.pkl') and os.path.exists(
    'x_val.pkl') and os.path.exists('y_val.pkl')


if not already_preprocessed:
    # read and preprocess the images
    for category in CATEGORIES:
        path = os.path.join(image_dataset_path, category) # path of dataset
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img) #Getting the image path
                label = CATEGORIES.index(category)# Assigning label to image
                arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Converting image to grey scale
                new_arr = cv2.resize(arr, (100, 100)) # Resize image
                data.append([new_arr, label]) # appedning image and label in list
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

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)  # 0.125 = 0.1 / 0.8


    # save preprocessed data to pickle files
    with open('x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)

    with open('y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)

    with open('x_val.pkl', 'wb') as f:
        pickle.dump(x_val, f)

    with open('y_val.pkl', 'wb') as f:
        pickle.dump(y_val, f)

    with open('x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)

    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
else:
    # load preprocessed data from pickle files
    x_train = pickle.load(open('x_train.pkl', 'rb'))
    y_train = pickle.load(open('y_train.pkl', 'rb'))
    x_val = pickle.load(open('x_val.pkl', 'rb'))
    y_val = pickle.load(open('y_val.pkl', 'rb'))
    x_test = pickle.load(open('x_test.pkl', 'rb'))
    y_test = pickle.load(open('y_test.pkl', 'rb'))

#load
# normalize images
x_train = x_train / 255
x_val = x_val / 255
x_test = x_test / 255

# reshape images for CNN
x_train = x_train.reshape(-1, width, height, 1)
x_val = x_val.reshape(-1, width, height, 1)
x_test = x_test.reshape(-1, width, height, 1)


# define the CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(7, activation='softmax'))

cnn_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the CNN model
cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# evaluate the CNN model
print("CNN for test:", cnn_model.evaluate(x_test, y_test))  # evaluate test data
print("CNN for train:",cnn_model.evaluate(x_train, y_train))  # evaluate train data

# extract features(Neural Code) from the CNN model
train_features = cnn_model.predict(x_train)
val_features = cnn_model.predict(x_val)
test_features = cnn_model.predict(x_test)

# train and evaluate KNN model
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(train_features, y_train)
print("KNN for validation: ", neigh.score(val_features, y_val))
print("KNN for test: ", neigh.score(test_features, y_test))
print("KNN for train: ", neigh.score(train_features, y_train))

