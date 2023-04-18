import os
import pandas as pd
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.python.data import AUTOTUNE

image_dataset_dir = '/home/ubuntu/Project/Data/'

IMAGE_SIZE = 512
CHANNELS = 3
n_epoch = 10
PATH = image_dataset_dir
BATCH_SIZE = 70

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
        self.out = tf.keras.layers.Dense(7, activation='softmax')



def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Binary   target = (1,0)

    :return:
    '''


    class_names = np.sort(xdf_data['target'].unique())

    if target_type == 1:

        x = lambda x: tf.argmax(x == class_names).numpy()

        final_target = xdf_data['target'].apply(x)

        final_target = to_categorical(list(final_target))

        xfinal=[]
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in  (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal

        xdf_data['target_class'] = final_target


    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))

        xdepth = len(class_names)

        final_target = tf.one_hot(target, xdepth)

        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            final_target = xfinal

        xdf_data['target_class'] = final_target

    if target_type == 3:
        # target_class is already done
        pass

    return class_names

def process_path(feature, target):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''

    label = target

    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)
    img = tf.image.per_image_standardization(img)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])

    # augmentation
    img = tf.image.flip_left_right(img)
    img = tf.image.flip_up_down(img)
    img = tf.reshape(img, [-1])

    return img, label

def get_target(num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    end = np.zeros(num_classes)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:])

    return y_target

def train_func(train_ds):
    '''
        train the model
    '''

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience = 100)
    check_point = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='accuracy', save_best_only=True)
    model = model_definition()

    model.fit(train_ds,  epochs=n_epoch, callbacks=[early_stop, check_point])

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open('summary.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

def model_definition():
    model = MyModel()

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=100)
    check_point = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='accuracy',
                                                     save_best_only=True)
    model.fit(train_ds, epochs=n_epoch, callbacks=[early_stop, check_point])

    save_model(model) #print Summary
    return model


def read_data(num_classes):
    '''
          reads the dataset and process the target
    '''

    ds_inputs = np.array(image_dataset_dir + xdf_dset['image_name'])
    ds_targets = get_target(num_classes)

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) # creates a tensor from the image paths and targets

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    return final_ds


if __name__ == "__main__":
    for file in os.listdir(PATH+os.path.sep + "Excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "Excel" + os.path.sep + file

    xdf_data = pd.read_excel('/home/ubuntu/DL/Project/Excel/image_dataset.xlsx')
    print(xdf_data)
    class_names = process_target(2)  # 1: Multiclass 2: Multilabel 3:Binary

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    ## Processing Train dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    train_ds = read_data(OUTPUTS_a)
    train_func(train_ds)




