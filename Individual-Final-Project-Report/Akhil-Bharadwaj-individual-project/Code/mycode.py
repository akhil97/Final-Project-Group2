import tensorflow as tf
import os
from keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization

Image_Size = 100
CHANNELS = 3  # set number of channels to 3 for RGB images

# image dataset path
code_dir = os.getcwd()
os.chdir("..") # Change to the parent directory
project_dir = os.getcwd()
image_dataset_path = project_dir + os.path.sep + 'Data'


class VGG16(tf.keras.Model):
    def __init__(self, num_classes, learning_rate):
        super(VGG16, self).__init__()
        self.vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(Image_Size, Image_Size, CHANNELS))
        for layer in self.vgg16.layers:
            layer.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs):
        x = self.vgg16(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        outputs = self.dense3(x)
        return outputs

class VGG19(tf.keras.Model):
    def __init__(self, num_classes, learning_rate):
        super(VGG19, self).__init__()
        self.vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(Image_Size, Image_Size, CHANNELS))
        for layer in self.vgg19.layers:
            layer.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs):
        x = self.vgg19(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        outputs = self.dense3(x)
        return outputs

class InceptionModel(tf.keras.Model):
    def __init__(self, num_classes, learning_rate):
        super(InceptionModel, self).__init__()

        # Load the pre-trained InceptionV3 model
        self.inceptionv3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',
                                                             input_shape=(Image_Size, Image_Size, CHANNELS))

        # Freeze the weights of the pre-trained layers
        for layer in self.inceptionv3.layers:
            layer.trainable = False

        # Add custom layers on top of the pre-trained model
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs):
        x = self.inceptionv3(inputs)
        x = self.avgpool(x)
        x = self.dense1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        outputs = self.dense3(x)
        return outputs

class ResNet50(tf.keras.Model):
    def __init__(self, num_classes, learning_rate):
        super(ResNet50, self).__init__()
        self.resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(Image_Size, Image_Size, CHANNELS))
        for layer in self.resnet50.layers:
            layer.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs):
        x = self.resnet50(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        outputs = self.dense3(x)
        return outputs

class Xception(tf.keras.Model):
    def __init__(self, num_classes, learning_rate):
        super(Xception, self).__init__()
        self.xception = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(Image_Size, Image_Size, CHANNELS))
        for layer in self.xception.layers:
            layer.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs):
        x = self.xception(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        outputs = self.dense3(x)
        return outputs

if __name__ == "__main__":

    # split data into train and test sets
    x = []
    y = []

    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(x, y)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='VGG16, VGG19, Inception, Resnet, Xception, CNNmodel')
    args = parser.parse_args()
    if args.model == 'VGG16':
        vgg16 = VGG16(num_classes, learning_rate)
        train_model(vgg16, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(vgg16, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'VGG19':
        vgg19 = VGG19(num_classes, learning_rate)
        train_model(vgg19, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(vgg19, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'Inception':
        inception = InceptionModel(num_classes, learning_rate)
        train_model(inception, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(inception, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'Resnet':
        resnet = ResNet50(num_classes, learning_rate)
        train_model(resnet, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(resnet, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'Xception':
        xception = Xception(num_classes, learning_rate)
        train_model(xception, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(xception, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    if args.model == 'CNNmodel':
        model = model_definition(num_classes, learning_rate)
        train_model(model, X_train, Y_train, X_val, Y_val, n_epochs, batch_size)
        evaluate(model, X_train, Y_train, X_val, Y_val, X_test, Y_test)