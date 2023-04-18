import tensorflow as tf

class VGG16(tf.keras.Model):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
        for layer in self.vgg16.layers:
            layer.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.vgg16(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        outputs = self.dense2(x)
        return outputs


class VGG19(tf.keras.Model):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
        for layer in self.vgg19.layers:
            layer.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.vgg19(inputs)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        outputs = self.dense2(x)
        return outputs

class InceptionModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(InceptionModel, self).__init__()

        # Load the pre-trained InceptionV3 model
        self.inceptionv3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',
                                                             input_shape=(512, 512, 3))

        # Freeze the weights of the pre-trained layers
        for layer in self.inceptionv3.layers:
            layer.trainable = False

        # Add custom layers on top of the pre-trained model
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.inceptionv3(inputs)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.fc1(x)
        output = self.fc2(x)
        return output


class ResNet50(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
        for layer in self.resnet50.layers:
            layer.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.resnet50(inputs)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        outputs = self.dense2(x)
        return outputs


class Xception(tf.keras.Model):
    def __init__(self, num_classes):
        super(Xception, self).__init__()
        self.xception = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
        for layer in self.xception.layers:
            layer.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.xception(inputs)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        outputs = self.dense2(x)
        return outputs