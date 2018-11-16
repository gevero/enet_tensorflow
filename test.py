import tensorflow as tf
import numpy as np

# import encoders
import sys
sys.path.append('../')
import encoders

# import dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# reshape dataset
x_train = x_train.reshape([-1,28,28,1])
x_test = x_test.reshape([-1,28,28,1])
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# input layer
input_layer_enet = tf.keras.layers.Input(shape=(28,28,1))

# mnist model enet
output_layer_enet = encoders.enet_encoder_mnist(input_layer_enet)
mnist_model_enet = tf.keras.Model(inputs=input_layer_enet, outputs=output_layer_enet)
