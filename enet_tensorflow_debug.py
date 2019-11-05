# importing standard libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds

# Importing utils and models
from utils import process_path, tf_dataset_generator, get_class_weights
from models import EnetEncoder, TestModel
print(tf.__version__)

# creating datasets
train_path = 'dataset/train/images'
val_path = 'dataset/val/images'
test_path = 'dataset/test/images'
train_ds = tf_dataset_generator(train_path, batch_size=8)
val_ds = tf_dataset_generator(val_path, train=False, cache=False, batch_size=8)
test_ds = tf_dataset_generator(test_path,
                               train=False,
                               cache=False,
                               batch_size=8)

list_ds = tf.data.Dataset.list_files(train_path + '/*')
data_set = list_ds.map(process_path,
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)

# get class weights
class_weights = get_class_weights(
    tf_dataset_generator(train_path, train=False, cache=False))

# define model
Enet = EnetEncoder(C=12)
for img, iml in train_ds.take(1):
    img_test = img
    iml_test = iml
img_out = Enet(img_test)

# compile model
Enet.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# train model
Enet.fit(x=train_ds, epochs=10, steps_per_epoch=367 // 8)
