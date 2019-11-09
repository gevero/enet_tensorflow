# importing standard libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds

# Importing utils and models
from utils import process_path, process_path_double_obj, tf_dataset_generator, get_class_weights
from models import EnetModel
print(tf.__version__)

# creating datasets for single objective
train_path = 'dataset/train/images'
val_path = 'dataset/val/images'
test_path = 'dataset/test/images'
train_ds = tf_dataset_generator(train_path, process_path, batch_size=8)
val_ds = tf_dataset_generator(val_path, process_path, batch_size=8)
test_ds = tf_dataset_generator(test_path, process_path, batch_size=8)

# creating datasets for single objective
train_do_ds = tf_dataset_generator(train_path,
                                   process_path_double_obj,
                                   batch_size=8)
val_do_ds = tf_dataset_generator(val_path,
                                 process_path_double_obj,
                                 batch_size=8)
test_do_ds = tf_dataset_generator(test_path,
                                  process_path_double_obj,
                                  batch_size=8)

list_ds = tf.data.Dataset.list_files(train_path + '/*')
data_set = list_ds.map(process_path_double_obj,
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)

# get class weights
class_weights = get_class_weights(
    tf_dataset_generator(train_path, process_path, train=False, cache=False))

# define single headed model
EnetSingle = EnetModel(C=12)
for img, iml in train_ds.take(1):
    img_test = img
    iml_test = iml
img_out = EnetSingle(img_test)

# define double headed model
EnetDouble = EnetModel(C=12, MultiObjective=True)
for img, iml in train_do_ds.take(1):
    img_do_test = img
    iml_do_test = iml
img_do_out = EnetDouble(img_do_test)

# compile and train double model
EnetDouble.compile(optimizer='adam',
                   loss=[
                       'sparse_categorical_crossentropy',
                       'sparse_categorical_crossentropy'
                   ],
                   metrics=['accuracy', 'accuracy'])
EnetDouble.fit(x=train_do_ds, epochs=1, steps_per_epoch=367 // 8)

# compile and train single model
EnetSingle.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
EnetSingle.fit(x=train_ds, epochs=1, steps_per_epoch=367 // 8)