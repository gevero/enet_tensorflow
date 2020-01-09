# importing standard libraries
import tensorflow as tf
import numpy as np
import datetime

# Importing utils and models
from utils import preprocess_img_label, map_singlehead, map_doublehead
from utils import tf_dataset_generator, get_class_weights, map_label
from models import EnetModel

# -------------------- Defining the hyperparameters --------------------
batch_size = 8
epochs = 50
training_type = 0
learning_rate = 5e-4
num_classes = 12
weight_decay = 2e-4
img_pattern = "./datasets/CamVid/train/images/*.png"
label_pattern = "./datasets/CamVid/train/labels/*.png"
img_pattern_val = "./datasets/CamVid/val/images/*.png"
label_pattern_val = "./datasets/CamVid/train/labels/*.png"
tb_logs = './tb_logs/'
img_width = 480
img_height = 360
save_model = './Enet.tf'
cache_train = ''
cache_val = ''
cache_test = ''
print('[INFO]Defined all the hyperparameters successfully!')

# setup tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

# encoder and decoder dimensions
h_enc = img_height // 8
w_enc = img_width // 8
h_dec = img_height
w_dec = img_width

# create (img,label) string tensor lists
filelist_train = preprocess_img_label(img_pattern, label_pattern)
filelist_val = preprocess_img_label(img_pattern_val, label_pattern_val)

# training dataset size
n_train = tf.data.experimental.cardinality(filelist_train).numpy()
n_val = tf.data.experimental.cardinality(filelist_val).numpy()

# define mapping functions for single and double head nets
map_single = lambda img_file, label_file: map_singlehead(
    img_file, label_file, h_dec, w_dec)
map_double = lambda img_file, label_file: map_doublehead(
    img_file, label_file, h_enc, w_enc, h_dec, w_dec)

# create dataset
if training_type == 0 or training_type == 1:
    map_fn = map_double
else:
    map_fn = map_single
train_ds = filelist_train.shuffle(n_train).map(map_fn).cache(
    cache_train).batch(batch_size).repeat()
val_ds = filelist_val.map(map_fn).cache(cache_val).batch(batch_size).repeat()

# final training and validation datasets

# -------------------- get the class weights --------------------
print('[INFO]Starting to define the class weights...')
label_filelist = tf.data.Dataset.list_files(label_pattern, shuffle=False)
label_ds = label_filelist.map(lambda x: map_label(x, h_dec, w_dec))
class_weights = get_class_weights(label_ds).tolist()
print('[INFO]Fetched all class weights successfully!')

# -------------------- istantiate model --------------------
if training_type == 0 or training_type == 1:
    Enet = EnetModel(C=num_classes, MultiObjective=True, l2=weight_decay)
else:
    Enet = EnetModel(C=num_classes, l2=weight_decay)
print('[INFO]Model Instantiated!')

# -------------------- start training --------------------
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# -- two stages training --
if training_type == 0:

    # freeze decoder layers
    for layer in Enet.layers[-6:]:
        layer.trainable = False

    # compile encoder: only the first objective matters
    Enet.compile(optimizer=adam_optimizer,
                 loss=[
                     'sparse_categorical_crossentropy',
                     'sparse_categorical_crossentropy'
                 ],
                 metrics=['accuracy', 'accuracy'],
                 loss_weights=[1.0, 0.0])

    # train encoder
    Enet.fit(x=train_ds,
             epochs=epochs,
             steps_per_epoch=n_train // batch_size,
             validation_data=val_ds,
             validation_steps=n_val // batch_size // 5,
             class_weight=[class_weights, class_weights],
             callbacks=[tensorboard_callback])

    # freeze encoder and unfreeze decoder
    for layer in Enet.layers[-6:]:
        layer.trainable = True
    for layer in Enet.layers[:-6]:
        layer.trainable = False

    # compile model: only the second objective matters
    Enet.compile(optimizer=adam_optimizer,
                 loss=[
                     'sparse_categorical_crossentropy',
                     'sparse_categorical_crossentropy'
                 ],
                 metrics=['accuracy', 'accuracy'],
                 loss_weights=[0.0, 1.0])

    # train decoder
    enet_hist = Enet.fit(x=train_ds,
                         epochs=epochs,
                         steps_per_epoch=n_train // batch_size,
                         validation_data=val_ds,
                         validation_steps=n_val // batch_size // 5,
                         class_weight=[class_weights, class_weights],
                         callbacks=[tensorboard_callback])

# -- simultaneous double objective trainings --
elif training_type == 1:

    # compile model
    Enet.compile(optimizer=adam_optimizer,
                 loss=[
                     'sparse_categorical_crossentropy',
                     'sparse_categorical_crossentropy'
                 ],
                 metrics=['accuracy', 'accuracy'],
                 loss_weights=[0.5, 0.5])

    # fit model
    print('train: ', n_train, 'batch: ', batch_size)
    enet_hist = Enet.fit(x=train_ds,
                         epochs=epochs,
                         steps_per_epoch=n_train // batch_size,
                         validation_data=val_ds,
                         validation_steps=n_val // batch_size // 5,
                         class_weight=[class_weights, class_weights],
                         callbacks=[tensorboard_callback])

# -- end to end training --
else:

    # compile model
    Enet.compile(optimizer=adam_optimizer,
                 loss=['sparse_categorical_crossentropy'],
                 metrics=['accuracy'])

    enet_hist = Enet.fit(x=train_ds,
                         epochs=epochs,
                         steps_per_epoch=n_train // batch_size,
                         validation_data=val_ds,
                         validation_steps=n_val // batch_size // 5,
                         class_weight=class_weights,
                         callbacks=[tensorboard_callback])

# -------------------- save model --------------------
Enet.save_weights(save_model)
