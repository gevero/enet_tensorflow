# importing standard libraries
import tensorflow as tf
import numpy as np
import datetime

# Importing utils and models
from functools import partial
from utils import preprocess_img_label, map_singlehead, map_doublehead
from utils import tf_dataset_generator, get_class_weights
from models import EnetModel


def train(FLAGS):

    # -------------------- Defining the hyperparameters --------------------
    batch_size = FLAGS.batch_size  #
    epochs = FLAGS.epochs  #
    training_type = FLAGS.training_type  #
    learning_rate = FLAGS.learning_rate  #
    save_every = FLAGS.save_every  #
    num_classes = FLAGS.num_classes  #
    weight_decay = FLAGS.weight_decay
    img_pattern = FLAGS.img_pattern  #
    label_pattern = FLAGS.label_pattern  #
    img_pattern_val = FLAGS.img_pattern_val  #
    label_pattern_val = FLAGS.label_pattern_val  #
    tb_logs = FLAGS.tensorboard_logs  #
    img_width = FLAGS.img_width  #
    img_height = FLAGS.img_height  #
    save_model = FLAGS.save_model  #
    cache_train = FLAGS.cache_train  #
    cache_val = FLAGS.cache_val  #
    cache_test = FLAGS.cache_test  #
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
    map_single = partial(map_singlehead, h_img=h_dec, w_img=w_dec)
    map_double = partial(map_doublehead,
                         h_enc=h_enc,
                         w_enc=w_enc,
                         h_dec=h_dec,
                         w_dec=w_dec)

    # create dataset
    if training_type == 0 or training_type == 1:
        map_fn = map_double
    else:
        map_fn = map_single
    train_ds = filelist_train.shuffle(n_train).map(map_fn).cache(
        cache_train).batch(batch_size).repeat()
    val_ds = filelist_val.map(map_fn).cache(cache_val).batch(
        batch_size).repeat()

    # final training and validation datasets

    # -------------------- get the class weights --------------------
    print('[INFO]Starting to define the class weights...')
    label_filelist = tf.data.Dataset.list_files(label_pattern, shuffle=False)
    label_ds = label_filelist.map(lambda x: process_label(x, h_dec, w_dec))
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
                 class_weight=class_weights,
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
                             class_weight=class_weights,
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
                             class_weight=class_weights,
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
