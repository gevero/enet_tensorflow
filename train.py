# importing standard libraries
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np

# Importing utils and models
from utils import process_img, process_label
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
    print('[INFO]Defined all the hyperparameters successfully!')

    # encoder and decoder dimensions
    h_enc = img_height // 8
    w_enc = img_width // 8
    h_dec = img_height
    w_dec = img_width

    # raw training datasets
    train_img_dec_ds = tf_dataset_generator(
        img_pattern, lambda x: process_img(x, h_dec, w_dec))
    train_label_dec_ds = tf_dataset_generator(
        label_pattern, lambda x: process_label(x, h_dec, w_dec))
    train_label_enc_ds = tf_dataset_generator(
        label_pattern, lambda x: process_label(x, h_enc, w_enc))

    # raw validation datasets
    val_img_dec_ds = tf_dataset_generator(
        img_pattern_val, lambda x: process_img(x, h_dec, w_dec))
    val_label_dec_ds = tf_dataset_generator(
        label_pattern_val, lambda x: process_label(x, h_dec, w_dec))
    val_label_enc_ds = tf_dataset_generator(
        label_pattern_val, lambda x: process_label(x, h_enc, w_enc))

    # final training and validation datasets
    if training_type == 0 or training_type == 1:
        train_ds = tf.data.Dataset.zip(
            (train_img_dec_ds,
             (train_label_enc_ds, train_label_dec_ds))).cache().shuffle(
                 buffer_size=shuffle_buffer_size).batch(batch_size).repeat()
        val_ds = tf.data.Dataset.zip(
            (val_img_dec_ds,
             (val_label_enc_ds, val_label_dec_ds))).cache().shuffle(
                 buffer_size=shuffle_buffer_size).batch(batch_size).repeat()
    else:
        train_ds = tf.data.Dataset.zip(
            (train_img_dec_ds, train_label_dec_ds)).cache().shuffle(
                buffer_size=shuffle_buffer_size).batch(batch_size).repeat()
        val_ds = tf.data.Dataset.zip(
            (val_img_dec_ds, val_label_dec_ds)).cache().shuffle(
                buffer_size=shuffle_buffer_size).batch(batch_size).repeat()

    # training and validation dataset size
    n_train = tf.data.experimental.cardinality(train_img_dec_ds)
    n_val = tf.data.experimental.cardinality(val_img_dec_ds)

    # -------------------- get the class weights --------------------
    print('[INFO]Starting to define the class weights...')
    class_weights = get_class_weights(
        tf.data.Dataset.zip((train_img_dec_ds, train_label_dec_ds)))
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
                 class_weight=class_weights)

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
                             class_weight=class_weights)

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
        enet_hist = Enet.fit(x=train_ds,
                             epochs=epochs,
                             steps_per_epoch=n_train // batch_size,
                             validation_data=val_ds,
                             validation_steps=n_val // batch_size // 5,
                             class_weight=class_weights)

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
                             class_weight=class_weights)
