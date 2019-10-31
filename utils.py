import tensorflow as tf
import numpy as np


def process_path(file_path):
    '''
    Function to process the path containing the images and the
    labels for the input pipeline

    Arguments
    ----------
    'file_path' = path containing the images and
                  label folders

    Returns
    -------
    'img,iml' = image and label tensors
    '''

    # img file
    img_file = file_path

    # label file
    label_file = tf.strings.regex_replace(img_file, "/images", "/labels")
    print(img_file, label_file)

    # decoding image
    img = tf.io.read_file(img_file)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # decoding label
    iml = tf.io.read_file(label_file)
    iml = tf.image.decode_png(iml, channels=1)

    return img, iml


def tf_dataset_generator(dataset_path,
                         batch_size=32,
                         cache=True,
                         train=True,
                         shuffle_buffer_size=1000):
    '''
    Creates a training tf.dataset from images in the dataset_path

    Arguments
    ----------
    'dataset_path' = path containing the dataset images

    Returns
    -------
    'data_set' = training tf.dataset to plug in in model.fit()
    '''

    # create a list of the training images
    data_filelist_ds = tf.data.Dataset.list_files(dataset_path + '/*')

    # create the labeled dataset (returns (img,label) pairs)
    data_set = data_filelist_ds.map(process_path)

    # For a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that
    # don't fit in memory.
    if cache:
        if isinstance(cache, str):
            data_set = data_set.cache(cache)
        else:
            data_set = data_set.cache()

    # if training i want to shuffle, repeat and define a batch
    if train:

        data_set = data_set.shuffle(buffer_size=shuffle_buffer_size)

        # define the batch size
        data_set = data_set.batch(batch_size)

        # Repeat forever
        data_set = data_set.repeat()

    # `prefetch` lets the dataset fetch batches in the background while the
    # model is training.
    data_set = data_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return data_set


def get_class_weights(data_set, num_classes=12, c=1.02):
    '''
    Gets segmentation class weights from the dataset

    Arguments
    ----------
    'tf.dataset' = tf.dataset as returned from tf_dataset_generator

    Returns
    -------
    'class_weights' = class weights for the segmentation classes
    '''

    # building a giant array to count how many pixels per label
    label_list = []
    for img, label in data_set.take(-1):
        label_list.append(label.numpy())
    label_array = np.array(label_list).flatten()

    # counting the pixels
    each_class = np.bincount(label_array, minlength=num_classes)

    # computing the weights as in the original paper
    prospensity_score = each_class / len(label_array)
    class_weights = 1 / (np.log(c + prospensity_score))

    return class_weights
