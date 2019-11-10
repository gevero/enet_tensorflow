import tensorflow as tf
import numpy as np


def process_path_enc(file_path):
    '''
    Function to process the path containing the images and the
    labels for the input pipeline. In this case we work for the 
    encoder output

    Arguments
    ----------
    'file_path' = path containing the images and
                  label folders

    Returns
    -------
    'img' = image tensors
    'iml_end, iml_dec' = label tensors for the encorer and 
                         decoder heads
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
    img = tf.image.resize(img, [360, 480])

    # decoding label
    print(label_file)
    iml = tf.io.read_file(label_file)
    iml = tf.image.decode_png(iml, channels=1)
    iml = tf.image.convert_image_dtype(iml, tf.uint8)
    iml_enc = tf.image.resize(iml, [45, 60], method='nearest')

    return img, iml_enc


def process_path_dec(file_path):
    '''
    Function to process the path containing the images and the
    labels for the input pipeline. In this case we work for the 
    decoder output

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
    img = tf.image.resize(img, [360, 480])

    # decoding label
    print(label_file)
    iml = tf.io.read_file(label_file)
    iml = tf.image.decode_png(iml, channels=1)
    iml = tf.image.convert_image_dtype(iml, tf.uint8)
    iml = tf.image.resize(iml, [360, 480], method='nearest')  # 45,60

    return img, iml


def process_path_encdec(file_path):
    '''
    Function to process the path containing the images and the
    labels for the input pipeline. In this case we work for a 
    double objective function, one from the encoder and one from
    the decoder

    Arguments
    ----------
    'file_path' = path containing the images and
                  label folders

    Returns
    -------
    'img' = image tensors
    'iml_end, iml_dec' = label tensors for the encorer and 
                         decoder heads
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
    img = tf.image.resize(img, [360, 480])

    # decoding label
    print(label_file)
    iml = tf.io.read_file(label_file)
    iml = tf.image.decode_png(iml, channels=1)
    iml = tf.image.convert_image_dtype(iml, tf.uint8)
    iml_enc = tf.image.resize(iml, [45, 60], method='nearest')
    iml_dec = tf.image.resize(iml, [360, 480], method='nearest')

    return img, (iml_enc, iml_dec)


def tf_dataset_generator(dataset_path,
                         map_fn,
                         batch_size=16,
                         cache=True,
                         train=True,
                         shuffle_buffer_size=1000):
    '''
    Creates a training tf.dataset from images in the dataset_path

    Arguments
    ----------
    'dataset_path' = path containing the dataset images
    'map_fn' = function to map for the image processing

    Returns
    -------
    'data_set' = training tf.dataset to plug in in model.fit()
    '''

    # create a list of the training images
    data_filelist_ds = tf.data.Dataset.list_files(dataset_path + '/*')

    # create the labeled dataset (returns (img,label) pairs)
    data_set = data_filelist_ds.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
