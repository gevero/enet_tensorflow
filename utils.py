import tensorflow as tf
import numpy as np


def process_img(img_file, h_img, w_img):
    '''
    Function to process the image file. It takes the image file
    name as input and return the appropriate tensor as output.

    Arguments
    ----------
    'img_file' = String: image filename
    'h_img' = Integer: output tensor height
    'w_img' = Integer: output tensor width

    Returns
    -------
    'img' = image tensor
    '''

    # decoding image
    img = tf.io.read_file(img_file)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [h_img, w_img])

    return img


def process_label(label_file, h_iml, w_iml):
    '''
    Function to process the label file. It takes the label file
    name as input and return the appropriate tensor as output.

    Arguments
    ----------
    'label_file' = String: label filename
    'h_iml' = Integer: output tensor height
    'w_iml' = Integer: output tensor width

    Returns
    -------
    'iml' = image tensor
    '''

    # decoding image
    iml = tf.io.read_file(label_file)
    iml = tf.image.decode_png(iml, channels=1)
    iml = tf.image.convert_image_dtype(iml, tf.uint8)
    iml = tf.image.resize(iml, [h_iml, w_iml], method='nearest')

    return iml


def tf_dataset_generator(file_pattern,
                         map_fn)
    '''
    Creates a training tf.dataset from images or labels matching
    the 'file_pattern' in the 'dataset_path'. Here we do not batch
    or cache the dataset, because this will be done by chaining
    methods in a subsequent passage

    Arguments
    ----------
    'file_pattern' = glob pattern to match dataset files
    'map_fn' = function to map the filename to a tf tensor

    Returns
    -------
    'data_set' = training tf.dataset to plug in in model.fit()
    '''

    # create a list of the training images
    data_filelist_ds = tf.data.Dataset.list_files(file_pattern)

    # create the labeled dataset (returns (img,label) pairs)
    data_set = data_filelist_ds.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
