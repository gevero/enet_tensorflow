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


def map_label(label_file, h_iml, w_iml):
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


def tf_dataset_generator(file_pattern, map_fn):
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
    data_filelist_ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)

    # create the labeled dataset (returns (img,label) pairs)
    data_set = data_filelist_ds.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return data_set


def get_class_weights(data_set, num_classes=12, c=1.02):
    '''
    Gets segmentation class weights from the dataset. This
    is thought for a large dataset where the calculation
    of the weights must be done out of memory. If data_set
    is already divided in batches, the calculation should be
    rather efficient and the memory should not blow up.

    Arguments
    ----------
    'data_set' = already batched tf.dataset containing the
                 correctly resized label tensors 

    Returns
    -------
    'class_weights' = class weights for the segmentation classes
    '''

    # building a giant array to count how many pixels per label
    each_class = 0.0
    tot_num_pixel = 0.0
    for label in data_set.take(-1):

        # flatten the barch of label arrays
        label_array = np.array(label.numpy()).flatten()

        # counting the pixels
        each_class = each_class + np.bincount(label_array,
                                              minlength=num_classes)
        tot_num_pixel = tot_num_pixel + len(label_array)

    # computing the weights as in the original paper
    prospensity_score = each_class / tot_num_pixel
    class_weights = 1 / (np.log(c + prospensity_score))

    return class_weights


def preprocess_img_label(img_pattern, label_pattern):
    '''
    Creates the string tensor pairs (img_file, label_file)

    Arguments
    ----------
    'img_pattern' = glob pattern to match img files
    'label_pattern' = glob pattern to match label files

    Returns
    -------
    'pair_filelist' = returns the list of pair files
    '''

    # create a list of the training images
    img_filelist = tf.data.Dataset.list_files(img_pattern, shuffle=False)

    # create a list of the label images
    label_filelist = tf.data.Dataset.list_files(label_pattern, shuffle=False)

    # pair filenames
    pair_filelist = tf.data.Dataset.zip((img_filelist, label_filelist))

    return pair_filelist


def map_singlehead(img_file, label_file, h_img, w_img):
    '''
    Takes the string tensor pair (img_file, label_file)
    and outputs the (img,label) tf tensor pair

    Arguments
    ----------
    'img_file' = string tensor with img filename
    'label_file' = string tensor with label filename
    'h_iml' = Integer: output tensor height
    'w_iml' = Integer: output tensor width

    Returns
    -------
    '(img,iml)' = image and label tensor
    '''

    # decoding image
    img = tf.io.read_file(img_file)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [h_img, w_img])

    # decoding label
    iml = tf.io.read_file(label_file)
    iml = tf.image.decode_png(iml, channels=1)
    iml = tf.image.convert_image_dtype(iml, tf.uint8)
    iml = tf.image.resize(iml, [h_img, w_img], method='nearest')

    return (img, iml)


def map_doublehead(img_file, label_file, h_enc, w_enc, h_dec, w_dec):
    '''
    Takes the string tensor pair (img_file, label_file)
    and outputs the (img,label) tf tensor pair

    Arguments
    ----------
    'img_file' = string tensor with img filename
    'label_file' = string tensor with label filename
    'h_enc' = Integer: output tensor height at encoder head
    'w_enc' = Integer: output tensor width at encoder head
    'h_dec' = Integer: output tensor height at decoder head
    'w_dec' = Integer: output tensor width at decoder head

    Returns
    -------
    '(img,(iml_enc,iml_dec))' = image and label tensors
    '''

    # decoding image
    img = tf.io.read_file(img_file)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [h_dec, w_dec])

    # decoding label
    iml = tf.io.read_file(label_file)
    iml = tf.image.decode_png(iml, channels=1)
    iml = tf.image.convert_image_dtype(iml, tf.uint8)
    iml_enc = tf.image.resize(iml, [h_enc, w_enc], method='nearest')
    iml_dec = tf.image.resize(iml, [h_dec, w_dec], method='nearest')
    return (img, (iml_enc, iml_dec))
