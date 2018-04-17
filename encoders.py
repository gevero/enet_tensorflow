import tensorflow as tf
import numpy as np

def spatial_dropout(input_layer,rate=0.5,train=False,name=None):
    '''
    Simple dropout wrapper to perform spatial dropout. This is based on
    the implementation found in https://github.com/fregu856/segmentation

    Arguments
    ----------
    'input_layer' = A `Tensor` with type `float32` with shape
                    [batch_size, im_height, im_width, filters]
    'rate' = a `float`: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop
                        out 10% of input units
    'training' = a `booleand`: training or evaluation mode for dropout
    'name' = a `String`: gives the name_scope and variable_scope

    Returns
    -------
    'drop_layer' = A `Tensor` with the same type as `input_layer`.

    '''

    # name_scope
    with tf.name_scope(name):
        with tf.variable_scope(name):

            # noise_shape
            input_shape = input_layer.get_shape().as_list()
            batch_size = input_shape[0]
            filters = input_shape[3]
            if batch_size == None:
                noise_shape = tf.constant(value=[1, 1, 1, 1])
            else:
                noise_shape = tf.constant(value=[batch_size, 1, 1, filters])

            # spatial dropout
            drop_layer = tf.layers.dropout(input_layer,rate,
                                           noise_shape=noise_shape,
                                           training=train,
                                           name='dropout')

            return drop_layer

def prelu(x, name='prelu'):
    '''
    Parametric Rectified Linear Unit inspired by:
    https://stackoverflow.com/a/40264459

    Arguments
    ----------
    'x' = A `Tensor` with type `float32`
    'name' = a `String`: gives the name_scope and variable_scope

    Returns
    -------
    'x' = A `Tensor` with the same type as `x`.
    '''

    # name_scope
    with tf.name_scope(name):
        with tf.variable_scope(name):

            # initializing variables
            alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)

            # getting output
            x = tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5

    return x


def bottleneck(input_layer,
               train,
               output_filters=128,
               kernel_size=[3,3], kernel_strides=[1,1],
               padding='same',dilation_rate=[1,1],
               internal_comp_ratio=4,dropout_prob=0.1,
               downsample = False,
               name='bottleneck'):

    '''
    Enet bottleneck module as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural
        Network Architecture for Real-Time Semantic Segmentation. arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html


    Arguments
    ----------
    'input_layer' = input `Tensor` with type `float32`
    'train' = a `boolean`: training or evaluation mode for dropout
    'output_filters' = an `Integer`: number of output filters (or channels, as you like it)
    'kernel_size' = a `List`: size of the kernel for the central convolution
    'kernel_strides' = a `List`: length of the strides for the central convolution
    'padding' = a `String`: padding of the central convolution
    'dilation_rate' = a `List`: dilation rate of the central convolution
    'internal_comp_ratio' = an `Integer`: compression ratio of the bottleneck
    'dropout_prob' = a `float`: dropout at the end of the main connection
    'downsample' = a `String`: downsampling flag
    'name' = a `String`: name of the bottleneck

    Returns
    -------
    'output_layer' = A `Tensor` with the same type as `input_layer`.
    '''

    # namescoping for tensorboard???
    with tf.name_scope(name):
        with tf.variable_scope(name):

            # number of internal filters
            input_filters = input_layer.get_shape().as_list()[-1]
            internal_filters = input_filters // internal_comp_ratio

            # downsampling or not
            if downsample:
                down_kernel = [2,2]
                down_strides = [2,2]
            else:
                down_kernel = [1,1]
                down_strides = [1,1]

            # -------main connection----------

            # bottleneck representation compression with valid padding
            # 1x1 usually, 2x2 if downsampling
            main1_1 = tf.layers.conv2d(inputs=input_layer,
                                        filters=internal_filters,
                                        kernel_size=down_kernel,
                                        strides=down_strides,
                                        use_bias=False,
                                        name='main1_1')
            main1_2 = tf.layers.batch_normalization(main1_1,
                                                    training=train,name='main1_2')  # batch norm
            main1_3 = prelu(main1_2,name='main1_3')     # PReLU activation


            # central convolution
            asym_flag = kernel_size[0] != kernel_size[1]
            main1_4 = tf.layers.conv2d(inputs=main1_3,          # main convolution
                                    filters=internal_filters,
                                    kernel_size=kernel_size,
                                    strides=kernel_strides,
                                    padding=padding,
                                    dilation_rate=dilation_rate,
                                    use_bias=not(asym_flag),
                                    name='main1_4a')      # no bias only for asymmetric conv
            if asym_flag:                                         # second convolution if asymmetric
                main1_4 = tf.layers.conv2d(inputs=main1_4,
                                    filters=internal_filters,
                                    kernel_size=kernel_size[::-1],
                                    strides=kernel_strides,
                                    padding=padding,
                                    dilation_rate=dilation_rate,
                                    name='main1_4b')
            main1_5 = tf.layers.batch_normalization(main1_4,
                                                    training=train,name='main1_5')    # batchnorm
            main1_6 = prelu(main1_5,name='main1_6')  # PReLU


            # bottleneck representation expansion with 1x1 valid convolution
            main1_7 = tf.layers.conv2d(inputs=main1_6,
                                        filters=output_filters,
                                        kernel_size=[1,1],
                                        strides=[1,1],
                                        use_bias=False,
                                        name='main1_7')
            main1_8 = tf.layers.batch_normalization(main1_7,
                                                    training=train,name='main1_8')  # batchnorm
            main1_9 = spatial_dropout(main1_8,rate=dropout_prob,
                                       train=train,name='main1_9')  # dropout

            # -------skip connection-------
            skip1_1 = input_layer
            # downsampling
            if downsample:
                skip1_1 = tf.layers.max_pooling2d(inputs=skip1_1,
                                                   pool_size=down_kernel,
                                                   strides=down_strides,
                                                   name='skip1_1a')
            # padding filter dimension if input_filters != output_filters
            if input_filters != output_filters:
                n_pad = output_filters - input_filters
                skip1_1 = tf.pad(skip1_1,tf.constant([[0,0],[0,0],[0,0],[n_pad,0]]),
                                 name='skip1_1b')


            # -------output-------
            output_layer = prelu(tf.add(main1_9,skip1_1,name='addition'),
                                 name='output_layer')
            return output_layer


def init_block(input_layer,
               train,
               conv_filters=13,
               kernel_size=[3,3], kernel_strides=[2,2],
               pool_size=[2,2], pool_strides=[2,2],
               padding='valid',
               name='init_block'):

    '''
    Enet init_block as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural Network
        Architecture for Real-Time Semantic Segmentation. arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html


    Arguments
    ----------
    'input_layer' = input `Tensor` with type `float32`
    'train' = a `boolean`: training or evaluation mode for dropout
    'conv_filters' = an `Integer`: number filters for the convolution (or channels, if you like it)
    'kernel_size' = a `List`: size of the kernel for the convolution
    'kernel_strides' = a `List`: length of the strides for the convolution
    'pool_size' = a `List`: size of the pool for the maxpooling
    'pool_strides' = a `List`: length of the strides for the maxpooling
    'padding' = a `String`: padding for the convolution and the maxpooling
    'name' = a `String`: name of the init_block

    Returns
    -------
    'out_init' = A `Tensor` with the same type as `input_layer`.
    '''

    # namescoping for tensorboard???
    with tf.name_scope(name):
        with tf.variable_scope(name):

            # conv connection: need the padding to match the dimension of pool_init
            padded_init = tf.pad(input_layer,tf.constant([[0,0],[1,1],[1,1],[0,0]]))
            conv_init = tf.layers.conv2d(inputs=padded_init,
                                         filters=conv_filters,
                                         kernel_size=kernel_size,
                                         strides=kernel_strides,
                                         padding='valid')

            # maxpooling connection, where pool_init is to be concatenated with conv_init
            pool_init = tf.layers.max_pooling2d(inputs=input_layer,
                                                pool_size=pool_size,
                                                strides=pool_strides,
                                                padding='valid')

            # concatenating the two connections
            out_init = tf.concat([conv_init,pool_init],-1)
            out_init = tf.layers.batch_normalization(out_init,training=train)
            out_init = prelu(out_init,name='out_init')

            return out_init


def enet_encoder(input_layer,train,n_classes=10):

    '''
    Enet encoder as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural Network
        Architecture for Real-Time Semantic Segmentation. arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html


    Arguments
    ----------
    'input_layer' = input `Tensor` with type `float32` and shape
                    [batch_size,x_pixels,y_pixels,n_channels]
    'train' = a `boolean`: training or evaluation mode for dropout
    'n_classes' = an `Integer`: number of classes

    Returns
    -------
    'output_layer' = A `Tensor` with the same type as `input_layer` and shape
                    [batch_size,x_pixels,y_pixels,n_classes] for the segmentation.
    '''

    # ---------Initial block---------
    out_init =  init_block(input_layer)

    # --------first block---------
    # first bottleneck with downsampling
    bt1_0  = bottleneck(out_init,train, output_filters=64,dropout_prob=0.01,downsample=True,name='bt1_0_ds')

    # four bottlenecks without downsampling
    bt1_1  = bottleneck(bt1_0,train,output_filters=64,dropout_prob=0.01,name='bt1_1')
    bt1_2  = bottleneck(bt1_1,train,output_filters=64,dropout_prob=0.01,name='bt1_2')
    bt1_3  = bottleneck(bt1_2,train,output_filters=64,dropout_prob=0.01,name='bt1_3')
    bt1_4  = bottleneck(bt1_3,train,output_filters=64,dropout_prob=0.01,name='bt1_4')

    # --------second block---------
    bt2_0  = bottleneck(bt1_4,train,output_filters=128,downsample=True,name='bt2_0_ds')
    bt2_1  = bottleneck(bt2_0,train,output_filters=128,name='bt2_1')
    bt2_2  = bottleneck(bt2_1,train,output_filters=128,dilation_rate=[2,2],name='bt2_2_dl')
    bt2_3  = bottleneck(bt2_2,train,output_filters=128,kernel_size=[7,1],name='bt2_3_as')
    bt2_4  = bottleneck(bt2_3,train,output_filters=128,dilation_rate=[4,4],name='bt2_4_dl')
    bt2_5  = bottleneck(bt2_4,train,output_filters=128,name='bt2_5')
    bt2_6  = bottleneck(bt2_5,train,output_filters=128,dilation_rate=[8,8],name='bt2_6_d')
    bt2_7  = bottleneck(bt2_6,train,output_filters=128,kernel_size=[5,1],name='bt2_7_as')
    bt2_8  = bottleneck(bt2_7,train,output_filters=128,dilation_rate=[16,16],name='bt2_8_dl')

    # --------third block---------
    bt3_0  = bottleneck(bt2_8,train,output_filters=128,name='bt3_0')
    bt3_1  = bottleneck(bt3_0,train,output_filters=128,dilation_rate=[2,2],name='bt3_1_dl')
    bt3_2  = bottleneck(bt3_1,train,output_filters=128,kernel_size=[7,1],name='bt3_2_as')
    bt3_3  = bottleneck(bt3_2,train,output_filters=128,dilation_rate=[4,4],name='bt3_3_dl')
    bt3_4  = bottleneck(bt3_3,train,output_filters=128,name='bt3_4')
    bt3_5  = bottleneck(bt3_4,train,output_filters=128,dilation_rate=[8,8],name='bt3_5_d')
    bt3_6  = bottleneck(bt3_5,train,output_filters=128,kernel_size=[5,1],name='bt3_6_as')
    bt3_7  = bottleneck(bt3_6,train,output_filters=128,dilation_rate=[16,16],name='bt3_7_dl')

    # --------compression to get to a downsampled representation of segmentation---------
    output_layer = tf.layers.conv2d(inputs=bt3_7,
                                    filters=n_classes,
                                    kernel_size=[1,1],
                                    strides=[1,1],
                                    padding='valid')

    return output_layer


def enet_encoder_v3(input_layer,train,n_classes=10):

    '''
    Enet encoder v3 for imagenet as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation. arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html


    Arguments
    ----------
    'input_layer' = input `Tensor` with type `float32` and
                    shape [batch_size,x_pixels,y_pixels,n_channels]
    'train' = a `boolean`: training or evaluation mode for dropout
    'n_classes' = an `Integer`: number of classes

    Returns
    -------
    'logits' = A `Tensor` with the same type as `input_layer` and shape [batch_size,n_classes]
    '''

    # ---------Initial block---------
    out_init =  init_block(input_layer)

    # --------first block---------
    # first bottleneck with downsampling
    bt1_0  = bottleneck(out_init,train,output_filters=64,dropout_prob=0.01,downsample=True,name='bt1_0_ds')

    # four bottlenecks without downsampling
    bt1_1  = bottleneck(bt1_0,train,output_filters=64,dropout_prob=0.01,name='bt1_1')
    bt1_2  = bottleneck(bt1_1,train,output_filters=64,dropout_prob=0.01,name='bt1_2')
    bt1_3  = bottleneck(bt1_2,train,output_filters=64,dropout_prob=0.01,name='bt1_3')
    bt1_4  = bottleneck(bt1_3,train,output_filters=64,dropout_prob=0.01,name='bt1_4')

    # --------second block---------
    bt2_0  = bottleneck(bt1_4,train,output_filters=128,downsample=True,name='bt2_0_ds')
    bt2_1  = bottleneck(bt2_0,train,output_filters=128,name='bt2_1')
    bt2_2  = bottleneck(bt2_1,train,output_filters=128,dilation_rate=[2,2],name='bt2_2_dl')
    bt2_3  = bottleneck(bt2_2,train,output_filters=128,kernel_size=[7,1],name='bt2_3_as')
    bt2_4  = bottleneck(bt2_3,train,output_filters=128,dilation_rate=[4,4],name='bt2_4_dl')
    bt2_5  = bottleneck(bt2_4,train,output_filters=128,name='bt2_5')
    bt2_6  = bottleneck(bt2_5,train,output_filters=128,dilation_rate=[8,8],name='bt2_6_d')
    bt2_7  = bottleneck(bt2_6,train,output_filters=128,kernel_size=[5,1],name='bt2_7_as')
    bt2_8  = bottleneck(bt2_7,train,output_filters=256,dilation_rate=[16,16],name='bt2_8_dl')
    bt2_9  = bottleneck(bt2_8,train,output_filters=256,downsample=True,name='bt2_9_ds')

    # --------third block---------
    bt3_0  = bottleneck(bt2_9,train,output_filters=256,name='bt3_0')
    bt3_1  = bottleneck(bt3_0,train,output_filters=256,dilation_rate=[2,2],name='bt3_1_dl')
    bt3_2  = bottleneck(bt3_1,train,output_filters=256,kernel_size=[7,1],name='bt3_2_as')
    bt3_3  = bottleneck(bt3_2,train,output_filters=256,dilation_rate=[4,4],name='bt3_3_dl')
    bt3_4  = bottleneck(bt3_3,train,output_filters=256,name='bt3_4')
    bt3_5  = bottleneck(bt3_4,train,output_filters=256,dilation_rate=[8,8],name='bt3_5_d')
    bt3_6  = bottleneck(bt3_5,train,output_filters=256,kernel_size=[5,1],name='bt3_6_as')
    bt3_7  = bottleneck(bt3_6,train,output_filters=512,dilation_rate=[16,16],name='bt3_7_dl')
    bt3_8  = bottleneck(bt3_8,train,output_filters=512,downsample=True,name='bt3_8_ds')

    # --------logits---------
    r_mean= tf.reduce_mean(bt3_8,axis=[1,2],keepdims=True)
    r_mean = tf.identity(r_mean, 'final_reduce_mean')
    r_mean_reshape = tf.reshape(r_mean, [-1, x.get_shape().as_list()[-1]])
    logits = tf.layers.dense(inputs=r_mean_reshape, units=n_classes)
    logits = tf.identity(logits, 'logits')

    return logits


def enet_encoder_mnist(input_layer,train,n_classes=10):

    '''
    Slimmed down Enet encoder for mnist. Actually, does not look like enet at all.
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural Network
        Architecture for Real-Time Semantic Segmentation. arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html


    Arguments
    ----------
    'input_layer' = input `Tensor` with type `float32` and
                    shape [batch_size,28,28,1]
    'train' = a `boolean`: training or evaluation mode for dropout
    'n_classes' = an `Integer`: number of classes

    Returns
    -------
    'logits' = A `Tensor` with the same type as `input_layer` and shape [batch_size,n_classes]
    '''

    # ---------Initial block---------
    out_init =  init_block(input_layer,train,conv_filters=15) # 14x14

    # --------first block---------
    # first bottleneck with downsampling
    bt1_0  = bottleneck(out_init,train,output_filters=32,dropout_prob=0.5,
                          downsample=True,name='bt1_0_ds')  # 7x7

    # four bottlenecks without downsampling
    # bt1_1  = bottleneck(bt1_0,train,output_filters=64,dropout_prob=0.5,name='bt1_1')
    # bt1_2  = bottleneck(bt1_1,train,output_filters=64,dropout_prob=0.5,name='bt1_2')
    # bt1_3  = bottleneck(bt1_2,train,output_filters=64,dropout_prob=0.5,name='bt1_3')
    # bt1_4  = bottleneck(bt1_3,train,output_filters=64,dropout_prob=0.5,name='bt1_4')

    # --------logits---------
    # bt_shape = bt1_0.get_shape().as_list()
    # bt1_0_flat = tf.reshape(bt1_0, [-1, bt_shape[1] * bt_shape[2] * bt_shape[3]])
    # dense = tf.layers.dense(inputs=bt1_0_flat, units=1024, activation=tf.nn.relu)
    r_mean= tf.reduce_mean(bt1_0,axis=[1,2],keepdims=True,name='r_mean')
    r_mean_reshape = tf.reshape(r_mean, [-1, r_mean.get_shape().as_list()[-1]],
                                name='r_mean_reshape')
    r_mean_reshape = tf.identity(r_mean_reshape,name='final_reduce_mean')
    logits = tf.layers.dense(inputs=r_mean_reshape, units=n_classes,name='dense_logits')
    logits = tf.identity(logits,name='logits')

    return logits


def mnist_test(features,mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # additional bottlenecks
  train  = (mode == tf.estimator.ModeKeys.TRAIN)
  bt1_1  = bottleneck(conv1,train,output_filters=32,dropout_prob=0.5,name='bt1_1')

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=bt1_1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)
  return logits
