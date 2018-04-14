import tensorflow as tf
import numpy as np

def spatial_dropout(input_layer,rate=0.5,name=None):
    '''
    Simple dropout wrapper to perform spatial dropout. This is based on
    the implementation found in https://github.com/fregu856/segmentation

    Arguments
    ----------
    'input_layer' = A `Tensor` with type `float32` with shape
                    [batch_size, im_height, im_width, filters]
    'rate' = a `float`: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop
                        out 10% of input units
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
            noise_shape = tf.constant(value=[batch_size, 1, 1, filters])

            # spatial dropout
            drop_layer = tf.layers.dropout(input_layer,rate,
                                           noise_shape=noise_shape,
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
            main_1_1 = tf.layers.conv2d(inputs=input_layer,
                                        filters=internal_filters,
                                        kernel_size=down_kernel,
                                        strides=down_strides,
                                        use_bias=False)
            main_1_2 = tf.layers.batch_normalization(main_1_1)  # batch norm
            main_1_3 = prelu(main_1_2,name='main_1_3')     # PReLU activation


            # central convolution
            asym_flag = kernel_size[0] != kernel_size[1]
            main_1_4 = tf.layers.conv2d(inputs=main_1_3,          # main convolution
                                    filters=internal_filters,
                                    kernel_size=kernel_size,
                                    strides=kernel_strides,
                                    padding=padding,
                                    dilation_rate=dilation_rate,
                                    use_bias=not(asym_flag))      # no bias only for asymmetric conv
            if asym_flag:                                         # second convolution if asymmetric
                main_1_4 = tf.layers.conv2d(inputs=main_1_4,
                                    filters=internal_filters,
                                    kernel_size=kernel_size[::-1],
                                    strides=kernel_strides,
                                    padding=padding,
                                    dilation_rate=dilation_rate)
            main_1_5 = tf.layers.batch_normalization(main_1_4)    # batchnorm
            main_1_6 = prelu(main_1_5,name='main_1_6')  # PReLU


            # bottleneck representation expansion with 1x1 valid convolution
            main_1_7 = tf.layers.conv2d(inputs=main_1_6,
                                        filters=output_filters,
                                        kernel_size=[1,1],
                                        strides=[1,1],
                                        use_bias=False)
            main_1_8 = tf.layers.batch_normalization(main_1_7)  # batchnorm
            main_1_9 = spatial_dropout(main_1_8,rate=dropout_prob,name='dropout')  # dropout

            # -------skip connection-------
            skip_1_1 = input_layer
            # downsampling
            if downsample:
                skip_1_1 = tf.layers.max_pooling2d(inputs=skip_1_1,
                                                   pool_size=down_kernel,
                                                   strides=down_strides)
            # padding filter dimension if input_filters != output_filters
            if input_filters != output_filters:
                n_pad = output_filters - input_filters
                skip_1_1 = tf.pad(skip_1_1,tf.constant([[0,0],[0,0],[0,0],[n_pad,0]]))


            # -------output-------
            output_layer = prelu(main_1_9 + skip_1_1,name='output_layer')
            return output_layer


def init_block(input_layer,
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
            out_init = tf.layers.batch_normalization(out_init)
            out_init = prelu(out_init,name='out_init')

            return out_init


def enet_encoder(input_layer,n_classes=10):

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
    conv1_0  = bottleneck(out_init,output_filters=64,dropout_prob=0.01,downsample=True,name='conv1_0_ds')

    # four bottlenecks without downsampling
    conv1_1  = bottleneck(conv1_0,output_filters=64,dropout_prob=0.01,name='conv1_1')
    conv1_2  = bottleneck(conv1_1,output_filters=64,dropout_prob=0.01,name='conv1_2')
    conv1_3  = bottleneck(conv1_2,output_filters=64,dropout_prob=0.01,name='conv1_3')
    conv1_4  = bottleneck(conv1_3,output_filters=64,dropout_prob=0.01,name='conv1_4')

    # --------second block---------
    conv2_0  = bottleneck(conv1_4,output_filters=128,downsample=True,name='conv2_0_ds')
    conv2_1  = bottleneck(conv2_0,output_filters=128,name='conv2_1')
    conv2_2  = bottleneck(conv2_1,output_filters=128,dilation_rate=[2,2],name='conv2_2_dl')
    conv2_3  = bottleneck(conv2_2,output_filters=128,kernel_size=[7,1],name='conv2_3_as')
    conv2_4  = bottleneck(conv2_3,output_filters=128,dilation_rate=[4,4],name='conv2_4_dl')
    conv2_5  = bottleneck(conv2_4,output_filters=128,name='conv2_5')
    conv2_6  = bottleneck(conv2_5,output_filters=128,dilation_rate=[8,8],name='conv2_6_d')
    conv2_7  = bottleneck(conv2_6,output_filters=128,kernel_size=[5,1],name='conv2_7_as')
    conv2_8  = bottleneck(conv2_7,output_filters=128,dilation_rate=[16,16],name='conv2_8_dl')

    # --------third block---------
    conv3_0  = bottleneck(conv2_8,output_filters=128,name='conv3_0')
    conv3_1  = bottleneck(conv3_0,output_filters=128,dilation_rate=[2,2],name='conv3_1_dl')
    conv3_2  = bottleneck(conv3_1,output_filters=128,kernel_size=[7,1],name='conv3_2_as')
    conv3_3  = bottleneck(conv3_2,output_filters=128,dilation_rate=[4,4],name='conv3_3_dl')
    conv3_4  = bottleneck(conv3_3,output_filters=128,name='conv3_4')
    conv3_5  = bottleneck(conv3_4,output_filters=128,dilation_rate=[8,8],name='conv3_5_d')
    conv3_6  = bottleneck(conv3_5,output_filters=128,kernel_size=[5,1],name='conv3_6_as')
    conv3_7  = bottleneck(conv3_6,output_filters=128,dilation_rate=[16,16],name='conv3_7_dl')

    # --------compression to get to a downsampled representation of segmentation---------
    output_layer = tf.layers.conv2d(inputs=conv3_7,
                                    filters=n_classes,
                                    kernel_size=[1,1],
                                    strides=[1,1],
                                    padding='valid')

    return output_layer


def enet_encoder_mnist(input_layer,n_classes=10):

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
    'n_classes' = an `Integer`: number of classes

    Returns
    -------
    'logits' = A `Tensor` with the same type as `input_layer` and shape [batch_size,n_classes]
    '''

    # ---------Initial block---------
    out_init =  init_block(input_layer) # 14x14

    # --------first block---------
    # first bottleneck with downsampling
    conv1_0  = bottleneck(out_init,output_filters=64,dropout_prob=0.5,
                          downsample=True,name='conv1_0_ds')  # 7x7

    # four bottlenecks without downsampling
    conv1_1  = bottleneck(conv1_0,output_filters=64,dropout_prob=0.5,name='conv1_1')
    conv1_2  = bottleneck(conv1_1,output_filters=64,dropout_prob=0.5,name='conv1_2')
    conv1_3  = bottleneck(conv1_2,output_filters=64,dropout_prob=0.5,name='conv1_3')
    conv1_4  = bottleneck(conv1_3,output_filters=64,dropout_prob=0.5,name='conv1_4')

    # --------logits---------
    r_mean= tf.reduce_mean(conv1_4,axis=[1,2],keepdims=True)
    r_mean_reshape = tf.reshape(r_mean, [-1, r_mean.get_shape().as_list()[-1]])
    r_mean = tf.identity(r_mean, 'final_reduce_mean')
    logits = tf.layers.dense(inputs=r_mean_reshape, units=n_classes)
    logits = tf.identity(logits, 'logits')

    return logits


def enet_encoder_v3(input_layer,n_classes=10):

    '''
    Enet encoder v3 for imagenet as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation. arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html


    Arguments
    ----------
    'input_layer' = input `Tensor` with type `float32` and
                    shape [batch_size,x_pixels,y_pixels,n_channels]
    'n_classes' = an `Integer`: number of classes

    Returns
    -------
    'logits' = A `Tensor` with the same type as `input_layer` and shape [batch_size,n_classes]
    '''

    # ---------Initial block---------
    out_init =  init_block(input_layer)

    # --------first block---------
    # first bottleneck with downsampling
    conv1_0  = bottleneck(out_init,output_filters=64,dropout_prob=0.01,downsample=True,name='conv1_0_ds')

    # four bottlenecks without downsampling
    conv1_1  = bottleneck(conv1_0,output_filters=64,dropout_prob=0.01,name='conv1_1')
    conv1_2  = bottleneck(conv1_1,output_filters=64,dropout_prob=0.01,name='conv1_2')
    conv1_3  = bottleneck(conv1_2,output_filters=64,dropout_prob=0.01,name='conv1_3')
    conv1_4  = bottleneck(conv1_3,output_filters=64,dropout_prob=0.01,name='conv1_4')

    # --------second block---------
    conv2_0  = bottleneck(conv1_4,output_filters=128,downsample=True,name='conv2_0_ds')
    conv2_1  = bottleneck(conv2_0,output_filters=128,name='conv2_1')
    conv2_2  = bottleneck(conv2_1,output_filters=128,dilation_rate=[2,2],name='conv2_2_dl')
    conv2_3  = bottleneck(conv2_2,output_filters=128,kernel_size=[7,1],name='conv2_3_as')
    conv2_4  = bottleneck(conv2_3,output_filters=128,dilation_rate=[4,4],name='conv2_4_dl')
    conv2_5  = bottleneck(conv2_4,output_filters=128,name='conv2_5')
    conv2_6  = bottleneck(conv2_5,output_filters=128,dilation_rate=[8,8],name='conv2_6_d')
    conv2_7  = bottleneck(conv2_6,output_filters=128,kernel_size=[5,1],name='conv2_7_as')
    conv2_8  = bottleneck(conv2_7,output_filters=256,dilation_rate=[16,16],name='conv2_8_dl')
    conv2_9  = bottleneck(conv2_8,output_filters=256,downsample=True,name='conv2_9_ds')

    # --------third block---------
    conv3_0  = bottleneck(conv2_9,output_filters=256,name='conv3_0')
    conv3_1  = bottleneck(conv3_0,output_filters=256,dilation_rate=[2,2],name='conv3_1_dl')
    conv3_2  = bottleneck(conv3_1,output_filters=256,kernel_size=[7,1],name='conv3_2_as')
    conv3_3  = bottleneck(conv3_2,output_filters=256,dilation_rate=[4,4],name='conv3_3_dl')
    conv3_4  = bottleneck(conv3_3,output_filters=256,name='conv3_4')
    conv3_5  = bottleneck(conv3_4,output_filters=256,dilation_rate=[8,8],name='conv3_5_d')
    conv3_6  = bottleneck(conv3_5,output_filters=256,kernel_size=[5,1],name='conv3_6_as')
    conv3_7  = bottleneck(conv3_6,output_filters=512,dilation_rate=[16,16],name='conv3_7_dl')
    conv3_8  = bottleneck(conv3_8,output_filters=512,downsample=True,name='conv3_8_ds')

    # --------logits---------
    r_mean= tf.reduce_mean(conv3_8,axis=[1,2],keepdims=True)
    r_mean = tf.identity(r_mean, 'final_reduce_mean')
    r_mean_reshape = tf.reshape(r_mean, [-1, x.get_shape().as_list()[-1]])
    logits = tf.layers.dense(inputs=r_mean_reshape, units=n_classes)
    logits = tf.identity(logits, 'logits')

    return logits
