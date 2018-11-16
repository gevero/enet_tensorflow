import tensorflow as tf


def bottleneck(input_layer,
               output_filters=128,
               kernel_size=[3, 3],
               kernel_strides=[1, 1],
               padding='same',
               dilation_rate=[1, 1],
               internal_comp_ratio=4,
               dropout_prob=0.1,
               downsample=False,
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
    with tf.variable_scope(name):

        # number of internal filters
        input_filters = input_layer.get_shape().as_list()[-1]
        internal_filters = input_filters // internal_comp_ratio

        # downsampling or not
        if downsample:
            down_kernel = [2, 2]
            down_strides = [2, 2]
        else:
            down_kernel = [1, 1]
            down_strides = [1, 1]

        # -------main connection----------

        # bottleneck representation compression with valid padding
        # 1x1 usually, 2x2 if downsampling
        main1_1 = tf.keras.layers.Conv2D(internal_filters,
                                         down_kernel,
                                         strides=down_strides,
                                         use_bias=False,
                                         name=name + '.' + 'main1_1')(input_layer)
        main1_2 = tf.keras.layers.BatchNormalization(name=name + '.' + 'main1_2')(main1_1)
        main1_3 = tf.keras.layers.PReLU(name=name + '.' + 'main1_3')(main1_2)

        # central convolution
        asym_flag = kernel_size[0] != kernel_size[1]
        main1_4 = tf.keras.layers.Conv2D(internal_filters,
                                         kernel_size,
                                         strides=kernel_strides,
                                         padding=padding,
                                         dilation_rate=dilation_rate,
                                         use_bias=not (asym_flag),
                                         name=name + '.' + 'main1_4a')(main1_3)
        if asym_flag:
            main1_4 = tf.keras.layers.Conv2D(internal_filters,
                                             kernel_size[::-1],
                                             strides=kernel_strides,
                                             padding=padding,
                                             dilation_rate=dilation_rate,
                                             name=name + '.' + 'main1_4b')(main1_4)
        main1_5 = tf.keras.layers.BatchNormalization(name=name + '.' + 'main1_5')(main1_4)
        main1_6 = tf.keras.layers.PReLU(name=name + '.' + 'main1_6')(main1_5)

        # bottleneck representation expansion with 1x1 valid convolution
        main1_7 = tf.keras.layers.Conv2D(output_filters, [1, 1],
                                         strides=[1, 1],
                                         use_bias=False,
                                         name=name + '.' + 'main1_7')(main1_6)
        main1_8 = tf.keras.layers.BatchNormalization(name=name + '.' + 'main1_8')(main1_7)
        main1_9 = tf.keras.layers.SpatialDropout2D(dropout_prob, name=name + '.' + 'main1_9')(main1_8)

        # -------skip connection-------
        skip1_1 = input_layer

        # downsampling
        if downsample:
            skip1_1 = tf.keras.layers.MaxPool2D(pool_size=down_kernel,
                                                strides=down_strides,
                                                name=name + '.' + 'skip1_1a')(skip1_1)

        # matching filter dimension with learned 1x1 convolution
        if input_filters != output_filters:
            skip1_1 = tf.keras.layers.Conv2D(output_filters,
                                             kernel_size=[1, 1],
                                             padding='valid',
                                             use_bias=False)(skip1_1)

        # -------output-------
        output_layer = tf.keras.layers.PReLU(name=name + '.' + 'output_layer')(
                       tf.keras.layers.Add(name=name + '.' + 'addition')([main1_9, skip1_1]))

        return output_layer


def init_block(input_layer,
               conv_filters=13,
               kernel_size=[3, 3],
               kernel_strides=[2, 2],
               pool_size=[2, 2],
               pool_strides=[2, 2],
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
    with tf.variable_scope(name):

        # conv connection: need the padding to match the dimension of pool_init
        padded_init = tf.keras.layers.ZeroPadding2D()(input_layer)
        conv_init = tf.keras.layers.Conv2D(conv_filters, kernel_size,
                                           strides=kernel_strides,
                                           padding='valid')(padded_init)

        # maxpooling connection, where pool_init is to be concatenated with conv_init
        pool_init = tf.keras.layers.MaxPool2D(pool_size=pool_size,
                                              strides=pool_strides,
                                              padding='valid')(input_layer)

        # concatenating the two connections
        out_init = tf.keras.layers.Concatenate(axis=-1)([conv_init, pool_init])
        out_init = tf.keras.layers.BatchNormalization()(out_init)
        out_init = tf.keras.layers.PReLU(name=name + '.' + 'out_init')(out_init)

        return out_init


def enet_encoder(input_layer, train, n_classes=10):
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
    out_init = init_block(input_layer)

    # --------first block---------
    # first bottleneck with downsampling
    bt1_0 = bottleneck(out_init,train,output_filters=64,dropout_prob=0.01,
                       downsample=True,name='bt1_0_ds')

    # four bottlenecks without downsampling
    bt1_1 = bottleneck(bt1_0, train, output_filters=64, dropout_prob=0.01, name='bt1_1')
    bt1_2 = bottleneck(bt1_1, train, output_filters=64, dropout_prob=0.01, name='bt1_2')
    bt1_3 = bottleneck(bt1_2, train, output_filters=64, dropout_prob=0.01, name='bt1_3')
    bt1_4 = bottleneck(bt1_3, train, output_filters=64, dropout_prob=0.01, name='bt1_4')

    # --------second block---------
    bt2_0 = bottleneck(bt1_4, train, output_filters=128, downsample=True, name='bt2_0_ds')
    bt2_1 = bottleneck(bt2_0, train, output_filters=128, name='bt2_1')
    bt2_2 = bottleneck(
        bt2_1,
        train,
        output_filters=128,
        dilation_rate=[2, 2],
        name='bt2_2_dl')
    bt2_3 = bottleneck(
        bt2_2, train, output_filters=128, kernel_size=[7, 1], name='bt2_3_as')
    bt2_4 = bottleneck(
        bt2_3,
        train,
        output_filters=128,
        dilation_rate=[4, 4],
        name='bt2_4_dl')
    bt2_5 = bottleneck(bt2_4, train, output_filters=128, name='bt2_5')
    bt2_6 = bottleneck(
        bt2_5, train, output_filters=128, dilation_rate=[8, 8], name='bt2_6_d')
    bt2_7 = bottleneck(
        bt2_6, train, output_filters=128, kernel_size=[5, 1], name='bt2_7_as')
    bt2_8 = bottleneck(
        bt2_7,
        train,
        output_filters=128,
        dilation_rate=[16, 16],
        name='bt2_8_dl')

    # --------third block---------
    bt3_0 = bottleneck(bt2_8, train, output_filters=128, name='bt3_0')
    bt3_1 = bottleneck(
        bt3_0,
        train,
        output_filters=128,
        dilation_rate=[2, 2],
        name='bt3_1_dl')
    bt3_2 = bottleneck(
        bt3_1, train, output_filters=128, kernel_size=[7, 1], name='bt3_2_as')
    bt3_3 = bottleneck(
        bt3_2,
        train,
        output_filters=128,
        dilation_rate=[4, 4],
        name='bt3_3_dl')
    bt3_4 = bottleneck(bt3_3, train, output_filters=128, name='bt3_4')
    bt3_5 = bottleneck(
        bt3_4, train, output_filters=128, dilation_rate=[8, 8], name='bt3_5_d')
    bt3_6 = bottleneck(
        bt3_5, train, output_filters=128, kernel_size=[5, 1], name='bt3_6_as')
    bt3_7 = bottleneck(
        bt3_6,
        train,
        output_filters=128,
        dilation_rate=[16, 16],
        name='bt3_7_dl')

    # --------compression to get to a downsampled representation of segmentation---------
    output_layer = tf.keras.layers.Conv2D(
        n_classes, [1, 1], strides=[1, 1], padding='valid')(bt3_7)
    # output_layer = tf.layers.conv2d(inputs=bt3_7,
    #                                 filters=n_classes,
    #                                 kernel_size=[1,1],
    #                                 strides=[1,1],
    #                                 padding='valid')

    return output_layer


def enet_encoder_v3(input_layer, train, n_classes=10):
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
    out_init = init_block(input_layer)

    # --------first block---------
    # first bottleneck with downsampling
    bt1_0 = bottleneck(
        out_init,
        train,
        output_filters=64,
        dropout_prob=0.01,
        downsample=True,
        name='bt1_0_ds')

    # four bottlenecks without downsampling
    bt1_1 = bottleneck(
        bt1_0, train, output_filters=64, dropout_prob=0.01, name='bt1_1')
    bt1_2 = bottleneck(
        bt1_1, train, output_filters=64, dropout_prob=0.01, name='bt1_2')
    bt1_3 = bottleneck(
        bt1_2, train, output_filters=64, dropout_prob=0.01, name='bt1_3')
    bt1_4 = bottleneck(
        bt1_3, train, output_filters=64, dropout_prob=0.01, name='bt1_4')

    # --------second block---------
    bt2_0 = bottleneck(
        bt1_4, train, output_filters=128, downsample=True, name='bt2_0_ds')
    bt2_1 = bottleneck(bt2_0, train, output_filters=128, name='bt2_1')
    bt2_2 = bottleneck(
        bt2_1,
        train,
        output_filters=128,
        dilation_rate=[2, 2],
        name='bt2_2_dl')
    bt2_3 = bottleneck(
        bt2_2, train, output_filters=128, kernel_size=[7, 1], name='bt2_3_as')
    bt2_4 = bottleneck(
        bt2_3,
        train,
        output_filters=128,
        dilation_rate=[4, 4],
        name='bt2_4_dl')
    bt2_5 = bottleneck(bt2_4, train, output_filters=128, name='bt2_5')
    bt2_6 = bottleneck(
        bt2_5, train, output_filters=128, dilation_rate=[8, 8], name='bt2_6_d')
    bt2_7 = bottleneck(
        bt2_6, train, output_filters=128, kernel_size=[5, 1], name='bt2_7_as')
    bt2_8 = bottleneck(
        bt2_7,
        train,
        output_filters=256,
        dilation_rate=[16, 16],
        name='bt2_8_dl')
    bt2_9 = bottleneck(
        bt2_8, train, output_filters=256, downsample=True, name='bt2_9_ds')

    # --------third block---------
    bt3_0 = bottleneck(bt2_9, train, output_filters=256, name='bt3_0')
    bt3_1 = bottleneck(
        bt3_0,
        train,
        output_filters=256,
        dilation_rate=[2, 2],
        name='bt3_1_dl')
    bt3_2 = bottleneck(
        bt3_1, train, output_filters=256, kernel_size=[7, 1], name='bt3_2_as')
    bt3_3 = bottleneck(
        bt3_2,
        train,
        output_filters=256,
        dilation_rate=[4, 4],
        name='bt3_3_dl')
    bt3_4 = bottleneck(bt3_3, train, output_filters=256, name='bt3_4')
    bt3_5 = bottleneck(
        bt3_4, train, output_filters=256, dilation_rate=[8, 8], name='bt3_5_d')
    bt3_6 = bottleneck(
        bt3_5, train, output_filters=256, kernel_size=[5, 1], name='bt3_6_as')
    bt3_7 = bottleneck(
        bt3_6,
        train,
        output_filters=512,
        dilation_rate=[16, 16],
        name='bt3_7_dl')
    bt3_8 = bottleneck(
        bt3_7, train, output_filters=512, downsample=True, name='bt3_8_ds')

    # --------logits---------
    r_mean = tf.reduce_mean(bt3_8, axis=[1, 2], keepdims=True)
    r_mean = tf.identity(r_mean, 'final_reduce_mean')
    r_mean_reshape = tf.reshape(r_mean, [-1, r_mean.get_shape().as_list()[-1]])
    logits = tf.layers.dense(inputs=r_mean_reshape, units=n_classes)
    logits = tf.identity(logits, 'logits')

    return logits


def enet_encoder_mnist(input_layer, n_classes=10, dropout=0.0):
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
    'dropout' = dropout for init and bottleneck blocks

    Returns
    -------
    'logits' = A `Tensor` with the same type as `input_layer` and shape [batch_size,n_classes]
    '''

    # ---------Initial block---------
    out_init = init_block(input_layer, conv_filters=15)  # 14x14

    # --------first block---------
    # first bottleneck with downsampling
    bt1_0 = bottleneck(
        out_init,
        output_filters=32,
        dropout_prob=dropout,
        downsample=True,
        name='bt1_0_ds')  # 7x7

    # four bottlenecks without downsampling
    bt1_1 = bottleneck(bt1_0, output_filters=64, dropout_prob=dropout, name='bt1_1')
    bt1_2 = bottleneck(bt1_1, output_filters=64, dropout_prob=dropout, name='bt1_2')
    bt1_3 = bottleneck(bt1_2, output_filters=64, dropout_prob=dropout, name='bt1_3')
    bt1_4 = bottleneck(bt1_3, output_filters=64, dropout_prob=dropout, name='bt1_4')

    # four bottlenecks without downsampling
    bt1_5 = bottleneck(bt1_4, output_filters=64, dropout_prob=dropout, name='bt1_5')
    bt1_6 = bottleneck(bt1_5, output_filters=64, dropout_prob=dropout, name='bt1_6')
    bt1_7 = bottleneck(bt1_6, output_filters=64, dropout_prob=dropout, name='bt1_7')
    bt1_8 = bottleneck(bt1_7, output_filters=64, dropout_prob=dropout, name='bt1_8')

    # four bottlenecks without downsampling
    bt1_9 = bottleneck(bt1_8, output_filters=128, dropout_prob=dropout, name='bt1_9')
    bt1_10 = bottleneck(bt1_9, output_filters=128, dropout_prob=dropout, name='bt1_10')
    bt1_11 = bottleneck(bt1_10, output_filters=128, dropout_prob=dropout, name='bt1_11')
    bt1_12 = bottleneck(bt1_11, output_filters=n_classes, dropout_prob=dropout, name='bt1_12')

    # logits
    pre_logits = tf.keras.layers.AvgPool2D(pool_size=(7, 7),
                                           padding='valid',
                                           name='pre_logits')(bt1_12)
    flat_pre_logits = tf.keras.layers.Flatten()(pre_logits)
    logits = tf.keras.layers.Softmax(name='logits')(flat_pre_logits)

    return logits


def mnist_test(input_layer):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    # input_layer = tf.reshape(input_layer, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.keras.layers.Conv2D(
        32, (5, 5),
        padding='same',
        activation=tf.keras.activations.relu,
        input_shape=input_layer.get_shape().as_list()[1:])(input_layer)

    # Pooling Layer #1
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv1)

    # Convolutional Layer #2
    conv2 = tf.keras.layers.Conv2D(
        64, (5, 5), padding='same',
        activation=tf.keras.activations.relu)(pool1)

    # Pooling Layer #2
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv2)

    # Flatten tensor into a batch of vectors
    pool2_flat = tf.keras.layers.Flatten()(pool2)

    # Dense Layer
    dense = tf.keras.layers.Dense(
        1024, activation=tf.keras.activations.relu)(pool2_flat)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.keras.layers.Dropout(0.4)(dense)

    # Output Tensor Shape: [batch_size, 10]
    logits = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dropout)

    return logits
