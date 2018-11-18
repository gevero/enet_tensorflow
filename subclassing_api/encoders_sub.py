import tensorflow as tf


class bottleneck(tf.keras.Model):
    '''
    Enet bottleneck module as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural
        Network Architecture for Real-Time Semantic Segmentation. arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html


    Arguments
    ----------
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

    def __init__(self,
        output_filters=128,
        kernel_size=[3, 3],
        kernel_strides=[1, 1],
        padding='same',
        dilation_rate=[1, 1],
        internal_comp_ratio=4,
        dropout_prob=0.1,
        downsample=False,
        name='bottleneck'):
        super(bottleneck, self).__init__(name=name)

        # ------- bottleneck parameters -------
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.kernel_strides = kernel_strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.internal_comp_ratio = internal_comp_ratio
        self.dropout_prob = dropout_prob
        self.downsample = downsample
        

        # Derived parameters
        self.internal_filters = self.output_filters // self.internal_comp_ratio

        # downsampling or not
        if self.downsample:
            self.down_kernel = [2, 2]
            self.down_strides = [2, 2]
        else:
            self.down_kernel = [1, 1]
            self.down_strides = [1, 1]

        # ------- main connection layers -------

        # bottleneck representation compression with valid padding
        # 1x1 usually, 2x2 if downsampling
        self.main1_1 = tf.keras.layers.Conv2D(
            self.internal_filters,
            self.down_kernel,
            strides=self.down_strides,
            use_bias=False,
            name=self.name + '.' + 'main1_1')
        self.main1_2 = tf.keras.layers.BatchNormalization(
            name=self.name + '.' + 'main1_2')
        self.main1_3 = tf.keras.layers.PReLU(name=self.name + '.' + 'main1_3')

        # central convolution
        self.asym_flag = self.kernel_size[0] != self.kernel_size[1]
        self.main1_4 = tf.keras.layers.Conv2D(
            self.internal_filters,
            self.kernel_size,
            strides=self.kernel_strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            use_bias=not(self.asym_flag),
            name=self.name + '.' + 'main1_4a')
        if self.asym_flag:
            self.main1_4b = tf.keras.layers.Conv2D(
                self.internal_filters,
                self.kernel_size[::-1],
                strides=self.kernel_strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                name=self.name + '.' + 'main1_4b')
        self.main1_5 = tf.keras.layers.BatchNormalization(
            name=self.name + '.' + 'main1_5')
        self.main1_6 = tf.keras.layers.PReLU(name=self.name + '.' + 'main1_6')

        # bottleneck representation expansion with 1x1 valid convolution
        self.main1_7 = tf.keras.layers.Conv2D(
            self.output_filters, [1, 1],
            strides=[1, 1],
            use_bias=False,
            name=self.name + '.' + 'main1_7')
        self.main1_8 = tf.keras.layers.BatchNormalization(
            name=self.name + '.' + 'main1_8')
        self.main1_9 = tf.keras.layers.SpatialDropout2D(
            dropout_prob, name=self.name + '.' + 'main1_9')

        # ------- skip connection layers -------

        # downsampling layer
        self.skip1_1 = tf.keras.layers.MaxPool2D(
            pool_size=self.down_kernel,
            strides=self.down_strides,
            name=self.name + '.' + 'skip1_1a')

        # matching filter dimension with learned 1x1 convolution
        self.skip1_2 = tf.keras.layers.Conv2D(
            self.output_filters,
            kernel_size=[1, 1],
            padding='valid',
            use_bias=False,
            name=name + '.' + 'filter_matching')

        # ------- output layer -------
        self.addition = tf.keras.layers.Add(name=self.name + '.' + 'addition')
        self.prelu = tf.keras.layers.PReLU(name=self.name + '.' + 'output_layer')

    def call(self, input_layer):

        # input filter from incoming layer
        input_filters = input_layer.get_shape().as_list()[-1]

        # ----- main connection ------
        main = self.main1_1(input_layer)
        main = self.main1_2(main)
        main = self.main1_3(main)
        main = self.main1_4(main)
        if self.asym_flag:
            main = self.main1_4b(main)
        main = self.main1_5(main)
        main = self.main1_6(main)
        main = self.main1_7(main)
        main = self.main1_8(main)
        main = self.main1_9(main)

        # ----- skip connection ------
        skip = input_layer

        # downsampling if necessary
        if self.downsample:
            print('here')
            skip = self.skip1_1(input_layer)
            
        # matching filter dimension with learned 1x1 convolution
        if input_filters != self.output_filters:
            skip = self.skip1_2(skip)
                
        # ------- output layer -------
        addition_layer = self.addition([main,skip])
        output_layer = self.prelu(addition_layer)
                
        return output_layer


class init_block(tf.keras.Model):
    '''
    Enet init_block as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural Network
        Architecture for Real-Time Semantic Segmentation. arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html


    Arguments
    ----------
    'conv_filters' = an `Integer`: number filters for the convolution (or channels, if you like it)
    'kernel_size' = a `List`: size of the kernel for the convolution
    'kernel_strides' = a `List`: length of the strides for the convolution
    'pool_size' = a `List`: size of the pool for the maxpooling
    'pool_strides' = a `List`: length of the strides for the maxpooling
    'padding' = a `String`: padding for the convolution and the maxpooling
    'name' = a `String`: name of the init_block
    '''

    def __init__(self,
        conv_filters=13,
        kernel_size=[3, 3],
        kernel_strides=[2, 2],
        pool_size=[2, 2],
        pool_strides=[2, 2],
        padding='valid',
        name='init_block'):
        super(init_block, self).__init__(name=name)

        # ------- init_block parameters -------
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.kernel_strides = kernel_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.padding = padding

        # ------- init_block layers -------
        
        # conv connection: need the padding to match the dimension of pool_init
        self.padded_init = tf.keras.layers.ZeroPadding2D()
        self.conv_init = tf.keras.layers.Conv2D(
            conv_filters, kernel_size, strides=kernel_strides,
            padding='valid')

        # maxpooling connection, where pool_init is to be concatenated with conv_init
        self.pool_init = tf.keras.layers.MaxPool2D(
            pool_size=pool_size, strides=pool_strides,
            padding='valid')

        # concatenating the two connections
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.prelu = tf.keras.layers.PReLU(name=self.name + '.' + 'out_init')


    def call(self, input_layer):

        # ----- conv connection ------
        # conv connection: need the padding to match the dimension of pool_init
        conv_conn = self.padded_init(input_layer)
        conv_conn = self.conv_init(conv_conn)
  
        # ----- pool connection ------
        pool_conn = self.pool_init(input_layer)
            
        # ------- concat to output layer -------
        output_layer = self.concatenate([conv_conn, pool_conn])
        output_layer = self.batch_norm(output_layer)
        output_layer = self.prelu(output_layer)
        
        return output_layer


class fashion_mnist_enc(tf.keras.Model):
    '''
    Slimmed down Enet encoder for  fashion mnist. Actually, does not look like enet at all,
    but uses the bottleneck module as a flexible building block.
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

    def __init__(self, n_classes=10, dropout=0.0):
        super(fashion_mnist_enc, self).__init__(name='fashion_mnist_enc')
        self.n_classes = n_classes
        self.dropout = dropout

        # init layer
        self.out_init = init_block(conv_filters=15)  # 14x14

        # first bottleneck with downsampling
        self.bt1_0 = bottleneck(output_filters=32,
            dropout_prob=dropout,
            downsample=True,
            name='bt1_0_ds')  # 7x7

        # first four bottlenecks without downsampling
        self.bt1_1 = bottleneck(output_filters=64, dropout_prob=dropout, name='bt1_1')
        self.bt1_2 = bottleneck(output_filters=64, dropout_prob=dropout, name='bt1_2')
        self.bt1_3 = bottleneck(output_filters=64, dropout_prob=dropout, name='bt1_3')
        self.bt1_4 = bottleneck(output_filters=64, dropout_prob=dropout, name='bt1_4')

        # second four bottlenecks without downsampling
        self.bt1_5 = bottleneck(output_filters=64, dropout_prob=dropout, name='bt1_5')
        self.bt1_6 = bottleneck(output_filters=64, dropout_prob=dropout, name='bt1_6')
        self.bt1_7 = bottleneck(output_filters=64, dropout_prob=dropout, name='bt1_7')
        self.bt1_8 = bottleneck(output_filters=64, dropout_prob=dropout, name='bt1_8')

        # third four bottlenecks without downsampling
        self.bt1_9 = bottleneck(output_filters=128, dropout_prob=dropout, name='bt1_9')
        self.bt1_10 = bottleneck(output_filters=128, dropout_prob=dropout, name='bt1_10')
        self.bt1_11 = bottleneck(output_filters=128, dropout_prob=dropout, name='bt1_11')
        self.bt1_12 = bottleneck(output_filters=n_classes, dropout_prob=dropout, name='bt1_12')

        # logits
        self.pre_logits = tf.keras.layers.AvgPool2D(
            pool_size=(7, 7), padding='valid', name='pre_logits')
        self.flat_pre_logits = tf.keras.layers.Flatten()
        self.logits = tf.keras.layers.Softmax(name='logits')

    def call(self, input_layer):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        
        x = self.out_init(input_layer)

        # first bottleneck
        x = self.bt1_0(x)

        # first four bottlenecks
        x = self.bt1_1(x)
        x = self.bt1_2(x)
        x = self.bt1_3(x)
        x = self.bt1_4(x)

        # second four bottlenecks
        x = self.bt1_5(x)
        x = self.bt1_6(x)
        x = self.bt1_7(x)
        x = self.bt1_8(x)

        # third four bottlenecks
        x = self.bt1_9(x)
        x = self.bt1_10(x)
        x = self.bt1_11(x)
        x = self.bt1_12(x)

        # logits
        x = self.pre_logits(x)
        x = self.flat_pre_logits(x)
        output_layer = self.logits(x)
        
        return output_layer


def fashion_mnist_baseline(input_layer):
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
