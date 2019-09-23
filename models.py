import tensorflow as tf


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
        self.bt1_1 = BottleEnc(output_filters=64,
                               dropout_prob=dropout,
                               name='bt1_1')
        self.bt1_2 = BottleEnc(output_filters=64,
                               dropout_prob=dropout,
                               name='bt1_2')
        self.bt1_3 = BottleEnc(output_filters=64,
                               dropout_prob=dropout,
                               name='bt1_3')
        self.bt1_4 = BottleEnc(output_filters=64,
                               dropout_prob=dropout,
                               name='bt1_4')

        # second four bottlenecks without downsampling
        self.bt1_5 = BottleEnc(output_filters=64,
                               dropout_prob=dropout,
                               name='bt1_5')
        self.bt1_6 = BottleEnc(output_filters=64,
                               dropout_prob=dropout,
                               name='bt1_6')
        self.bt1_7 = BottleEnc(output_filters=64,
                               dropout_prob=dropout,
                               name='bt1_7')
        self.bt1_8 = BottleEnc(output_filters=64,
                               dropout_prob=dropout,
                               name='bt1_8')

        # third four bottlenecks without downsampling
        self.bt1_9 = BottleEnc(output_filters=128,
                               dropout_prob=dropout,
                               name='bt1_9')
        self.bt1_10 = BottleEnc(output_filters=128,
                                dropout_prob=dropout,
                                name='bt1_10')
        self.bt1_11 = BottleEnc(output_filters=128,
                                dropout_prob=dropout,
                                name='bt1_11')
        self.bt1_12 = BottleEnc(output_filters=n_classes,
                                dropout_prob=dropout,
                                name='bt1_12')

        # logits
        self.pre_logits = tf.keras.layers.AvgPool2D(pool_size=(7, 7),
                                                    padding='valid',
                                                    name='pre_logits')
        self.flat_pre_logits = tf.keras.layers.Flatten()
        self.logits = tf.keras.layers.Softmax(name='logits')

    def call(self, input_layer):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).

        x = self.out_init(input_layer)

        # first bottleneck with downsampling
        x, argmax = self.bt1_0(x)

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
    conv2 = tf.keras.layers.Conv2D(64, (5, 5),
                                   padding='same',
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
