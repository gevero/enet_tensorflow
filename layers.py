import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils

# Tensorflow implementations of max_pooling and unpooling


# Keras layers for pooling and unpooling
class MaxPoolWithArgmax2D(Layer):
    """2D Pooling layer with pooling indices.

    Arguments
    ----------
    'pool_size' = An integer or tuple/list of 2 integers:
        (pool_height, pool_width) specifying the size of the
        pooling window. Can be a single integer to specify
        the same value for all spatial dimensions.
    'strides' = An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    'padding' = A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
    'data_format' = A string, one of `channels_last` (default)
        or `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds
        to inputs with shape `(batch, channels, height, width)`.
    'name' = A string, the name of the layer.
    """
    def __init__(self,
                 pool_size,
                 strides,
                 padding='valid',
                 data_format=None,
                 name=None,
                 **kwargs):
        super(MaxPoolWithArgmax2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs):

        pool_shape = (1, ) + self.pool_size + (1, )
        strides = (1, ) + self.strides + (1, )

        if self.data_format == 'channels_last':
            outputs, argmax = tf.nn.max_pool_with_argmax(
                inputs,
                ksize=pool_shape,
                strides=strides,
                padding=self.padding.upper())
            return (outputs, argmax)
        else:
            outputs, argmax = tf.nn.max_pool_with_argmax(
                tf.transpose(inputs, perm=[0, 2, 3, 1]),
                ksize=pool_shape,
                strides=strides,
                padding=self.padding.upper())
            return (tf.transpose(outputs, perm=[0, 3, 1, 2]),
                    tf.transpose(argmax, perm=[0, 3, 1, 2]))

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], rows, cols])
        else:
            return tensor_shape.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'padding': self.padding,
            'strides': self.strides,
            'data_format': self.data_format
        }
        base_config = super(MaxPoolingWithArgmax2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxUnpool2D(Layer):
    def __init__(self, data_format='channels_last', name=None, **kwargs):
        super(MaxUnpool2D, self).__init__(**kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(min_ndim=2, max_ndim=4)

    def call(self, inputs, argmax, spatial_output_shape):

        # standardize spatial_output_shape
        spatial_output_shape = conv_utils.normalize_tuple(
            spatial_output_shape, 2, 'spatial_output_shape')

        # getting input shape
        input_shape = tf.shape(inputs)

        # checking if spatial shape is ok
        if self.data_format == 'channels_last':
            output_shape = (input_shape[0],) + \
                spatial_output_shape + (input_shape[3],)
            assert output_shape[1] * output_shape[2] > tf.math.reduce_max(
                argmax), "HxW <= Max(argmax)"
        else:
            output_shape = (input_shape[0],
                            input_shape[1]) + spatial_output_shape
            assert output_shape[2] * output_shape[3] > tf.math.reduce_max(
                argmax), "HxW <= Max(argmax)"

        # N * H_in * W_in * C
        flat_input_size = tf.reduce_prod(input_shape)

        # flat output_shape = [N, H_out * W_out * C]
        flat_output_shape = [
            output_shape[0],
            output_shape[1] * output_shape[2] * output_shape[3]
        ]

        # flatten input tensor for the use in tf.scatter_nd
        inputs_ = tf.reshape(inputs, [flat_input_size])

        # create the tensor [ [[[0]]], [[[1]]], ..., [[[N-1]]] ]
        # corresponding to the batch size but transposed in 4D
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64),
                                          dtype=argmax.dtype),
                                 shape=[input_shape[0], 1, 1, 1])

        # b is a tensor of size (N, H, W, C) or (N, C, H, W) whose
        # first element of the batch are 3D-array full of 0, ...
        # second element of the batch are 3D-array full of 1, ...
        b = tf.ones_like(argmax) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])

        # argmax_ = [ [0, argmax_1], [0, argmax_2], ... [0, argmax_k], ...,
        # [N-1, argmax_{N*H*W*C}], [N-1, argmax_{N*H*W*C-1}] ]
        argmax_ = tf.reshape(argmax, [flat_input_size, 1])
        argmax_ = tf.concat([b, argmax_], axis=-1)

        # reshaping output tensor
        ret = tf.scatter_nd(argmax_,
                            inputs_,
                            shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)
        # ret.set_shape(output_shape)

        return ret

    def compute_output_shape(self, input_shape, spatial_output_shape):

        # getting input shape
        input_shape = tensor_shape.TensorShape(input_shape).as_list()

        # standardize spatial_output_shape
        spatial_output_shape = conv_utils.normalize_tuple(
            spatial_output_shape, 2, 'spatial_output_shape')

        # checking if spatial shape is ok
        if self.data_format == 'channels_last':
            output_shape = (input_shape[0],) + \
                self.spatial_output_shape + (input_shape[3],)
            assert output_shape[1] * output_shape[2] > tf.math.reduce_max(
                self.argmax), "HxW <= Max(argmax)"
        else:
            output_shape = (input_shape[0],
                            input_shape[1]) + self.spatial_output_shape
            assert output_shape[2] * output_shape[3] > tf.math.reduce_max(
                self.argmax), "HxW <= Max(argmax)"

        return output_shape

        def get_config(self):
            config = {
                'spatial_output_shape': self.spatial_output_shape,
                'data_format': self.data_format
            }
            base_config = super(MaxPoolingWithArgmax2D, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


class BottleNeck(tf.keras.Model):
    '''
    Enet bottleneck module as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural
        Network Architecture for Real-Time Semantic Segmentation.
        arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html

    This is the general bottleneck modules. It is used both in the encoding and
    decoding paths, the only exception being the upsampling decoding, where
    we use BottleDeck

    Arguments
    ----------
    'output_filters' = an `Integer`: number of output filters
    'kernel_size' = a `List`: size of the kernel for the central convolution
    'kernel_strides' = a `List`: length of the strides for the central conv
    'padding' = a `String`: padding of the central convolution
    'dilation_rate' = a `List`: dilation rate of the central convolution
    'internal_comp_ratio' = an `Integer`: compression ratio of the bottleneck
    'dropout_prob' = a `float`: dropout at the end of the main connection
    'downsample' = a `String`: downsampling flag
    'name' = a `String`: name of the bottleneck

    Returns
    -------
    'output_layer' = A `Tensor` with the same type as `input_layer`
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
                 name='BottleEnc'):
        super(BottleNeck, self).__init__(name=name)

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
        self.main1_1 = tf.keras.layers.Conv2D(self.internal_filters,
                                              self.down_kernel,
                                              strides=self.down_strides,
                                              use_bias=False,
                                              name=self.name + '.' + 'main1_1')
        self.main1_2 = tf.keras.layers.BatchNormalization(name=self.name +
                                                          '.' + 'main1_2')
        self.main1_3 = tf.keras.layers.PReLU(name=self.name + '.' + 'main1_3')

        # central convolution
        self.asym_flag = self.kernel_size[0] != self.kernel_size[1]
        self.main1_4 = tf.keras.layers.Conv2D(self.internal_filters,
                                              self.kernel_size,
                                              strides=self.kernel_strides,
                                              padding=self.padding,
                                              dilation_rate=self.dilation_rate,
                                              use_bias=not (self.asym_flag),
                                              name=self.name + '.' +
                                              'main1_4a')
        if self.asym_flag:
            self.main1_4b = tf.keras.layers.Conv2D(
                self.internal_filters,
                self.kernel_size[::-1],
                strides=self.kernel_strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                name=self.name + '.' + 'main1_4b')
        self.main1_5 = tf.keras.layers.BatchNormalization(name=self.name +
                                                          '.' + 'main1_5')
        self.main1_6 = tf.keras.layers.PReLU(name=self.name + '.' + 'main1_6')

        # bottleneck representation expansion with 1x1 valid convolution
        self.main1_7 = tf.keras.layers.Conv2D(self.output_filters, [1, 1],
                                              strides=[1, 1],
                                              use_bias=False,
                                              name=self.name + '.' + 'main1_7')
        self.main1_8 = tf.keras.layers.BatchNormalization(name=self.name +
                                                          '.' + 'main1_8')
        self.main1_9 = tf.keras.layers.SpatialDropout2D(dropout_prob,
                                                        name=self.name + '.' +
                                                        'main1_9')

        # ------- skip connection layers -------

        # downsampling layer
        self.skip1_1 = MaxPoolWithArgmax2D(pool_size=self.down_kernel,
                                           strides=self.down_strides,
                                           name=self.name + '.' + 'skip1_1a')

        # matching filter dimension with learned 1x1 convolution
        # this is done differently than in vanilla enet, where
        # you shold just pad with zeros.
        self.skip1_2 = tf.keras.layers.Conv2D(self.output_filters,
                                              kernel_size=[1, 1],
                                              padding='valid',
                                              use_bias=False,
                                              name=name + '.' +
                                              'filter_matching')

        # ------- output layer -------
        self.addition = tf.keras.layers.Add(name=self.name + '.' + 'addition')
        self.prelu = tf.keras.layers.PReLU(name=self.name + '.' +
                                           'output_layer')

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
            skip, argmax = self.skip1_1(input_layer)

        # matching filter dimension with learned 1x1 convolution
        # this is done differently than in vanilla enet, where
        # you shold just pad with zeros.
        if input_filters != self.output_filters:
            skip = self.skip1_2(skip)

        # ------- output layer -------
        addition_layer = self.addition([main, skip])
        output_layer = self.prelu(addition_layer)

        # I need the input layer, I see no other way round
        # because i neet to pass it to the decoder
        if self.downsample:
            return output_layer, argmax, input_layer
        else:
            return output_layer


class BottleDeck(tf.keras.Model):
    '''
    Enet bottleneck module as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural
        Network Architecture for Real-Time Semantic Segmentation.
        arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html

    This is the general bottleneck decoding modules. It is used only in the
    decoding path when we use the upsampling. In the forward pass we have
    three input tensors:
    - input: the real input tensor
    - enc_tensor: coming from the encoder path, used to get the shape of the
                 output tensor
    - argmax: the tensor for the mapping of the upsampled values

    Arguments
    ----------
    'output_filters' = an `Integer`: number of output filters
    'kernel_size' = a `List`: size of the kernel for the central convolution
    'kernel_strides' = a `List`: length of the strides for the central conv
    'padding' = a `String`: padding of the central convolution
    'dilation_rate' = a `List`: dilation rate of the central convolution
    'internal_comp_ratio' = an `Integer`: compression ratio of the bottleneck
    'dropout_prob' = a `float`: dropout at the end of the main connection
    'upsample' = a `String`: upsampling flag
    'name' = a `String`: name of the bottleneck

    Returns
    -------
    'output_layer' = A `Tensor` with the same type as `input_layer`
    '''
    def __init__(self,
                 output_filters=128,
                 kernel_size=[3, 3],
                 kernel_strides=[2, 2],
                 padding='same',
                 dilation_rate=[1, 1],
                 internal_comp_ratio=4,
                 dropout_prob=0.1,
                 name='BottleDeck'):
        super(BottleDeck, self).__init__(name=name)

        # ------- bottleneck parameters -------
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.kernel_strides = kernel_strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.internal_comp_ratio = internal_comp_ratio
        self.dropout_prob = dropout_prob

        # Derived parameters
        self.internal_filters = self.output_filters // self.internal_comp_ratio

        # ------- main connection layers -------

        # bottleneck representation compression with valid padding
        # 1x1 usually, 2x2 if downsampling
        self.main1_1 = tf.keras.layers.Conv2D(self.internal_filters,
                                              kernel=[1, 1],
                                              strides=[1, 1],
                                              use_bias=False,
                                              name=self.name + '.' + 'main1_1')
        self.main1_2 = tf.keras.layers.BatchNormalization(name=self.name +
                                                          '.' + 'main1_2')
        self.main1_3 = tf.keras.layers.PReLU(name=self.name + '.' + 'main1_3')

        # central convolution: am i using "same" padding?
        self.main1_4 = tf.keras.layers.Conv2DTranspose(
            self.internal_filters,
            self.kernel_size,
            strides=self.kernel_strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            use_bias=True,
            name=self.name + '.' + 'main1_4a')
        self.main1_5 = tf.keras.layers.BatchNormalization(name=self.name +
                                                          '.' + 'main1_5')
        self.main1_6 = tf.keras.layers.PReLU(name=self.name + '.' + 'main1_6')

        # bottleneck representation expansion with 1x1 valid convolution
        self.main1_7 = tf.keras.layers.Conv2D(self.output_filters, [1, 1],
                                              strides=[1, 1],
                                              use_bias=False,
                                              name=self.name + '.' + 'main1_7')
        self.main1_8 = tf.keras.layers.BatchNormalization(name=self.name +
                                                          '.' + 'main1_8')
        self.main1_9 = tf.keras.layers.SpatialDropout2D(dropout_prob,
                                                        name=self.name + '.' +
                                                        'main1_9')

        # ------- skip connection layers -------

        # downsampling layer
        self.skip1_1 = MaxUnpool2D(name=self.name + '.' + 'skip1_1a')

        # matching filter dimension with learned 1x1 convolution
        # this is done differently than in vanilla enet, where
        # you shold just pad with zeros.
        self.skip1_2 = tf.keras.layers.Conv2D(self.output_filters,
                                              kernel_size=[1, 1],
                                              padding='valid',
                                              use_bias=False,
                                              name=name + '.' +
                                              'filter_matching')

        # ------- output layer -------
        self.addition = tf.keras.layers.Add(name=self.name + '.' + 'addition')
        self.prelu = tf.keras.layers.PReLU(name=self.name + '.' +
                                           'output_layer')

    def call(self, input_layer, argmax, upsample_layer):

        # input filter from incoming layer, and upsample layer spatial shape
        input_filters = input_layer.get_shape().as_list()[-1]
        upsample_layer_shape = upsample_layer.get_s, hape().as_list()[1:3]

        # ----- main connection ------
        main = self.main1_1(input_layer)
        main = self.main1_2(main)
        main = self.main1_3(main)

        main = self.main1_4(main)
        main = self.main1_5(main)
        main = self.main1_6(main)

        main = self.main1_7(main)
        main = self.main1_8(main)
        main = self.main1_9(main)

        # ----- skip connection ------
        skip = input_layer

        # downsampling if necessary
        skip = self.skip1_1(input_layer, argmax, upsample_layer_shape)

        # matching filter dimension with learned 1x1 convolution
        # this is done differently than in vanilla enet, where
        # you shold just pad with zeros.
        if input_filters != self.output_filters:
            skip = self.skip1_2(skip)

        # ------- output layer -------
        addition_layer = self.addition([main, skip])
        output_layer = self.prelu(addition_layer)

        return output_layer


class init_block(tf.keras.Model):
    '''
    Enet init_block as in:
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E. ENet: A Deep Neural Network
        Architecture for Real-Time Semantic Segmentation. arXiv:1606.02147 [cs] 2016.
    (2) https://github.com/e-lab/ENet-training/blob/master/train/models/encoderI.lua
    (3) https://culurciello.github.io/tech/2016/06/20/training-enet.html


    Arguments
    ----------
    'conv_filters' = an `Integer`: number filters for the convolution
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
        self.conv_init = tf.keras.layers.Conv2D(conv_filters,
                                                kernel_size,
                                                strides=kernel_strides,
                                                padding='valid')

        # maxpool, where pool_init is to be concatenated with conv_init
        self.pool_init = tf.keras.layers.MaxPool2D(pool_size=pool_size,
                                                   strides=pool_strides,
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
