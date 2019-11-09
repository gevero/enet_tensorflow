import tensorflow as tf
from layers import BottleDeck, BottleNeck, InitBlock
import numpy as np


class EnetModel(tf.keras.Model):
    '''
    Enet encoder.
    (1) Paszke, A.; Chaurasia, A.; Kim, S.; Culurciello, E.
        ENet: A Deep Neural Network Architecture for Real-Time Semantic
        Segmentation. arXiv:1606.02147 [cs] 2016.

    Arguments
    ----------
    'input_layer' = input `Tensor` with type `float32` and
                    shape [batch_size,w,h,1]
    'n_classes' = an `Integer`: number of classes
    'dropout' = dropout for init and bottleneck blocks

    Returns
    -------
    'output_layer' = A `Tensor` with the same type as `input_layer`
    '''
    def __init__(self, C=12, l2=0.0, MultiObjective=False, **kwargs):
        super(EnetModel, self).__init__(**kwargs)

        # initialize parameters
        self.C = C
        self.l2 = l2
        self.MultiObjective = MultiObjective

        # # layers
        self.InitBlock = InitBlock(conv_filters=13)

        # # first block of bottlenecks
        self.BNeck1_0 = BottleNeck(output_filters=64,
                                   downsample=True,
                                   dropout_prob=0.01,
                                   l2=l2,
                                   name='BNeck1_0')
        self.BNeck1_1 = BottleNeck(output_filters=64,
                                   dropout_prob=0.01,
                                   l2=l2,
                                   name='BNeck1_1')
        self.BNeck1_2 = BottleNeck(output_filters=64,
                                   dropout_prob=0.01,
                                   l2=l2,
                                   name='BNeck1_2')
        self.BNeck1_3 = BottleNeck(output_filters=64,
                                   dropout_prob=0.01,
                                   l2=l2,
                                   name='BNeck1_3')
        self.BNeck1_4 = BottleNeck(output_filters=64,
                                   dropout_prob=0.01,
                                   l2=l2,
                                   name='BNeck1_4')

        # # second block of bottlenecks
        self.BNeck2_0 = BottleNeck(output_filters=128,
                                   downsample=True,
                                   l2=l2,
                                   name='BNeck2_0')
        self.BNeck2_1 = BottleNeck(output_filters=128, l2=l2, name='BNeck2_1')
        self.BNeck2_2 = BottleNeck(output_filters=128,
                                   dilation_rate=(2, 2),
                                   l2=l2,
                                   name='BNeck2_2')
        self.BNeck2_3 = BottleNeck(output_filters=128,
                                   kernel_size=(5, 1),
                                   l2=l2,
                                   name='BNeck2_3')
        self.BNeck2_4 = BottleNeck(output_filters=128,
                                   dilation_rate=(4, 4),
                                   l2=l2,
                                   name='BNeck2_4')
        self.BNeck2_5 = BottleNeck(output_filters=128, l2=l2, name='BNeck2_5')
        self.BNeck2_6 = BottleNeck(output_filters=128,
                                   dilation_rate=(8, 8),
                                   l2=l2,
                                   name='BNeck2_6')
        self.BNeck2_7 = BottleNeck(output_filters=128,
                                   kernel_size=(5, 1),
                                   l2=l2,
                                   name='BNeck2_7')
        self.BNeck2_8 = BottleNeck(output_filters=128,
                                   dilation_rate=(16, 16),
                                   l2=l2,
                                   name='BNeck2_8')

        # # third block of bottlenecks
        self.BNeck3_1 = BottleNeck(output_filters=128, l2=l2, name='BNeck3_1')
        self.BNeck3_2 = BottleNeck(output_filters=128,
                                   dilation_rate=(2, 2),
                                   l2=l2,
                                   name='BNeck3_2')
        self.BNeck3_3 = BottleNeck(output_filters=128,
                                   kernel_size=(5, 1),
                                   l2=l2,
                                   name='BNeck3_3')
        self.BNeck3_4 = BottleNeck(output_filters=128,
                                   dilation_rate=(4, 4),
                                   l2=l2,
                                   name='BNeck3_4')
        self.BNeck3_5 = BottleNeck(output_filters=128, l2=l2, name='BNeck3_5')
        self.BNeck3_6 = BottleNeck(output_filters=128,
                                   dilation_rate=(8, 8),
                                   l2=l2,
                                   name='BNeck3_6')
        self.BNeck3_7 = BottleNeck(output_filters=128,
                                   kernel_size=(5, 1),
                                   l2=l2,
                                   name='BNeck3_7')
        self.BNeck3_8 = BottleNeck(output_filters=128,
                                   dilation_rate=(16, 16),
                                   l2=l2,
                                   name='BNeck3_8')

        # project the encoder output to the number of classes
        # to get the output of the encoder head
        self.ConvEncOut = tf.keras.layers.Conv2D(
            self.C,
            kernel_size=[1, 1],
            padding='valid',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            name='EncOut')

        # fourth block of bottlenecks
        self.BNeck4_0 = BottleDeck(output_filters=64,
                                   internal_comp_ratio=2,
                                   l2=l2,
                                   name='BNeck4_0')
        self.BNeck4_1 = BottleNeck(output_filters=64, l2=l2, name='BNeck4_1')
        self.BNeck4_2 = BottleNeck(output_filters=64, l2=l2, name='BNeck4_2')

        # fourth block of bottlenecks
        self.BNeck5_0 = BottleDeck(output_filters=16,
                                   internal_comp_ratio=2,
                                   l2=l2,
                                   name='BNeck5_0')
        self.BNeck5_1 = BottleNeck(output_filters=16, l2=l2, name='BNeck5_1')

        # Final ConvTranspose Layer
        self.FullConv = tf.keras.layers.Conv2DTranspose(
            self.C,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            name='DecOut')

    def call(self, inputs):

        # init block
        x = self.InitBlock(inputs)

        # first block of bottlenecks - downsampling
        x, x_argmax1_0, x_upsample1_0 = self.BNeck1_0(x)  # downsample
        x = self.BNeck1_1(x)
        x = self.BNeck1_2(x)
        x = self.BNeck1_3(x)
        x = self.BNeck1_4(x)

        # second block of bottlenecks - downsampling
        x, x_argmax2_0, x_upsample2_0 = self.BNeck2_0(x)  # downsample
        x = self.BNeck2_1(x)
        x = self.BNeck2_2(x)
        x = self.BNeck2_3(x)
        x = self.BNeck2_4(x)
        x = self.BNeck2_5(x)
        x = self.BNeck2_6(x)
        x = self.BNeck2_7(x)
        x = self.BNeck2_8(x)

        # third block of bottlenecks
        x = self.BNeck3_1(x)
        x = self.BNeck3_2(x)
        x = self.BNeck3_3(x)
        x = self.BNeck3_4(x)
        x = self.BNeck3_5(x)
        x = self.BNeck3_6(x)
        x = self.BNeck3_7(x)
        x = self.BNeck3_8(x)

        EncOut = self.ConvEncOut(x)

        # fourth block of bottlenecks - upsampling
        x = self.BNeck4_0(x, x_argmax2_0, x_upsample2_0)
        x = self.BNeck4_1(x)
        x = self.BNeck4_2(x)

        # fifth block of bottlenecks - upsampling
        x = self.BNeck5_0(x, x_argmax1_0, x_upsample1_0)
        x = self.BNeck5_1(x)

        # final full conv to the segmentation maps
        DecOut = self.FullConv(x)

        # what i return, depends on the multiobjective flag
        if self.MultiObjective:
            return EncOut, DecOut
        else:
            return DecOut
