# -*- coding: utf-8 -*-
"""## Libraries"""

# numpy
import numpy as np

# enet
import sys
from layers import BottleDeck, BottleNeck, InitBlock
from models import EnetEncoder

# tensorflow
import tensorflow as tf

# create random input tensor
m = 32
h = 512
w = 512
c = 3
m_tf = tf.cast(np.random.rand(m, h, w, 3), tf.float32)

# instantiate model
Enet = EnetEncoder(C=10, dropout=0.0)
x = Enet(m_tf)
print(x.shape)
