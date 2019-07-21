import tensorflow as tf

sbnet_module = tf.load_op_library('sbnet_tensorflow/sbnet_ops/libsnet.so')
from sbnet_module import reduce_mask, sparse_gather, sparse_scatter
