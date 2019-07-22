import os
import tensorflow as tf

sbnet_module = tf.load_op_library(
  os.path.join(os.path.dirname(__file__),
               'sbnet_tensorflow/sbnet_ops/libsbnet.so'))
reduce_mask = sbnet_module.reduce_mask
sparse_gather = sbnet_module.sparse_gather
sparse_scatter = sbnet_module.sparse_scatter
sparse_scatter_var = sbnet_module.sparse_scatter_var
