import os
import tensorflow as tf
from tensorflow.python.framework import ops

sbnet_module = tf.load_op_library(
  os.path.join(os.path.dirname(__file__),
               'sbnet_tensorflow/sbnet_ops/libsbnet.so'))
reduce_mask = sbnet_module.reduce_mask
sparse_gather = sbnet_module.sparse_gather
sparse_scatter = sbnet_module.sparse_scatter
sparse_scatter_var = sbnet_module.sparse_scatter_var

# Gradients registration.
@ops.RegisterGradient("SparseGather")
def _sparse_gather_grad(op, grad):
  # x is shaped like full tensor [NHWC]
  # grad is shaped as gathered blocks [Nblocks*BH*BW*C]
  x = op.inputs[0]
  binCounts = op.inputs[1]
  activeBlockIndices = op.inputs[2]
  bsize = op.inputs[3]
  bstride = op.inputs[4]
  boffset = op.inputs[5]
  transpose = op.get_attr("transpose")

  # if scatter is overlapping then gradient should still work
  # because we are overwriting the same values
  # compute dOutput/dx
  result = sbnet_module.sparse_scatter(
    grad,
    binCounts,
    activeBlockIndices,
    tf.zeros_like(x),    # output base tensor to add on top of
    dynamic_bsize=bsize,
    dynamic_bstride=bstride,
    dynamic_boffset=boffset,
    add=True,
    transpose=transpose,
    atomic=True)

  return [result, None, None, None, None, None]    # no gradients wrt indices or block params


@ops.RegisterGradient("SparseScatter")
def _sparse_scatter_grad(op, grad):
  # x is shaped like blocked tensor of gathered blocks [Nblocks*BH*BW*C]
  # grad is shaped as output tensor [NHWC]
  blocksX = op.inputs[0]
  binCounts = op.inputs[1]
  activeBlockIndices = op.inputs[2]
  ybase = op.inputs[3]
  bsize = op.inputs[4]
  bstride = op.inputs[5]
  boffset = op.inputs[6]
  doAdd = op.get_attr("add")
  
  dout_dx = sbnet_module.sparse_gather(
    grad,
    binCounts,
    activeBlockIndices,
    dynamic_bsize=bsize,
    dynamic_bstride=bstride,
    dynamic_boffset=boffset)
  
  # return a list of gradients of output with respect to each input
  if not doAdd:
    # scatter blocks of zeroes over a base tensor of ones to compute a stamp-out gradient mask for dy_dybase
    stamp_out_blocks = sbnet_module.sparse_scatter(
      tf.zeros_like(blocksX),
      binCounts,
      activeBlockIndices,
      tf.ones_like(grad),
      dynamic_bsize=bsize,
      dynamic_bstride=bstride,
      dynamic_boffset=boffset,
      add=False)
    dy_dybase = grad * stamp_out_blocks
    return [dout_dx, None, None, dy_dybase, None, None, None]
  else:
    # d(x+ybase)/dybase = 1, so just pass back grad as dout_dybase
    return [dout_dx, None, None, grad, None, None, None]
  
