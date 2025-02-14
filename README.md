# Sparse Blocks Network (SBNet) (Fork)

This repository is forked from
[uber-research/sbnet](https://github.com/uber-research/sbnet). In addition to
the original README, included below, I've included a few notes about
installation on my particular system, for future/external reference.

## Installation

I work remotely on a cluster using Slurm, so configuring Tensorflow, Cuda, and
SBNet to work together was less than ideal. I have no root priveledges,
naturally. I used Cuda 9.0, Tensorflow 1.13.1, and a Tesla K80 graphics
card. These are the steps I took:
1. Ensure the Hardware requirements section is followed. My `Tesla K80` graphics
   card has code 37. In `sbnet_tensorflow/sbnet_ops/Makefile`, I removed lines
   31-34 and replaced them with ``` -gencode arch=compute_37,code=sm_37 \ ```
   which has the correct code. Without this change, I got errors like
```
/sbnet_ops/libsbnet.so: undefined symbol: _ZN10tensorflow8internal21CheckOpMessageBuilder9NewStringEv
```
   on running `make test`.
2. In that same file, line 17 locates the installation of CUDA. Since CUDA was
   installed as `path/to/cuda-9.0-el7-x86_64` rather than just cuda, a lot of `#include`
   statements wouldn't work. Therefore I symlinked: `path/to/cuda-9.0-e17-x86_64`
   to `~/local/cuda` with
```
ln -s path/to/cuda-9.0-e17-x86_64` /home/killeen/local/cuda`
```
   in my user directory. I then changed line 17 in
   `sbnet_tensorflow/sbnet_ops/Makefile` to
```
CUDA_INC = /home/killeen/local/cuda/include
```
3. Line 18 in the same file is confusing. Basically tensorflow has the line
```
#include cuda/includa/cuda.h
```
   in some places, so you also have to add the parent directory of your cuda
   installation to the nvcc flags. This is accomplished by changing `LOCAL_INC`
   on line 18, e.g.:
```
LOCAL_INC = /home/killeen/local
```
4. Similarly, `CUDA_LIB` on line 19 needs to be updated to point to the proper
   `lib64`. You get the idea.
5. Finally, I added `-DNDEBUG` to the nvcc flags on line 32. This was needed to
   get rid of this error during compile time:
```
error: constexpr function return is non-constant
```

# Sparse Bloocks Network (SBNet) (original)

This repository releases code for our paper [*SBNet: Sparse Blocks Network for Fast Inference*](https://arxiv.org/abs/1801.02108). Please refer to our [blog post](https://eng.uber.com/sbnet) for more context.
Note that benchmarking in the paper was performed with an older version of this repo using TensorFlow 1.2, cuDNN 6.1 and commit cf8ea06.

This repository contains 
1. a TensorFlow custom operations library that implements SBNet,
2. a Python implementation of sparse ResNet blocks, and
3. a benchmark for performance comparison with [Submanifold Sparse Convolutional Networks](https://arxiv.org/abs/1706.01307).

## Prerequisites

Installation was tested under Ubuntu 14.04 and 16.04 with TensorFlow 1.8, CUDA 9.0 and cuDNN 7.1.

## Hardware requirements

Code was tested on and compiled for NVIDIA CUDA 6.1, 6.0, 5.2 and 7.0
architectures (Titan XP, GTX 1080Ti, GTX 1080, P100, V100, TitanV, and most
Maxwell cards).  To compile for an older architecture please modify the Makefile
and add the corresponding line, such as `-gencode arch=compute_50,code=sm_50`
for older cards such as laptop Maxwell.  Please refer to [CUDA
Wikipedia](https://en.wikipedia.org/wiki/CUDA) page to lookup the architecture
code for your graphics card.


## Setup

To build a release version of the library, run

`cd sbnet_tensorflow/sbnet_ops && make`

To run tests:

`cd sbnet_tensorflow/sbnet_ops && make test`

The library will be built in sbnet_tensorflow/sbnet_ops/build/libsbnet.so and symlinked to sbnet_tensorflow/sbnet_ops/libsbnet.so.
To import the library into your TensorFlow Python code use the following command:

```
sbnet_module = tf.load_op_library('path_to_library/libsbnet.so')
```

The following Tensorflow ops are implemented in the op library:

```sbnet_module.reduce_mask```

```sbnet_module.sparse_gather```

```sbnet_module.sparse_scatter```


`reduce_mask` op converts a dense mask to a list of active block indices.

In the following snippet the mask is expected to be a tensor of dimensions `[N,H,W,1]`:

```
    indices = sbnet_module.reduce_mask(
        mask, tf.constant([BCH, BCW], dtype=tf.int32),
        bsize=[BSZH, BSZW],
        boffset=[BOFFSH, BOFFSW],
        bstride=[BSTRH, BSTRW],
        tol=0.5, # pooling threshold to consider a block as active
        avgpool=True) # max pooling by default
```

[BCH, BCW] are block counts in height and width dimensions.
[BSZH, BSZW], [BOFFSH, BOFSFW] and [BSTRH, BSTRW] are block sizes, offsets and strides in H and W dimensions.
`reduce_mask` performs a combined max pooling (or average pooling) operation localized to each block followed by generating
a list of triples of indices `[(ni, hi, wi)]` for blocks where either max or average pooling value exceeds specified tolerance `tol`.
In numpy terms each block is defined as a slice from the input mask of dimensions `[N,H,W,1]`, with following dimensions:
`[ni, BOFFSH+BSTRH*hi : BOFFSH+BSTRH*hi+BSZH, BOFFSW+BSTRW*wi : BOFFSW+BSTRW*wi+BSZW, :]`.

The resulting list of indices can then be passed to two other operations: `sbnet_module.sparse_scatter` and `sbnet_module.sparse_gather`.

The following snippets illustrate the use of these operations:
```
    blockStack = sbnet_module.sparse_gather(
        x,
        indices.bin_counts,
        indices.active_block_indices,
        bsize=[BSZH, BSZW], # block size
        boffset=[BOFFSH, BOFFSW], # block offset
        bstride=[BSTRH, BSTRW], # block stride
        transpose=do_transpose)
```

This operation will use the indices generated by reduce_mask and slice out tensors of channel depth C out of input tensor `x` of dimensions `[N,H,W,C]` as illustrated in the following pseudo-code snippet:

```
    for (ni, hi, wi) in indices.active_block_indices:
        channel_slice = x[ni, BOFFSH+BSTRH*hi : BOFFSH+BSTRH*hi+BSZH, BOFFSW+BSTRW*wi : BOFFSW+BSTRW*wi+BSZW, :]
        blockStack[ni, :, :, :] = channel_slice
```

If `do_transpose` is true, a fused transpose operation will also be performed and the resulting tensor will have dimensions `[nBlocks, C, BSZH, BSZW]`.
Any out-of-range values will be padded with zeroes.

The inverse operation is `sbnet_module.sparse_scatter`. The following snippet illustrates it's use:

```
    y = sbnet_module.sparse_scatter(
        blockStack,
        indices.bin_counts,
        indices.active_block_indices,
        x, # base tensor to copy to output and overwrite on top of
        bsize=[BSZH, BSZW],
        boffset=[BOFFSH, BOFFSW],
        bstride=[BSTRH, BSTRW],
        add=do_add,
        atomic=False, # use atomic or regular adds
        transpose=do_transpose)
```

Note that due to a limitation of TensorFlow API an intermediate tensor cannot be modified in place unless it's specified to be a tf.Variable.
This necessitates creating an intermediate tensor inside the op and performing a copy which has negative implications for performance.
So we created a second version of the op `sbnet_module.sparse_scatter_var` that expects x to be a `tf.Variable` and modifies it in place.
Using `sparse_scatter_var` is strongly recommended for maximum performance.

The effect of this operation is opposite to `sparse_gather` - the input blocks will be written on top of base tensor x, or added to it's contents if `do_add` is True.
The following pseudo-code snippet illustrates the semantics of `sparse_scatter`:

```
    for (ni, hi, wi) in indices.active_block_indices:
        if do_add:
            x[ni, BOFFSH+BSTRH*hi : BOFFSH+BSTRH*hi+BSZH, BOFFSW+BSTRW*wi : BOFFSW+BSTRW*wi+BSZW, :]\
                += blockStack[ni, :, :, :]
        else:
            x[ni, BOFFSH+BSTRH*hi : BOFFSH+BSTRH*hi+BSZH, BOFFSW+BSTRW*wi : BOFFSW+BSTRW*wi+BSZW, :]\
                = blockStack[ni, :, :, :]
```

So the blocks are 'put back in place', however the sizes and strides can be different from those passed to sparse_gather. This enables implementation of sparse ResNet blocks where output resolution is reduced
after a 'VALID' convolution. Similar to `sparse_gather`, if `do_transpose` is true, a fused transpose operation will also be performed by sparse_scatter, permuting the input `[N,C,H,W]` dimensions to `[N,H,W,C]` in the output.
Typically the block size for a 'VALID' convolution is reduced by 2 in each spatial dimension for each 3x3 convolution, thus creating non-overlapping outputs.
Note that even though currently we support atomic adds in scatter with add=True, the gradient is not implemented at this time if overlapping scatters are used the forward pass. 

## Benchmarks and tests

Benchmarks for SBNet are located in sbnet_tensorflow/benchmarks/ subdirectory.

To run benchmarks execute:

```
cd sbnet_tensorflow/benchmarks && ./run_all_behchmarks.bash
```

Note that we average over a number of runs and test many permutations of parameters so this may take about 20 minutes (on a Titan XP) and will produce a number of .csv files in your /home/user/ directory.
We benchmark individual sparse convolutions and entire sparse ResNet blocks on a synthetic mask with variable sparsity.

To run unit tests execute:
```
cd sbnet_tensorflow/sbnet_ops && make tests
```


## Submanifold Sparse Convolutional Networks Benchmark

For comparison we implemented benchmarking code for [Submanifold Sparse Convolutional Networks](https://github.com/facebookresearch/SparseConvNet).
Running this benchmark requires Submanifold Sparse Convolutions python package to be installed:
``` 
git clone https://github.com/facebookresearch/SparseConvNet.git 
```
Follow the setup instructions in SparseConvNet repo.

Code integration with Submanifold Sparse Convolutions was tested with git sha 609224df3c0e42b8a1dd4073aaa56fab805096c6. To reset the repo to this sha use the following sequence of commands:
```
cd SparseConvNet
git checkout 609224df3c0e42b8a1dd4073aaa56fab805096c6
```

The benchmark code is located in sbnet_tensorflow/benchmark_submanifold directory.


## Other notes

Current code is not tuned for performance with non-square block sizes and has specialized implementations for a specific list of block sizes. This includes square blocks of sizes 1 to 34 and a few others. To achieve maximum performance for these sizes you would need to add your custom template instantiations by modifying SIZE_TEMPLATES macro in `sparse_gather.cu`.


## Contributing to this repository

For now, we do not accept pull request to this repo, as we are currently setting up automated CI.
If you would like to contribute to this repository, feel free create a GitHub issue.

## Citation

If you use our code, please consider cite the following:
M. Ren, A. Pokrovsky, B. Yang, and R. Urtasun. SBNet: Sparse Blocks Network for Fast Inference. 
*CoRR*, abs/1801.02108, 2018.

```
@article{ren18sbnet,
  author    = {Mengye Ren and 
               Andrei Pokrovsky and
               Bin Yang and
               Raquel Urtasun},
  title     = {SBNet: Sparse Blocks Network for Fast Inference},
  journal   = {CoRR},
  volume    = {abs/1801.02108},
  year      = {2018},
}
```
