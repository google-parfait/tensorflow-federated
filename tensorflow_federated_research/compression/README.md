# Federated EMNIST Baseline Experiments with compression.

Note: This directory is a work-in-progress.

## Summary

This directory provides an end-to-end example of compressed model broadcasts and
compressed aggregated model updates for federated training (with the Federated
Averaging algorithm in particular).

*   [run_experiment.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated_research/compression/run_experiment.py)
    demonstrates how encoders implemented in the
    [`tensor_encoding`](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding)
    API can be quickly integrated into an existing federated training loop.
*   If you are interested in implementing your own compression algorithms for
    use in TFF, see
    [sparsity.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated_research/compression/sparsity.py)
    for an example of implementing a custom compression algorithm. See also
    [TFF for research](https://github.com/tensorflow/federated/blob/master/docs/tff_for_research.md).

The specific algorithm implemented here applies the following transform to every
model variable of more than `10,000` elements. It first applies uniform
quantization to a speficied number of bits, followed by simple concatenation of
the bits representing the quantized values into an `int32` tensor.

Note that the tooling is not limited to the specific ideas outlined above.
Rather, the use of `tensor_encoding` API enables the use of any compression
algorithm to be provided via the
`tff.learning.framework.build_encoded_broadcast_from_model` and
`tff.learning.framework.build_encoded_mean_from_model` utilities.
