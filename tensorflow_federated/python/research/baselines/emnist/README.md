# Federated EMNIST Baseline Experiments

Note: This directory is a work-in-progress.

## Overview

This directory contains multiple model architectures and experiment scripts for
training baseline federated and non-federated models on the
[Federated EMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)
dataset.

NOTE: The model architecture from the paper
[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629),
is exactly reproduced (`models.original_fedavg_cnn_model`). However, since we
use the Federated EMNIST dataset (with its natural user partitioning), rather
than the MNIST partitioning (either a synthetic non-IID or shuffled IID) from
the original paper, the results are not directly comparable.

## Citation

If you use these baselines and need to cite this work, please use:

```
@misc{mcmahan19emnist_baseline,
    author       = {H. Brendan McMahan and Jakub Kone{\v{c}}n{\'y}},
    title        = {{Federated EMNIST Baseline Training}},
    year         = 2019,
    url          = {https://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/research/baselines/emnist}
    }
```

## Dependencies

*   [tensorboard](https://pypi.org/project/tensorboard/)
