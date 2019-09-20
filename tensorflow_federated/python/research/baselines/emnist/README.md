# Federated EMNIST Baseline Experiments

Note: This directory is a work-in-progress.

## Overview

This directory contains models and experiment scripts for training baseline
federated and non-federated models on the
[Federated EMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)
dataset. The model architectures and follow those from the paper
[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629),
however, since we use the Federated EMNIST dataset (with its natural user
partitioning), rather than MNIST (with either a synthetic non-IID or shuffled
IID partitioning), the results are not directly comparable.

## Citation

If you use these baselines and need to cite this work, please use:

```
@misc{mcmahan19emnist_baseline,
    author       = {H. Brendan McMahan},
    title        = {{Federated EMNIST Baseline Training}},
    year         = 2019,
    url          =  {https://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/research/baselines/emnist}
    }
```

## Dependencies

*   [tensorboard](https://pypi.org/project/tensorboard/)
