# Targeted Attacks and Defenses in Federated Learning

Note: This directory is a work-in-progress.

## Overview

This directory contains source code for
[TFF](https://www.tensorflow.org/federated) implementation of targeted attacks
and defenses considered in
[Can You Really Back door Federated Learning?](https://arxiv.org/abs/1911.07963)
The TFF implementation of basic federated learning is based on
[Simple_Fedavg](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py).
The attack model we are considering is primarily based on the model replacement
attacks proposed in the following two papers:

*   [Analyzing Federated Learning Through an Adversarial Lens](https://arxiv.org/abs/1811.12470).
*   [How To Back door Federated Learning?](https://arxiv.org/abs/1807.00459)

## Implementation and Sample Script

For implementation of unconstrainted and norm bounded attack, please see
`ClientExplicitBoost` and `ClientProjectBoost` in `attacked_fedavg.py`. For the
implementation of various defenses, please see `aggregate_fn.py`. For a sample
script of how to use the package, please see `emnist_with_targeted_attack.py`.

## Dependencies

*   [tensorboard](https://pypi.org/project/tensorboard/)
*   [SciPY](https://www.scipy.org/)
