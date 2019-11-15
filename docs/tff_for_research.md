# Using TFF for Federated Learning Research

**Note: This page is currently being populated**

## Overview

TFF is an extensible, powerful framework for conducting federated learning (FL)
research by simulating federated computations on realistic proxy datasets. This
page describes the main concepts and components that are relevant for research
simulations, as well as detailed guidance for conducting different kinds of
research in TFF.

## The typical structure of research code in TFF

A research FL simulation implemented in TFF typically consists of three main
types of logic.

1.  Individual pieces of TensorFlow code, typically `tf.function`s, that
    encapsulate logic that runs in a single location (e.g., on clients or on a
    server). This code is typically written and tested without any `tff.*`
    references, and can be re-used outside of TFF. For example, the
    [local training loop in Federated Averaging](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py#L126)
    is implemented at this level.

1.  TensorFlow Federated orchestration logic, which binds together the
    individual `tf.function`s from 1. by wrapping them as `tff.tf_computation`s
    and then orchestrating them using abstractions like
    `tff.federated_broadcast` and `tff.federated_mean` inside a
    `tff.federated_comutation`. For example, this
    [orchestration for Federated Averaging](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py#L272).

1.  An outer driver script that simulates the control logic of a production FL
    system, selecting simulated clients from a dataset and then executing
    federated comptuations defined in 2. on those clients. For example,
    [a Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/baselines/emnist/run_federated.py#L70).

## Federated learning datasets

TensorFlow federated
[hosts multiple datasets](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/da tasets)
that are representative of the characteristics of real-world problems that could
be solved with federated learning. Datasets include:

*   [**StackOverflow**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data)
    A realistic text dataset for language modeling or supervised learning tasks,
    with 342,477 unique users with 135,818,730 examples (sentences) in the
    training set.

*   [**Federated EMNIST**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)
    A federated pre-processing of the EMNIST character and digit dataset, where
    each client corresponds to a different writer. The full train set contains
    3400 users with 671,585 examples from 62 labels.

*   [**Shakespeare**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data)
    A smaller char-level text dataset based on the complete works of William
    Shakespeare. The data set consists of 715 users (characters of Shakespeare
    plays), where each example corresponds to a contiguous set of lines spoken
    by the character in a given play.

## High performance simulations

<!-- TODO(b/143692319): Referent discussion in the in our paper. -->

While the wall-clock time of an FL _simulation_ is not a relevant metric for
evaluating algorithms (as simulation hardware isn't representative of real FL
deployment environments), being able to run FL simulations quickly is critical
for research productivity. Hence, TFF has invested heavily in providing
high-performance single and multi-machine runtimes. Documentation is under
development, but for now see the
[High-performance simulations with TFF](https://github.com/tensorflow/federated/blob/master/docs/tutorials/simulations.ipynb)
tutorial as well as instructions on
[setting up simulations with TFF on GCP](https://github.com/tensorflow/federated/blob/master/docs/gcp_setup.md).
For fast-single machine experiments, use

```python
tff.framework.set_default_executor(tff.framework.create_local_executor())
```

This should become the default soon.

## TFF for different research areas

### Federated optimization algorithms

<!-- TODO(b/144510813): Change references to the appropriate parts of the new simple fedavg once it is done. -->

Research on federated optimization algorithms can be done in different ways in
TFF, depending on the desired level of customization.

For simple variations of the
[Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm:

*   Custom client optimizers can easily be experimented with by compiling client
    models with the appropriate `tf.keras.optimizers` class, as in this
    [federated EMNIST experiment](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/baselines/emnist/run_federated.py#L65).

*   Experiments that want to use custom `tf.keras.optimizers` for the
    application of updates on the server can do this by passing the desired
    server optimizer to the federated training loop, as in this
    [federated EMNIST experiment](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/baselines/emnist/run_federated.py#L127-L136).

To implement more complicated federated optimization algorithms, you may need
customize your federated training loop in order to gain more control over the
orchestration and optimization logic of the experiment. Again,
[`simple_fedavg`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py)
may be a good place to start. For example, you could change the
[client update](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py#L129-L163)
function to implement a custom local training procedure, modify the
`tff.federated_computation` that controls the
[orchestration](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py#L274-L303)
to change what is broadcast from the server to client and what is aggregated
back, and alter
[the outer loop](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/utils/training_loops.py#L71-L83)
of the experiment to use different behaviors across rounds.

### Model and update compression

### Meta-learning and multi-task learning

### Differential privacy

### Robustness and attacks
