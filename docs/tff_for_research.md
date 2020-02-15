<!-- Note that some section headings are used as deep links into the document.
     If you update those section headings, please make sure you also update
     any links to the section. -->

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
    [client training loop in Federated Averaging](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py#L99)
    is implemented at this level.

1.  TensorFlow Federated orchestration logic, which binds together the
    individual `tf.function`s from 1. by wrapping them as `tff.tf_computation`s
    and then orchestrating them using abstractions like
    `tff.federated_broadcast` and `tff.federated_mean` inside a
    `tff.federated_computation`. See, for example, this
    [orchestration for Federated Averaging](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py#L200).

1.  An outer driver script that simulates the control logic of a production FL
    system, selecting simulated clients from a dataset and then executing
    federated comptuations defined in 2. on those clients. For example,
    [a Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg.py#L161-L171).

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
The high-performance TFF runtime is enabled by default.

## TFF for different research areas

### Federated optimization algorithms

<!-- TODO(b/144510813): Change references to the appropriate parts of the new simple fedavg once it is done. -->

Research on federated optimization algorithms can be done in different ways in
TFF, depending on the desired level of customization.

A minimal implementation of the
[Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided
[here](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py),
along with an example
[federated EMNIST experiment](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg.py).
The training example can easily be adapted for simple experiment changes:

*   Custom client optimizers can easily be experimented with by passing them
    directly to the federated averaging algorithm, as in the
    [federated EMNIST experiment](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg.py#L137-L138).
    Custom server optimizers can be experimented with in the
    [same way](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg.py#L133-L134).
*   Custom models and loss functions can be immediately used by implementing
    them using [Keras](https://www.tensorflow.org/guide/keras), and simply
    wrapping them as a `tff.learning.Model`, as in the
    [EMNIST training example](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg.py#L119-L130).

To implement more complicated federated optimization algorithms, you may need
customize your federated training loop in order to gain more control over the
orchestration and optimization logic of the experiment. Again,
[`simple_fedavg`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py)
may be a good place to start. For example, you could change the
[client update](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py#L98-L134)
function to implement a custom local training procedure, modify the
`tff.federated_computation` that controls the
[orchestration](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py#L198-L227)
to change what is broadcast from the server to client and what is aggregated
back, and alter
[the server update](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg.py#L65-L95)
to change how the server model is learned from the client updates.

### Model and update compression

TFF uses the
[tensor_encoding](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding)
API to enable lossy compression algorithms to reduce communicatation costs
between the server and clients. For an example of training with server-to-client
and client-to-server
[compression using Federated Averaging](https://arxiv.org/abs/1812.07210)
algorithm, see
[this experiment](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/compression/run_experiment.py).

To implement a custom compression algorithm and apply it to the training loop,
you can:

1.  Implement a new compression algorithm as a subclass of
    [`EncodingStageInterface`](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L75)
    or its more general variant,
    [`AdaptiveEncodingStageInterface`](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L274)
    following
    [this example](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/compression/sparsity.py).
1.  Construct your new
    [`Encoder`](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/core_encoder.py#L38)
    and specialize it for
    [model broadcast](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/compression/run_experiment.py#L118)
    or
    [model update averaging](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/compression/run_experiment.py#L144).
1.  Use those objects to build the entire
    [training computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/compression/run_experiment.py#L247).

### Differential privacy

TFF is interoperable with the
[TensorFlow Privacy](https://github.com/tensorflow/privacy) library to enable
research in new algorithms for federated training of models with differential
privacy. For an example of training with DP using
[the basic DP-FedAvg algorithm](https://arxiv.org/abs/1710.06963) and
[extensions](https://arxiv.org/abs/1812.06210), see
[this experiment driver](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/baselines/emnist/run_dp_experiment.py).

If you want to implement a custom DP algorithm and apply it to the aggregate
updates of federated averaging, you can:

1.  Implement a new DP mean algorithm as a subclass of
    [`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54),
1.  construct your new `DPQuery` similarly to the way standard `DPQueries` are
    constructed
    [here](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/utils/differential_privacy.py#L37-L134),
1.  and pass your query instance into `tff.utils.build_dp_aggregate()` similarly
    to
    [`run_dp_experiment`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/baselines/emnist/run_dp_experiment.py#L158).

Federated GANs (described [below](#generative_adversarial_networks)) are another
example of a TFF project implementing user-level differential privacy (e.g.,
[here in code](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L293)).

### Robustness and attacks

TFF can also be used to simulate the targeted attacks on federated learning
systems and differential privacy based defenses considered in
*[Can You Really Back door Federated Learning?](https://arxiv.org/abs/1911.07963)*.
This is done by building an iterative process with potentially malicious clients
(see
[`build_federated_averaging_process_attacked`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/attacked_fedavg.py#L412)).
The
[targeted_attack](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack)
directory contains more details.

*   New attacking algorithms can be implemented by writing a client update
    function which is a Tensorflow function, see
    [`ClientProjectBoost`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/attacked_fedavg.py#L460)
    for an example.
*   New defenses can be implemented by customizing
    ['tff.utils.StatefulAggregateFn'](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103)
    which aggregates client outputs to get a global update.

For an example script for simulation, see
[`emnist_with_targeted_attack.py`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/emnist_with_targeted_attack.py).

### Generative Adversarial Networks

GANs make for an interesting
[federated orchestration pattern](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L266-L316)
that looks a little different than standard Federated Averaging. They involve
two distinct networks (the generator and the discriminator) each trained with
their own optimization step.

TFF can be used for research on federated training of GANs. For example, the
DP-FedAvg-GAN algorithm presented in
[recent work](https://arxiv.org/abs/1911.06679) is
[implemented in TFF](https://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/research/gans).
This work demonstrates the effectiveness of combining federated learning,
generative models, and [differential privacy](#differential_privacy).
