# Using TFF for Federated Learning Research

<!-- Note that some section headings are used as deep links into the document.
     If you update those section headings, please make sure you also update
     any links to the section. -->

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
    [client training loop in Federated Averaging](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py#L184-L222)
    is implemented at this level.

1.  TensorFlow Federated orchestration logic, which binds together the
    individual `tf.function`s from 1. by wrapping them as `tff.tf_computation`s
    and then orchestrating them using abstractions like
    `tff.federated_broadcast` and `tff.federated_mean` inside a
    `tff.federated_computation`. See, for example, this
    [orchestration for Federated Averaging](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tff.py#L112-L140).

1.  An outer driver script that simulates the control logic of a production FL
    system, selecting simulated clients from a dataset and then executing
    federated comptuations defined in 2. on those clients. For example,
    [a Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py).

## Federated learning datasets

TensorFlow federated
[hosts multiple datasets](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets)
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

*   [**CIFAR-100**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)
    A federated partitioning of the CIFAR-100 dataset across 500 training
    clients and 100 test clients. Each client has 100 unique examples. The
    partitioning is done in a way to create more realistic heterogeneity between
    clients. For more details, see the
    [API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data).

## High performance simulations

While the wall-clock time of an FL _simulation_ is not a relevant metric for
evaluating algorithms (as simulation hardware isn't representative of real FL
deployment environments), being able to run FL simulations quickly is critical
for research productivity. Hence, TFF has invested heavily in providing
high-performance single and multi-machine runtimes. Documentation is under
development, but for now see the
[High-performance simulations with TFF](https://www.tensorflow.org/federated/tutorials/simulations)
tutorial as well as instructions on
[setting up simulations with TFF on GCP](https://www.tensorflow.org/federated/gcp_setup).
The high-performance TFF runtime is enabled by default.

## TFF for different research areas

### Federated optimization algorithms

Research on federated optimization algorithms can be done in different ways in
TFF, depending on the desired level of customization.

A minimal stand-alone implementation of the
[Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided
[here](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg).
The code includes
[TF functions](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py)
for local computation,
[TFF computations](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tff.py)
for orchestration, and a
[driver script](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py)
on the EMNIST dataset as an example. These files can easily be adapted for
customized applciations and algorithmic changes following detailed instructions
in the
[README](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/README.md).

A more general implementation of Federated Averaging can be found
[here](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/optimization/shared/fed_avg_schedule.py).
This implementation allows for more sophisticated optimization techniques,
including learning rate scheduling and the use of different optimizers on both
the server and client. Code that applies this generalized Federated Averaging to
various tasks and federated datasets can be found
[here](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/optimization).

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
[this experiment driver](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/differential_privacy/stackoverflow/run_federated.py).

If you want to implement a custom DP algorithm and apply it to the aggregate
updates of federated averaging, you can:

1.  Implement a new DP mean algorithm as a subclass of
    [`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54),
1.  construct your new `DPQuery` similarly to the way standard `DPQueries` are
    constructed
    [here](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/utils/differential_privacy.py#L37-L134),
1.  and pass your query instance into `tff.utils.build_dp_aggregate()` similarly
    to
    [`run_federated`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/differential_privacy/stackoverflow/run_federated.py#L199).

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

### Personalization

Personalization in the setting of federated learning is an active research area.
The goal of personalization is to provide different inference models to
different users. There are potentially different approaches to this problem.

One approach is to let each client fine-tune a single global model (trained
using federated learning) with their local data. This approach has connections
to meta-learning, see, e.g., [this paper](https://arxiv.org/abs/1909.12488). An
example of this approach is given in
[`run_emnist_p13n.py`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/personalization/run_emnist_p13n.py).
To explore and compare different fine-tuning strategies, you can:

*   Implement a `tf.function` that starts from an initial model, trains and
    evaluates a personalized model using each client's local datasets. An
    example is
    [`build_personalize_fn`](https://github.com/tensorflow/federated/blob/e157d7c2e2150f9e63c5c07a9ca17d741de510df/tensorflow_federated/python/research/personalization/p13n_utils.py#L36).

*   Define an `OrderedDict` that maps strategy names to the corresponding
    `tf.function`s (see
    [`personalize_fn_dict`](https://github.com/tensorflow/federated/blob/e157d7c2e2150f9e63c5c07a9ca17d741de510df/tensorflow_federated/python/research/personalization/run_emnist_p13n.py#L131)
    for an example), and construct the
    [TFF computation](https://github.com/tensorflow/federated/blob/e157d7c2e2150f9e63c5c07a9ca17d741de510df/tensorflow_federated/python/research/personalization/run_emnist_p13n.py#L152)
    to evaluate those strategies.
