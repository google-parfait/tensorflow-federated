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
    [client training loop in Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222)
    is implemented at this level.

1.  TensorFlow Federated orchestration logic, which binds together the
    individual `tf.function`s from 1. by wrapping them as `tff.tf_computation`s
    and then orchestrating them using abstractions like
    `tff.federated_broadcast` and `tff.federated_mean` inside a
    `tff.federated_computation`. See, for example, this
    [orchestration for Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140).

1.  An outer driver script that simulates the control logic of a production FL
    system, selecting simulated clients from a dataset and then executing
    federated computations defined in 2. on those clients. For example,
    [a Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py).

## Federated learning datasets

TensorFlow federated
[hosts multiple datasets](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets)
that are representative of the characteristics of real-world problems that could
be solved with federated learning.

Note: These datasets can also be consumed by any Python-based ML framework as
Numpy arrays, as documented in the
[ClientData API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData).

Datasets include:

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

*   [**Google Landmark v2 dataset**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data)
    The dataset consists of photos of various world landmarks, with images
    grouped by photographer to achieve a federated partitioning of the data. Two
    flavors of dataset are available: a smaller dataset with 233 clients and
    23080 images, and a larger dataset with 1262 clients and 164172 images.

*   [**CelebA**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data)
    A dataset of examples (image and facial attributes) of celebrity faces. The
    federated dataset has each celebrity's examples grouped together to form a
    client. There are 9343 clients, each with at least 5 examples. The dataset
    can be split into train and test groups either by clients or by examples.

*   [**iNaturalist**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data)
    A dataset consists of photos of various species. The dataset contains
    120,300 images for 1,203 species. Seven flavors of the dataset are
    available. One of them is grouped by the photographer and it consists of
    9257 clients. The rest of the datasets are grouped by the geo location where
    the photo was taken. These six flavors of the dataset consists of 11 - 3,606
    clients.

## High performance simulations

While the wall-clock time of an FL *simulation* is not a relevant metric for
evaluating algorithms (as simulation hardware isn't representative of real FL
deployment environments), being able to run FL simulations quickly is critical
for research productivity. Hence, TFF has invested heavily in providing
high-performance single and multi-machine runtimes. Documentation is under
development, but for now see the
[High-performance simulations with Kubernetes](https://www.tensorflow.org/federated/tutorials/high_performance_simulation_with_kubernetes)
tutorial, instructions on
[TFF simulations with accelerators](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators),
and instructions on
[setting up simulations with TFF on GCP](https://www.tensorflow.org/federated/gcp_setup).
The high-performance TFF runtime is enabled by default.

## TFF for different research areas

### Federated optimization algorithms

Research on federated optimization algorithms can be done in different ways in
TFF, depending on the desired level of customization.

A minimal stand-alone implementation of the
[Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided
[here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg).
The code includes
[TF functions](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py)
for local computation,
[TFF computations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py)
for orchestration, and a
[driver script](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py)
on the EMNIST dataset as an example. These files can easily be adapted for
customized applciations and algorithmic changes following detailed instructions
in the
[README](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/README.md).

A more general implementation of Federated Averaging can be found
[here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/fed_avg.py).
This implementation allows for more sophisticated optimization techniques,
including the use of different optimizers on both the server and client. Other
federated learning algorithms, including federated k-means clustering, can be
found
[here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/).

### Model update compression

Lossy compression of model updates can lead to reduced communication costs,
which in turn can lead to reduced overall training time.

To reproduce a recent [paper](https://arxiv.org/abs/2201.02664), see
[this research project](https://github.com/google-research/federated/tree/master/compressed_communication).
To implement a custom compression algorithm, see
[comparison_methods](https://github.com/google-research/federated/tree/master/compressed_communication/aggregators/comparison_methods)
in the project for baselines as an example, and
[TFF Aggregators tutorial](https://www.tensorflow.org/federated/tutorials/custom_aggregators)
if not already familiar with.

### Differential privacy

TFF is interoperable with the
[TensorFlow Privacy](https://github.com/tensorflow/privacy) library to enable
research in new algorithms for federated training of models with differential
privacy. For an example of training with DP using
[the basic DP-FedAvg algorithm](https://arxiv.org/abs/1710.06963) and
[extensions](https://arxiv.org/abs/1812.06210), see
[this experiment driver](https://github.com/google-research/federated/blob/master/differential_privacy/stackoverflow/run_federated.py).

If you want to implement a custom DP algorithm and apply it to the aggregate
updates of federated averaging, you can implement a new DP mean algorithm as a
subclass of
[`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54)
and create a `tff.aggregators.DifferentiallyPrivateFactory` with an instance of
your query. An example of implementing the
[DP-FTRL algorithm](https://arxiv.org/abs/2103.00039) can be found
[here](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)

Federated GANs (described [below](#generative_adversarial_networks)) are another
example of a TFF project implementing user-level differential privacy (e.g.,
[here in code](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L144)).

### Robustness and attacks

TFF can also be used to simulate the targeted attacks on federated learning
systems and differential privacy based defenses considered in
*[Can You Really Back door Federated Learning?](https://arxiv.org/abs/1911.07963)*.
This is done by building an iterative process with potentially malicious clients
(see
[`build_federated_averaging_process_attacked`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L412)).
The
[targeted_attack](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack)
directory contains more details.

*   New attacking algorithms can be implemented by writing a client update
    function which is a Tensorflow function, see
    [`ClientProjectBoost`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L460)
    for an example.
*   New defenses can be implemented by customizing
    ['tff.utils.StatefulAggregateFn'](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103)
    which aggregates client outputs to get a global update.

For an example script for simulation, see
[`emnist_with_targeted_attack.py`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/emnist_with_targeted_attack.py).

### Generative Adversarial Networks

GANs make for an interesting
[federated orchestration pattern](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L266-L316)
that looks a little different than standard Federated Averaging. They involve
two distinct networks (the generator and the discriminator) each trained with
their own optimization step.

TFF can be used for research on federated training of GANs. For example, the
DP-FedAvg-GAN algorithm presented in
[recent work](https://arxiv.org/abs/1911.06679) is
[implemented in TFF](https://github.com/tensorflow/federated/tree/main/federated_research/gans).
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
[`emnist_p13n_main.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/emnist_p13n_main.py).
To explore and compare different personalization strategies, you can:

*   Define a personalization strategy by implementing a `tf.function` that
    starts from an initial model, trains and evaluates a personalized model
    using each client's local datasets. An example is given by
    [`build_personalize_fn`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/p13n_utils.py).

*   Define an `OrderedDict` that maps strategy names to the corresponding
    personalization strategies, and use it as the `personalize_fn_dict` argument
    in
    [`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval).

Another approach is to avoid training a fully global model by training part of a
model entirely locally. An instantiation of this approach is described in
[this blog post](https://ai.googleblog.com/2021/12/a-scalable-approach-for-partially-local.html).
This approach is also connected to meta learning, see
[this paper](https://arxiv.org/abs/2102.03448). To explore partially local
federated learning, you can:

*   Check out the
    [tutorial](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization)
    for a complete code example applying Federated Reconstruction and
    [follow-up exercises](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization#further_explorations).

*   Create a partially local training process using
    [`tff.learning.reconstruction.build_training_process`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/reconstruction/build_training_process),
    modifying `dataset_split_fn` to customize process behavior.
