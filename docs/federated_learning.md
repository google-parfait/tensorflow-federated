# Federated Learning

## Overview

This document introduces interfaces that facilitate federated learning tasks,
such as federated training or evaluation with existing machine learning models
implemented in TensorFlow. In designing these interfaces, our primary goal was
to make it possible to experiment with federated learning without requiring the
knowledge of how it works under the hood, and to evaluate the implemented
federated learning algorithms on a variety of existing models and data. We
encourage you to contribute back to the platform. TFF has been designed with
extensibility and composability in mind, and we welcome contributions; we are
excited to see what you come up with!

The interfaces offered by this layer consist of the following three key parts:

*   **Models**. Classes and helper functions that allow you to wrap your
    existing models for use with TFF. Wrapping a model can be as simple as
    calling a single wrapping function (e.g., `tff.learning.from_keras_model`),
    or defining a subclass of the `tff.learning.Model` interface for full
    customizability.

*   **Federated Computation Builders**. Helper functions that construct
    federated computations for training or evaluation, using your existing
    models.

*   **Datasets**. Canned collections of data that you can download and access in
    Python for use in simulating federated learning scenarios. Although
    federated learning is designed for use with decentralized data that cannot
    be simply downloaded at a centralized location, at the research and
    development stages it is often convenient to conduct initial experiments
    using data that can be downloaded and manipulated locally, especially for
    developers who might be new to the approach.

These interfaces are defined primarily in the `tff.learning` namespace, except
for research data sets and other simulation-related capabilities that have been
grouped in `tff.simulation`. This layer is implemented using lower-level
interfaces offered by the [Federated Core (FC)](federated_core.md), which also
provides a runtime environment.

Before proceeding, we recommend that you first review the tutorials on
[image classification](tutorials/federated_learning_for_image_classification.ipynb)
and [text generation](tutorials/federated_learning_for_text_generation.ipynb),
as they introduce most of the concepts described here using concrete examples.
If you're interested in learning more about how TFF works, you may want to skim
over the [custom algorithms](tutorials/custom_federated_algorithms_1.ipynb)
tutorial as an introduction to the lower-level interfaces we use to express the
logic of federated computations, and to study the existing implementation of the
`tff.learning` interfaces.

## Models

### Architectural assumptions

#### Serialization

TFF aims at supporting a variety of distributed learning scenarios in which the
machine learning model code you write might be executing on a large number of
heterogeneous clients with diverse capabilities. While at one end of the
spectrum, in some applications those clients might be powerful database servers,
many important uses our platform intends to support involve mobile and embedded
devices with limited resources. We cannot assume that these devices are capable
of hosting Python runtimes; the only thing we can assume at this point is that
they are capable of hosting a local TensorFlow runtime. Thus, a fundamental
architectural assumption we make in TFF is that your model code must be
serializable as a TensorFlow graph.

You can (and should) still develop your TF code following the latest best
practices like using eager mode. However, the final code must be serializable
(e.g., can be wrapped as a `tf.function` for eager-mode code). This ensures that
any Python state or control flow necessary at execution time can be serialized
(possibly with the help of
[Autograph](https://www.tensorflow.org/guide/autograph)).

Currently, TensorFlow does not fully support serializing and deserializing
eager-mode TensorFlow. Thus, serialization in TFF currently follows the TF 1.0
pattern, where all code must be constructed inside a `tf.Graph` that TFF
controls. This means currently TFF cannot consume an already-constructed model;
instead, the model definition logic is packaged in a no-arg function that
returns a `tff.learning.Model`. This function is then called by TFF to ensure
all components of the model are serialized. In addition, being a strongly-typed
environment, TFF will require a little bit of additional *metadata*, such as a
specification of your model's input type.

#### Aggregation

We strongly recommend most users construct models using Keras, see the
[Converters for Keras](#converters-for-keras) section below. These wrappers
handle the aggregation of model updates as well as any metrics defined for the
model automatically. However, it may still be useful to understand how
aggregation is handled for a general `tff.learning.Model`.

There are always at least two layers of aggregation in federated learning: local
on-device aggregation, and cross-device (or federated) aggregation:

*   **Local aggregation**. This level of aggregation refers to aggregation
    across multiple batches of examples owned by an individual client. It
    applies to both the model parameters (variables), which continue to
    sequentially evolve as the model is locally trained, as well as the
    statistics you compute (such as average loss, accuracy, and other metrics),
    which your model will again update locally as it iterates over each
    individual client's local data stream.

    Performing aggregation at this level is the responsibility of your model
    code, and is accomplished using standard TensorFlow constructs.

    The general structure of processing is as follows:

    *   The model first constructs `tf.Variable`s to hold aggregates, such as
        the number of batches or the number of examples processed, the sum of
        per-batch or per-example losses, etc.

    *   TFF invokes the `forward_pass` method on your `Model` multiple times,
        sequentially over subsequent batches of client data, which allows you to
        update the variables holding various aggregates as a side effect.

    *   Finally, TFF invokes the `report_local_unfinalized_metrics` method on
        your Model to allow your model to compile all the summary statistics it
        collected into a compact set of metrics to be exported by the client.
        This is where your model code may, for example, divide the sum of losses
        by the number of examples processed to export the average loss, etc.

*   **Federated aggregation**. This level of aggregation refers to aggregation
    across multiple clients (devices) in the system. Again, it applies to both
    the model parameters (variables), which are being averaged across clients,
    as well as the metrics your model exported as a result of local aggregation.

    Performing aggregation at this level is the responsibility of TFF. As a
    model creator, however, you can control this process (more on this below).

    The general structure of processing is as follows:

    *   The initial model, and any parameters required for training, are
        distributed by a server to a subset of clients that will participate in
        a round of training or evaluation.

    *   On each client, independently and in parallel, your model code is
        invoked repeatedly on a stream of local data batches to produce a new
        set of model parameters (when training), and a new set of local metrics,
        as described above (this is local aggregation).

    *   TFF runs a distributed aggregation protocol to accumulate and aggregate
        the model parameters and locally exported metrics across the system.
        This logic is expressed in a declarative manner using TFF's own
        *federated computation* language (not in TensorFlow). See the
        [custom algorithms](tutorials/custom_federated_algorithms_1.ipynb)
        tutorial for more on the aggregation API.

### Abstract interfaces

This basic *constructor* + *metadata* interface is represented by the interface
`tff.learning.Model`, as follows:

*   The constructor, `forward_pass`, and `report_local_unfinalized_metrics`
    methods should construct model variables, forward pass, and statistics you
    wish to report, correspondingly. The TensorFlow constructed by those methods
    must be serializable, as discussed above.

*   The `input_spec` property, as well as the 3 properties that return subsets
    of your trainable, non-trainable, and local variables represent the
    metadata. TFF uses this information to determine how to connect parts of
    your model to the federated optimization algorithms, and to define internal
    type signatures to assist in verifying the correctness of the constructed
    system (so that your model cannot be instantiated over data that does not
    match what the model is designed to consume).

In addition, the abstract interface `tff.learning.Model` exposes a property
`metric_finalizers` that takes in a metric's unfinalized values (returned by
`report_local_unfinalized_metrics()`) and returns the finalized metric values.
The `metric_finalizers` and `report_local_unfinalized_metrics()` method will be
used together to build a cross-client metrics aggregator when defining the
federated training processes or evaluation computations. For example, a simple
`tff.learning.metrics.sum_then_finalize` aggregator will first sum the
unfinalized metric values from clients, and then call the finalizer functions at
the server.

You can find examples of how to define your own custom `tff.learning.Model` in
the second part of our
[image classification](tutorials/federated_learning_for_image_classification.ipynb)
tutorial, as well as in the example models we use for testing in
[`model_examples.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/model_examples.py).

### Converters for Keras

Nearly all the information that's required by TFF can be derived by calling
`tf.keras` interfaces, so if you have a Keras model, you can rely on
`tff.learning.from_keras_model` to construct a `tff.learning.Model`.

Note that TFF still wants you to provide a constructor - a no-argument *model
function* such as the following:

```python
def model_fn():
  keras_model = ...
  return tff.learning.from_keras_model(keras_model, sample_batch, loss=...)
```

In addition to the model itself, you supply a sample batch of data which TFF
uses to determine the type and shape of your model's input. This ensures that
TFF can properly instantiate the model for the data that will actually be
present on client devices (since we assume this data is not generally available
at the time you are constructing the TensorFlow to be serialized).

The use of Keras wrappers is illustrated in our
[image classification](tutorials/federated_learning_for_image_classification.ipynb)
and [text generation](tutorials/federated_learning_for_text_generation.ipynb)
tutorials.

## Federated Computation Builders

The `tff.learning` package provides several builders for `tff.Computation`s that
perform learning-related tasks; we expect the set of such computations to expand
in the future.

### Architectural assumptions

#### Execution

There are two distinct phases in running a federated computation.

*   **Compile**: TFF first *compiles* federated learning algorithms into an
    abstract serialized representation of the entire distributed computation.
    This is when TensorFlow serialization happens, but other transformations can
    occur to support more efficient execution. We refer to the serialized
    representation emitted by the compiler as a *federated computation*.

*   **Execute** TFF provides ways to *execute* these computations. For now,
    execution is only supported via a local simulation (e.g., in a notebook
    using simulated decentralized data).

A federated computation generated by TFF's Federated Learning API, such as a
training algorithm that uses
[federated model averaging](https://arxiv.org/abs/1602.05629), or a federated
evaluation, includes a number of elements, most notably:

*   A serialized form of your model code as well as additional TensorFlow code
    constructed by the Federated Learning framework to drive your model's
    training/evaluation loop (such as constructing optimizers, applying model
    updates, iterating over `tf.data.Dataset`s, and computing metrics, and
    applying the aggregated update on the server, to name a few).

*   A declarative specification of the communication between the *clients* and a
    *server* (typically various forms of *aggregation* across the client
    devices, and *broadcasting* from the server to all clients), and how this
    distributed communication is interleaved with the client-local or
    server-local execution of TensorFlow code.

The *federated computations* represented in this serialized form are expressed
in a platform-independent internal language distinct from Python, but to use the
Federated Learning API, you won't need to concern yourself with the details of
this representation. The computations are represented in your Python code as
objects of type `tff.Computation`, which for the most part you can treat as
opaque Python `callable`s.

In the tutorials, you will invoke those federated computations as if they were
regular Python functions, to be executed locally. However, TFF is designed to
express federated computations in a manner agnostic to most aspects of the
execution environment, so that they can potentially be deployable to, e.g.,
groups of devices running `Android`, or to clusters in a datacenter. Again, the
main consequence of this are strong assumptions about
[serialization](#serialization). In particular, when you invoke one of the
`build_...` methods described below the computation is fully serialized.

#### Modeling state

TFF is a functional programming environment, yet many processes of interest in
federated learning are stateful. For example, a training loop that involves
multiple rounds of federated model averaging is an example of what we could
classify as a *stateful process*. In this process, the state that evolves from
round to round includes the set of model parameters that are being trained, and
possibly additional state associated with the optimizer (e.g., a momentum
vector).

Since TFF is functional, stateful processes are modeled in TFF as computations
that accept the current state as an input and then provide the updated state as
an output. In order to fully define a stateful process, one also needs to
specify where the initial state comes from (otherwise we cannot bootstrap the
process). This is captured in the definition of the helper class
`tff.templates.IterativeProcess`, with the 2 properties `initialize` and `next`
corresponding to the initialization and iteration, respectively.

### Available builders

At the moment, TFF provides various builder functions that generate federated
computations for federated training and evaluation. Two notable examples
include:

*   `tff.learning.algorithms.build_weighted_fed_avg`, which takes as input a
    *model function* and a *client optimizer*, and returns a stateful
    `tff.learning.templates.LearningProcess` (which subclasses
    `tff.templates.IterativeProcess`).

*   `tff.learning.build_federated_evaluation` takes a *model function* and
    returns a single federated computation for federated evaluation of models,
    since evaluation is not stateful.

## Datasets

### Architectural assumptions

#### Client selection

In the typical federated learning scenario, we have a large *population* of
potentially hundreds of millions of client devices, of which only a small
portion may be active and available for training at any given moment (for
example, this may be limited to clients that are plugged in to a power source,
not on a metered network, and otherwise idle). Generally, the set of clients
available to participate in training or evaluation is outside of the developer's
control. Furthermore, as it's impractical to coordinate millions of clients, a
typical round of training or evaluation will include only a fraction of the
available clients, which may be
[sampled at random](https://arxiv.org/pdf/1902.01046.pdf).

The key consequence of this is that federated computations, by design, are
expressed in a manner that is oblivious to the exact set of participants; all
processing is expressed as aggregate operations on an abstract group of
anonymous *clients*, and that group might vary from one round of training to
another. The actual binding of the computation to the concrete participants, and
thus to the concrete data they feed into the computation, is thus modeled
outside of the computation itself.

In order to simulate a realistic deployment of your federated learning code, you
will generally write a training loop that looks like this:

```python
trainer = tff.learning.algorithms.build_weighted_fed_avg(...)
state = trainer.initialize()
federated_training_data = ...

def sample(federate_data):
  return ...

while True:
  data_for_this_round = sample(federated_training_data)
  result = trainer.next(state, data_for_this_round)
  state = result.state
```

In order to facilitate this, when using TFF in simulations, federated data is
accepted as Python `list`s, with one element per participating client device to
represent that device's local `tf.data.Dataset`.

### Abstract interfaces

In order to standardize dealing with simulated federated data sets, TFF provides
an abstract interface `tff.simulation.datasets.ClientData`, which allows one to
enumerate the set of clients, and to construct a `tf.data.Dataset` that contains
the data of a particular client. Those `tf.data.Dataset`s can be fed directly as
input to the generated federated computations in eager mode.

It should be noted that the ability to access client identities is a feature
that's only provided by the datasets for use in simulations, where the ability
to train on data from specific subsets of clients may be needed (e.g., to
simulate the diurnal avaiablity of different types of clients). The compiled
computations and the underlying runtime do *not* involve any notion of client
identity. Once data from a specific subset of clients has been selected as an
input, e.g., in a call to `tff.templates.IterativeProcess.next`, client
identities no longer appear in it.

### Available data sets

We have dedicated the namespace `tff.simulation.datasets` for datasets that
implement the `tff.simulation.datasets.ClientData` interface for use in
simulations, and seeded it with datasets to support the
[image classification](tutorials/federated_learning_for_image_classification.ipynb)
and [text generation](tutorials/federated_learning_for_text_generation.ipynb)
tutorials. We'd like to encourage you to contribute your own datasets to the
platform.
