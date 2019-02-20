# Federated Learning

## Overview

This document introduces interfaces designed to facilitate federated
learning tasks, such as federated training or evaluation with existing machine
learning models implemented in TensorFlow. In designing these interfaces, our
primary goal was to make it possible to experiment with federated learning
without requiring the knowledge of how it works under the hood, and to evaluate
the implemented federated learning algorithms on a variety of existing models
and data. We would like to also encourage you to contribute back to the
platform. TFF has been designed with extensibility and composability in mind,
and we welcome contributions; we are excited to see what you come up
with!

The interfaces offered by this layer consists of the following three key parts:

* **Models**. Classes and helper functions that allow you to wrap your existing
  models for use with TFF. Wrapping a model can be as simple as calling a
  single wrapping function in case of Keras models, or defining a subclass of
  an interface we provided in more specialized cases for customizability.

* **Builders**. Helper functions that construct basic types of federated tasks
  using your existing models. At the moment, we only offer builders for simple
  training and evalation.

* **Datasets**. Canned collections of data that you can download and access
  in Python for use in simulating federated learning scenarios. Although
  federated learning is designed for use with distributed data that cannot be
  simply downloaded at a centralized location, at the research and development
  stages it is often convenient to conduct initial experiments using data that
  can be downloaded and manipulated locally, especially for developers who
  might be new to the approach.

These interfaces are defined primarily in the `tff.learning` namespace, except
for research data sets and other simulation-related capabilities that have been
grouped in `tff.simulation`. This layer is implemented using lower-level
interfaces offered by the [Federated Core (FC)](federated_core.md). The latter
also serves as a runtime environment.

Before proceeding, we recommend that you first review the tutorials on
[image classification]
(tutorials/federated_learning_for_image_classification.ipynb) and
[text generation]
(tutorials/federated_learning_for_text_generation.ipynb), as they introduce
most of the concepts described here using concrete examples with real datasets
and code. If you're interested in learning more about how TFF works, you may
want to skim over the
[custom algorithms](tutorials/custom_federated_algorithms.ipynb) tutorial as a
quick introduction to the lower-level interfaces we use to express the logic of
federated computations, and to study the existing implementation of the
`tff.learning` interfaces.

## Models

### Architectural assumptions

#### Serialization

TFF aims at supporting a variety distributed learning scenarios in which the
machine learning model code you write might be executing on a large number of
heterogeneous clients with diverse capabilities. While at one end of the
spectrum, in some applications those clients might be powerful database
servers, many important uses our platform intends to support involve mobile
and embedded devices with limited resources. We cannot assume that these
devices are capable of hosting Python runtimes; the only thing we can assume
at this point is that they are capable of hosting a local TensorFlow runtime.
Thus, a fundamental architectural assumption we make in TFF is that your model
code must be serializable as a TensorFlow graph.

This does *not* mean you cannot or shouldn't write code in a manner that would
be compliant with the latest practices, such as the ability for TensorFlow to
automatically inject control dependencies, but it does mean that your code must
not rely on any Python state, or Python control flow constructs that cannot be
converted by tools such as
[Autograph](https://www.tensorflow.org/guide/autograph)
into a purely TensorFlow logic that can be serialized as graph operations in a
`tf.GraphDef`, as a `SavedModel`, or an equivalent future mechanism.

As a corollary of this, TFF cannot simply take an already-constructed model,
available as `tf.keras.models.Model` or other Python object. Your model logic
should be constructed in a context controlled by TFF in order for TFF to be
able to correctly serialize it. Thus, all of our APIs require you to provide
some form of a model *constructor* that will be invoked by TFF in the context
of an empty `tf.Graph` (to be serialized under the hood as a `tf.GraphDef`). In
addition, being a strongly-typed environment, TFF will require a little bit of
additional *metadata*, such as a specification of your model's input.

#### Aggregation

Depending on the concrete deployment scenario, your models may be instantiated
as a part of a system that consists of multiple layers, and this
has consequences on the structure of data flows, particularly with respect to
how various types of data defined by your model are aggregated.

There will always be at least two layers:

* **Local aggregation**. This level of aggregation refers to aggregation across
  multiple batches of examples owned by an individual client. It applies to
  both the model parameters (variables), which continue to sequentially evolve
  as your model instance on the given client is locally exposed to subsequent
  batches of data, as well as the statistics you compute (such as average loss,
  accuracy, and other metrics), which your model will again continue to update
  locally as it iterates over batches of data in each individual client's local
  data stream.

  Performing aggregation at this level is the responsibility of your model code.

  The general structure of processing is as follows:

  * Your model first constructs all the summary statistics it needs to perform
    aggregation, such as the number of batches or the number of examples
    processed, the sum of per-batch or per-example losses, etc.

  * TFF invokes a method exposed by your model (`forward_pass`) multiple times,
    sequentially over subsequent batches of client's data, which allows your
    trainable variables to evolve, as well as lets your model update all the
    summary metrics it defined.

  * Finally, TFF invokes another method exposed by your model
    (`report_local_outputs`) to allow your model to compile all the summary
    statistics it collected into a compact set of metrics to be exported by
    the client. This is where your model code may, for example, divide the
    sum of losses by the number of examples processsed to export the average
    loss, etc.

* **Federated aggregation**. This level of aggregation refers to aggregation
  across multiple clients (devices) in the system. Again, it applies to both the
  model parametres (variables), which are being averaged across clients, as
  well as the metrics your model exported as a result of local aggregation.

  Performing aggregation at this level is the responsibility of TFF. As a model
  creator, however, you can control this process (more on this below).

  The general structure of processing is as follows:

  * The initial model, and any parameters required for training, are distributed
    by a server to a subset of clients that will participate in a round of
    training or evaluation.

  * On each client, independently and in parallel, your model code is invoked
    repeatedly on a stream of local data batches to produce a new set of model
    parameters (when training), and a new set of local metrics, as described
    above (this is local aggregation).

  * TFF runs a distributed aggregation protocol to accumulate and aggregate the
    model parameters and locally exported metrics across the system. This logic
    is expressed in a declarative manner using TFF's own *federated computation*
    language (not in TensorFlow); you can learn more about it in the
    [custom algorithms](tutorials/custom_federated_algorithms.ipynb) tutorial,
    although in the vast majority of applications, you won't need to, as one of
    the existing example templates will likely fit your needs.

### Abstract interfaces

This basic *constructor* + *metadata* interface is represented by the interface
`tff.learning.Model`, as follows:

* The constructor, `forward_pass`, and `report_local_outputs` methods should
  construct model variables, forward pass, and statistics you wish to report,
  correspondingly. Those methods will be invoked in the context of an empty
  `tf.Graph` created by TFF, and thus, they will currently be executing in a
  non-eager context (although in future, we might be able to support eager
  execution of your model code in specialized circumstances, e.g., to support
  local simulations). We recommend that you write your model code in a manner
  such that it can work correctly in either eager or non-eager mode.

* The properties `input_spec`, as well as the 3 properties that return subsets
  of your trainable, non-trainable, and local variables represent the metadata.
  TFF uses this information to determine how to connect parts of your model to
  the federated averaging algorithm, and to define internal type signatures to
  assist in verifying the correctness of the constructed system (so that your
  model cannot be instantiated over data that does not match what the model is
  designed to consume).

Together, all of these methods and properties control the process by which the
model is serialized into a `tf.GraphDef` under the hood.

In addition, the abstract interface `tff.learning.Model` exposes a property
`federated_output_computation` that, together with the `report_local_outputs`
property mentioned earlier, allows you to control the process of aggregating
summary statistics.

Finally, the derived abstract interface `tff.learning.TrainableModel` allows
you to customize the manner in which TFF executes individual training steps,
such as by specifying your own optimizer, or defining separate metrics to be
computed before and after training. You can customize this by overriding the
newly introduced abstract method `train_on_batch`. You don't have to do it,
though, if you only plan to use your model for evaluation, or if you're content
with TFF choosing the standard optimizer for you.

You can find examples of how to define your own custom `tf.learning.Model` in
the second part of our [image classification]
(tutorials/federated_learning_for_image_classification.ipynb) tutorial, as well
as in the example models we use for testing in [`model_examples.py`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/model_examples.py).

### Converters for Keras

Nearly all the information that's required by TFF can be derived by calling
Keras interfaces, so if you have a Keras model, you can rely on either of the
two methods below to construct a `tff.learning.TrainableModel` instance for you:

* `tff.learning.from_keras_model`

* `tff.learning.from_compiled_keras_model`

Note that TFF still wants you to provide a constructor - a no-argument *model
function* such as the following:

```
def model_fn():
  keras_model = ...
  keras_model.compile(...)
  return tff.learning.from_compiled_keras_model(keras_model, ...)
```

The one additional bit of information expected by TFF that you need to supply
as an extra parameter to the converter functions is a sample batch of data.
TFF uses this sample batch to determine the form of your model's input and the
output metrics it will emit. While technically, such information can be obtained
by querying the Keras interfaces, it's not available until your model is invoked
on input data, which as noted earlier, in TFF would generally be done on client
devices after the model has been deployed, so it's a non-starter. Supplying the
dummy batch allows TFF to perform all the analysis it needs to perform
statically, at construction time.

The use of Keras wrappers is illustrated in our [image classification]
(tutorials/federated_learning_for_image_classification.ipynb) and
[text generation](tutorials/federated_learning_for_text_generation.ipynb)
tutorials.

## Builders

### Architectural assumptions

#### Execution

As noted above, TFF aims at supporting a variety of scenarios that may involve
devices incapable of hosting Python runtimes, and for this reason, the models
you will develop need to be serialized, so that they can be remotely executed
in a variety of target deployment environments.

The same holds true for the federated training and federated evaluation logic.

Rather than providing a single deployment environment to run your models, TFF
acts as a compiler pipeline - it first *compiles* federated learning algorithms
into an abstract serialized representation of the entire distributed system,
and then provides ways to *execute* that serialized representation in a variety
of environments (for now just locally, e.g., in a Jupyter notebook). We refer
to the serialized representations emitted by the compiler as *federated
computations*.

A federated computation generated by TFF's Federated Learning API, such as
a training algorithm that uses
[federated model averaging](https://arxiv.org/abs/1602.05629), or a federated
evaluation, includes a number of elements, most notably:

* A serialized form of your model code, currently as a `tf.GraphDef`, as well
  as additional TensorFlow code constructed by the Federated Learning framework
  to drive your model's training/evaluation loop (such as constructing
  optimizers, applying model updates, iterating over `tf.data.Dataset`s, and
  computing metrics, to name a few).

* A declarative specification of the kinds of participants that will engage
  in the computation (typically *clients* and a *server*), the communication
  patterns that will be used (typically various forms of *aggregation* across
  the client devices, and *broadcasting* from the server to all clients), and
  how distributed communication is interleaved with the execution of TensorFlow
  code.

The *federated computations* represented in this serialized form are expressed
in a platform-independent internal language distinct from Python, but to use
the Federated Learning API, you won't need to concern yourself with the details
of this representation. The computations are represented in your Python code as
objects of type `tff.Computation`, which for the most part you can treat as
opaque Python `callable`s.

In the tutorials, you will invoke those federated computations as if they were
regular Python functions, to be executed locally. However, as noted above,
you should be aware of the fact that TFF is designed in a way that allows you
to express federated computations in a manner agnostic to most aspects of the
execution environment, so that they can potentially be deployable to, e.g.,
groups of devices running `Android`, or to clusters in a datacenter.

The most notable consequence of this is that, any Python code or state you
write will be traced or captured at the compilation time (i.e., when you invoke
one of the `build_...` methods described below). This concerns especially any
kind of eager TensorFlow constructs, such as eager tensors or eager data sets.
Thus, for example, when you invoke the constructed federated computations in a
training loop, any changes you make to eager tensors or any other Python
objects you may have used in your code will not be reflected unless they're
explicitly supplied as input arguments at the time you invoke a computation.
The code that had been traced at compilation time is detached from, and
unaffected by any further changes to Python state after the `build_...` methods
return.

#### Modeling state

TFF is a functional programming environment, yet many processes of interest in
federated learning applications are stateful. For example, a training loop that
involves multiple rounds of federated model averaging is an example of what we
could classify as a *stateful process*. In this process, the state that evolves
from round to round includes the set of model parameters that emerges at the
server in subsequent rounds of federated averaging, and that links the
subsequent rounds into a meaningful sequence.

Since TFF is functional, processes that are stateful must be modeled in TFF as
computations that accept the state they modify at input, and include the new
version of the state in the output.

In order to fully define a stateful process, one also needs to specify
where the initial state comes from (otherwise we cannot bootstrap the process).
This is captured in the definition of the helper class
`tff.utils.IterativeProcess`, with the 2 properties `initialize` and `next`
corresponding to the initialization and iteration, respectively.

### Available builders

At the moment, TFF provides 2 builder methods that generate the federated
computations for federated training and evaluation, the example usage of which
is illustrated in the [image classification]
(tutorials/federated_learning_for_image_classification.ipynb) tutorial:

* `tff.learning.build_federated_averaging_process` takes a *model function*,
  and  returns a *pair* of federated computations for federated training,
  wrapped in a 2-element tuple `tff.utils.IterativeProcess`.

* `tff.learning.build_federated_evaluation` takes a *model function* and returns
  a federated computation for federated evaluation of models.


The first of the above is stateful (and therefore modeled as a pair of methods
*initialize* and *next*), the second isn't.

## Datasets

### Architectural assumptions

#### Client selection

In the typical federated learning scenario targeted by TFF, we are dealing with
a large *population* of potentially hundreds of millions of client devices, of
which only a small portion may be active and available for training at any given
moment (for example, this may be limited to clients that are plugged in to a
power source, not on a metered network, and otherwise idle). Generally, the set
of clients available to participate in training or evaluation is outside of the
developer's control. Furthermore, as it's impractical to coordinate millions of
clients, a typical round of training or evaluation will include only a fraction
of the available clients, which may be
[sampled at random](https://arxiv.org/pdf/1902.01046.pdf).

The key consequence of this is that federated computations, by design, are
expressed in a manner that is oblivious to the exact set of participants; all
processing is expressed as aggregate operations on an abstract group of
*clients*, and that group might vary from one round of training to another.
The actual binding of the computation to the concrete participants, and thus
to the concrete data they feed into the computation, is thus modeled outside
of the computation itself.

In order to simulate a realistic deployment of your federated learning code,
you will thus generally want to model it as a training loop that might look
more or less as follows:

```
trainer = tff.learning.build_federated_averaging_process(...)
state = trainer.initialize()
federated_training_data = ...

def sample(federate_data):
  return ...

while True:
  data_for_this_round = sample(federated_training_data)
  state, metrics = trainer.next(state, data_for_this_round)
  print (metrics)
```

In order to facilitate this, when using TFF in simulations, federated data is
accepted as Python `list`s, with one element per participating client device to
represent that device's local `tf.data.Dataset`. Thus, sampling a federated
data in a simulation generally involves simply picking a subset of elements
from a list.

### Abstract interfaces

In order to standardize dealing with federated data sets, TFF provides an
abstract interface `tff.simulation.ClientData`, which allows one to enumerate
the set of clients, and to construct a `tf.data.Dataset` that contains the
data of a paricular client. Those `tf.data.Dataset`s can be fed directly as
input to the generated federated computations in eager mode.

It should be noted that the ability to access client identities is a feature
that's only provided by the datasets for use in simulations, where the ability
to extract data from specific subsets of clients may be needed. The compiled
computations and the underlying runtime do *not* involve any notion of client
identity. Once data from a specific subset of clients has been selected as an
input, e.g., in a call to `tff.utils.IterativeProcess.next`, client identities
no longer appear in it.

### Available data sets

We have dedicated the namespace `tff.simulation.datasets` for datasets that
implement the `tff.simulation.ClientData` interface for use in simulations, and
seeded it with 2 data sets to support the
[image classification](tutorials/federated_learning_for_image_classification.ipynb)
and [text generation](tutorials/federated_learning_for_text_generation.ipynb)
tutorials. We'd like to encourage you to contribute your own data sets to the
platform.
