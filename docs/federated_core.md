# Federated Core

This document introduces the core layer of TFF that serves as a foundation for
[Federated Learning](federated_learning.md), and possible future non-learning
federated algorithms.

For a gentle introduction to Federated Core, please read the following
tutorials, as they introduce some of the fundamental concepts by example and
demonstrate step-by-step the construction of a simple federated averaging
algorithm.

*   [Custom Federated Algorithms, Part 1: Introduction to the Federated Core](tutorials/custom_federated_algorithms_1.ipynb).

*   [Custom Federated Algorithms, Part 2: Implementing Federated Averaging](tutorials/custom_federated_algorithms_2.ipynb).

We would also encourage you to familiarize yourself with
[Federated Learning](federated_learning.md) and the associated tutorials on
[image classification](tutorials/federated_learning_for_image_classification.ipynb)
and [text generation](tutorials/federated_learning_for_text_generation.ipynb),
as the uses of the Federated Core API (FC API) for federated learning provide
important context for some of the choices we've made in designing this layer.

## Overview

### Goals, Intended Uses, and Scope

Federated Core (FC) is best understood as a programming environment for
implementing distributed computations, i.e., computations that involve multiple
computers (mobile phones, tablets, embedded devices, desktop computers, sensors,
database servers, etc.) that may each perform non-trivial processing locally,
and communicate across the network to coordinate their work.

The term *distributed* is very generic, and TFF does not target all possible
types of distributed algorithms out there, so we prefer to use the less generic
term *federated computation* to describe the types of algorithms that can be
expressed in this framework.

While defining the term *federated computation* in a fully formal manner is
outside the scope of this document, think of the types of algorithms you might
see expressed in pseudocode in a
[research publication](https://arxiv.org/pdf/1602.05629.pdf) that describes a
new distributed learning algorithm.

The goal of FC, in a nusthell, is to enable similarly compact representation, at
a similar pseudocode-like level of abstraction, of program logic that is *not*
pseudocode, but rather, that's executable in a variety of target environments.

The key defining characteristic of the kinds of algorithms that FC is designed
to express is that actions of system participants are described in a collective
manner. Thus, we tend to talk about *each device* locally transforming data, and
the devices coordinating work by a centralized coordinator *broadcasting*,
*collecting*, or *aggregating* their results.

While TFF has been designed to be able to go beyond simple *client-server*
architectures, the notion of collective processing is fundamental. This is due
to the origins of TFF in federated learning, a technology originally designed to
support computations on potentially sensitive data that remains under control of
client devices, and that may not be simply downloaded to a centralized location
for privacy reasons. While each client in such systems contributes data and
processing power towards computing a result by the system (a result that we
would generally expect to be of value to all the participants), we also strive
at preserving each client's privacy and anonymity.

Thus, while most frameworks for distributed computing are designed to express
processing from the perspective of individual participants - that is, at the
level of individual point-to-point message exchanges, and the interdependence of
the participant's local state transitions with incoming and outgoing messages,
TFF's Federated Core is designed to describe the behavior of the system from the
*global* system-wide perspective (similarly to, e.g.,
[MapReduce](https://research.google/pubs/pub62/)).

Consequently, while distributed frameworks for general purposes may offer
operations such as *send* and *receive* as building blocks, FC provides building
blocks such as `tff.federated_sum`, `tff.federated_reduce`, or
`tff.federated_broadcast` that encapsulate simple distributed protocols.

## Language

### Python Interface

TFF uses an internal language to represent federated computations, the syntax of
which is defined by the serializable representation in
[computation.proto](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/proto/v0/computation.proto).
Users of FC API generally won't need to interact with this language directly,
though. Rather, we provide a Python API (the `tff` namespace) that wraps arounds
it as a way to define computations.

Specifically, TFF provides Python function decorators such as
`tff.federated_computation` that trace the bodies of the decorated functions,
and produce serialized representations of the federated computation logic in
TFF's language. A function decorated with `tff.federated_computation` acts as a
carrier of such serialized representation, and can embed it as a building block
in the body of another computation, or execute it on demand when invoked.

Here's just one example; more examples can be found in the
[custom algorithms](tutorials/custom_federated_algorithms_1.ipynb) tutorials.

```python
@tff.federated_computation(tff.type_at_clients(tf.float32))
def get_average_temperature(sensor_readings):
  return tff.federated_mean(sensor_readings)
```

Readers familiar with non-eager TensorFlow will find this approach analogous to
writing Python code that uses functions such as `tf.add` or `tf.reduce_sum` in a
section of Python code that defines a TensorFlow graph. Albeit the code is
technically expressed in Python, its purpose is to construct a serializable
representation of a `tf.Graph` underneath, and it is the graph, not the Python
code, that is internally executed by the TensorFlow runtime. Likewise, one can
think of `tff.federated_mean` as inserting a *federated op* into a federated
computation represented by `get_average_temperature`.

A part of the reason for FC defining a language has to do with the fact that, as
noted above, federated computations specify distributed collective behaviors,
and as such, their logic is non-local. For example, TFF provides operators,
inputs and outputs of which may exist in different places in the network.

This calls for a language and a type system that capture the notion of
distributedness.

### Type System

Federated Core offers the following categories of types. In describing these
types, we point to the type constructors as well as introduce a compact
notation, as it's a handy way or describing types of computations and operators.

First, here are the categories of types that are conceptually similar to those
found in existing mainstream languages:

*   **Tensor types** (`tff.TensorType`). Just as in TensorFlow, these have
    `dtype` and `shape`. The only difference is that objects of this type are
    not limited to `tf.Tensor` instances in Python that represent outputs of
    TensorFlow ops in a TensorFlow graph, but may also include units of data
    that can be produced, e.g., as an output of a distributed aggregation
    protocol. Thus, the TFF tensor type is simply an abstract version of a
    concrete physical representation of such type in Python or TensorFlow.

    The compact notation for tensor types is `dtype` or `dtype[shape]`. For
    example, `int32` and `int32[10]` are the types of integers and int vectors,
    respectively.

*   **Sequence types** (`tff.SequenceType`). These are TFF's abstract equivalent
    of TensorFlow's concrete concept of `tf.data.Dataset`s. Elements of
    sequences can be consumed in a sequential manner, and can include complex
    types.

    The compact representation of sequence types is `T*`, where `T` is the type
    of elements. For example `int32*` represents an integer sequence.

*   **Named tuple types** (`tff.StructType`). These are TFF's way of
    constructing tuples and dictionary-like structures that have a predefined
    number of *elements* with specific types, named or unnamed. Importantly,
    TFF's named tuple concept encompasses the abstract equivalent of Python's
    argument tuples, i.e., collections of elements of which some, but not all
    are named, and some are positional.

    The compact notation for named tuples is `<n_1=T_1, ..., n_k=T_k>`, where
    `n_k` are optional element names, and `T_k` are element types. For example,
    `<int32,int32>` is a compact notation for a pair of unnamed integers, and
    `<X=float32,Y=float32>` is a compact notation for a pair of floats named `X`
    and `Y` that may represent a point on a plane. Tuples can be nested as well
    as mixed with other types, e.g., `<X=float32,Y=float32>*` would be a compact
    notation for a sequence of points.

*   **Function types** (`tff.FunctionType`). TFF is a functional programming
    framework, with functions treated as
    [first-class values](https://en.wikipedia.org/wiki/First-class_citizen).
    Functions have at most one argument, and exactly one result.

    The compact notation for functions is `(T -> U)`, where `T` is the type of
    an argument, and `U` is the type of the result, or `( -> U)` if there's no
    argument (although no-argument functions are a degenerate concept that
    exists mostly just at the Python level). For example `(int32* -> int32)` is
    a notation for a type of functions that reduce an integer sequence to a
    single integer value.

The following types address the distributed systems aspect of TFF computations.
As these concepts are somewhat unique to TFF, we encourage you to refer to the
[custom algorithms](tutorials/custom_federated_algorithms_1.ipynb) tutorial for
additional commentary and examples.

*   **Placement type**. This type is not yet exposed in the public API other
    than in the form of 2 literals `tff.SERVER` and `tff.CLIENTS` that you can
    think of as constants of this type. It is used internally, however, and will
    be introduced in the public API in future releases. The compact
    representation of this type is `placement`.

    A *placement* represents a collective of system participants that play a
    particular role. The initial release is targeting client-server
    computations, in which there are 2 groups of participants: *clients* and a
    *server* (you can think of the latter as a singleton group). However, in
    more elaborate architectures, there could be other roles, such as
    intermediate aggregators in a multi-tiered system, who might be performing
    different types of aggregation, or use different types of data
    compression/decompression than those used by either the server or the
    clients.

    The primary purpose of defining the notion of placements is as a basis for
    defining *federated types*.

*   **Federated types** (`tff.FederatedType`). A value of a federated type is
    one that is hosted by a group of system participants defined by a specific
    placement (such as `tff.SERVER` or `tff.CLIENTS`). A federated type is
    defined by the *placement* value (thus, it is a
    [dependent type](https://en.wikipedia.org/wiki/Dependent_type)), the type of
    *member constituents* (what kind of content each of the participants is
    locally hosting), and the additional bit `all_equal` that specifies whether
    all participants are locally hosting the same item.

    The compact notation for federated type of values that include items (member
    constituents) of type `T`, each hosted by group (placement) `G` is `T@G` or
    `{T}@G` with the `all_equal` bit set or not set, respectively.

    For example:

    *   `{int32}@CLIENTS` represents a *federated value* that consists of a set
        of potentially distinct integers, one per client device. Note that we
        are talking about a single *federated value* as encompassing multiple
        items of data that appear in multiple locations across the network. One
        way to think about it is as a kind of tensor with a "network" dimension,
        although this analogy is not perfect because TFF does not permit
        [random access](https://en.wikipedia.org/wiki/Random_access) to member
        constituents of a federated value.

    *   `{<X=float32,Y=float32>*}@CLIENTS` represents a *federated data set*, a
        value that consists of multiple sequences of `XY` coordinates, one
        sequence per client device.

    *   `<weights=float32[10,5],bias=float32[5]>@SERVER` represents a named
        tuple of weight and bias tensors at the server. Since we've dropped the
        curly braces, this indicates the `all_equal` bit is set, i.e., there's
        only a single tuple (regardless of how many server replicas there might
        be in a cluster hosting this value).

### Building Blocks

The language of Federated Core is a form of
[lambda-calculus](https://en.wikipedia.org/wiki/Lambda_calculus), with a few
additional elements.

It provides the following programing abstractions currently exposed in the
public API:

*   **TensorFlow** computations (`tff.tf_computation`). These are sections of
    TensorFlow code wrapped as reusable components in TFF using the
    `tff.tf_computation` decorator. They always have functional types, and
    unlike functions in TensorFlow, they can take structured parameters or
    return structured results of a sequence type.

    Here's one example, a TF computation of type `(int32* -> int)` that uses the
    `tf.data.Dataset.reduce` operator to compute a sum of integers:

    ```python
    @tff.tf_computation(tff.SequenceType(tf.int32))
    def add_up_integers(x):
      return x.reduce(np.int32(0), lambda x, y: x + y)
    ```

*   **Intrinsics** or *federated operators* (`tff.federated_...`). This is a
    library of functions such as `tff.federated_sum` or
    `tff.federated_broadcast` that constitute the bulk of FC API, most of which
    represent distributed communication operators for use with TFF.

    We refer to these as *intrinsics* because, somewhat like
    [intrinsic functions](https://en.wikipedia.org/wiki/Intrinsic_function),
    they are an open-ended, extensible set of operators that are understood by
    TFF, and compiled down into lower-level code.

    Most of these operators have parameters and results of federated types, and
    most are templates that can be applied to various kinds of data.

    For example, `tff.federated_broadcast` can be thought of as a template
    operator of a functional type `T@SERVER -> T@CLIENTS`.

*   **Lambda expressions** (`tff.federated_computation`). A lambda expression in
    TFF is the equivalent of a `lambda` or `def` in Python; it consists of the
    parameter name, and a body (expression) that contains references to this
    parameter.

    In Python code, these can be created by decorating Python functions with
    `tff.federated_computation` and defining an argument.

    Here's an example of a lambda expression we've already mentioned earlier:

    ```python
    @tff.federated_computation(tff.type_at_clients(tf.float32))
    def get_average_temperature(sensor_readings):
      return tff.federated_mean(sensor_readings)
    ```

*   **Placement literals**. For now, only `tff.SERVER` and `tff.CLIENTS` to
    allow for defining simple client-server computations.

*   **Function invocations** (`__call__`). Anything that has a functional type
    can be invoked using the standard Python `__call__` syntax. The invocation
    is an expression, the type of which is the same as the type of the result of
    the function being invoked.

    For example:

    *   `add_up_integers(x)` represents an invocation of the TensorFlow
        computation defined earlier on an argument `x`. The type of this
        expression is `int32`.

    *   `tff.federated_mean(sensor_readings)` represents an invocation of the
        federated averaging operator on `sensor_readings`. The type of this
        expression is `float32@SERVER` (assuming context from the example
        above).

*   Forming **tuples** and **selecting** their elements. Python expressions of
    the form `[x, y]`, `x[y]`, or `x.y` that appear in the bodies of functions
    decorated with `tff.federated_computation`.
