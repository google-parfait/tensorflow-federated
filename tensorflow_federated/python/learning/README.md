# TensorFlow Federated's Learning API

This document contains brief guidelines on the organization of this component.
This is expected to evolve as new capabilities are added.

## What is `tff.learning` for?

This API contains libraries intended to make it easy to build federated learning
algorithms. The API is focused on `production-ready` algorithms, which typically
meet the following criteria:

1.  Allow integrations with privacy technologies. Currently, most privacy
    technologies offered by TFF are in the form of
    [aggregators](https://www.tensorflow.org/federated/api_docs/python/tff/aggregators),
    which can aggregate client updates in privacy-preserving ways.
2.  Are generally (but not exclusively) model-agnostic, either by applying to a
    large number of model architectures (such as Federated Averaging), or by
    providing functionality independent of the model architecture (such as
    optimizers).

See
[Federated Reconstruction](https://www.tensorflow.org/federated/api_docs/python/tff/learning/reconstruction)
for an example of a production-ready algorithm that differs from Federated
Averaging.

## How do I get my algorithm into `tff.learning`?

We welcome contributions. Generally, we require algorithms to be 1) well-tested,
2) have strong evidence that they solve a well-scoped problem in federated
learning, and 3) can be easily composed or applied to a wide variety of
federated settings. If your algorithm meets these criteria, please reach out and
we will work with you to to make sure the algorithm can be added to the API.

Note that algorithms are generally intended to be built by composing various
algorithmic building blocks. Please see the
[`tff.learning.templates`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/templates)
API for examples of these algorithmic building blocks, and see the
[`tff.learning.algorithms` API](https://www.tensorflow.org/federated/api_docs/python/tff/learning/algorithms)
for examples of algorithms composing these.

We also welcome contributions related to supporting libraries, including
metrics, learning-specific aggregators, and optimizers.

## Target Audience and Interface Surfaces

The interfaces and code in the `learning` component may be consumed by three
classes of users:

*   **Modelers**. Users who will supply their own machine learning models, and
    use federated learning interfaces to train them on their data. These users
    should only depend on the public API surface that consist of a short list of
    curated symbols in `__init__.py` and create a build dependency only on the
    `learning` library.

    The symbols included in this public API surface are documented in the
    automatically generated Python
    [API documentation](https://www.tensorflow.org/federated/api_docs/python/tff).
    Typically, a user will wrap their ML model as a subclass of
    `tff.learning.models.VariableModel`, and then invoke one of the constructors
    listed in the API to create one or more `tff.Computation`s that implement
    predefined federated tasks (such as training or evaluation), and that can be
    invoked to run on their data sets.

*   **Federated Learning Researchers**. Users who will develop new federated
    learning algorithms, and may selectively reuse or extend a subset of the
    components from the `learning` directory, but not contribute them as a part
    of TFF proper, or maintain them either externally, or in a `research`
    directory during the development phase.

    The public API surface that supports this class of users is currently under
    development. The curated list of symbols that make up this API is emerging
    in `framework/__init__.py`, and some of the components may eventually
    migrate there, but most are nested directly in `learning`. The symbols in
    this API are included in the documentation generated for
    `tff.learning.framework`.

*   **TensorFlow Federated Contributors**. Users who will contribute federated
    learning algorithms to TensorFlow Federated should nest their code within
    the
    [federated research repository](https://github.com/google-research/federated)
    and potentially contributing there first. symbols contained herein. If you
    are interested in contributing, but need to validate the algorithm's design
    and performance, we recommend checking out the
    [federated research repository](https://github.com/google-research/federated)
    and potentially contributing there first.
