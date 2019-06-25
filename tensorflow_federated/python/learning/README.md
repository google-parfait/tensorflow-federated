# TensorFlow Federated Learning

This document contains brief guidelines on the organization of this component,
to evolve incrementally as new capabilities are added.

## Target Audience and Interface Surfaces

The interfaces and code in the `learning` component may be consumed by three
classes of users:

*   **Modelers**. Users who will supply their own machine learning models, and
    use federated learning interfaces to train them on their data. These users
    should only depend on the public API surface that consist of a short list of
    curated symbols in `__init__.py` and create a build dependency only on the
    `learning` library.

    The symbols included in this public API surface are documented in the
    automatically generated Python documentation in `api_docs`. Typically, a
    user will wrap their ML model as a subclass of `tff.learning.Model`, and
    then invoke one of the constructors listed in the API to create one or more
    `tff.Computation`s that implement pre-defined federated tasks (such as
    training or evaluation), and that can be invoked to run on their data sets.

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

    TODO(b/117177413): Create the `research` directory and update the pointer to
    a more specific location.

*   **TensorFlow Federated Contributors**. Users who will contribute federated
    learning algorithms to TensorFlow Federated should nest their code within
    the `learning` directory, and may link directly against all the modules and
    symbols contained herein.
