# TensorFlow Federated

TensorFlow Federated (TFF) is an open-source framework for machine learning and
other computations on decentralized data. TFF has been developed to facilitate
open research and experimentation with
[Federated Learning (FL)](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html),
an approach to machine learning where a shared global model is trained across
many participating clients that keep their training data locally. For example,
FL has been used to train
[prediction models for mobile keyboards](https://arxiv.org/abs/1811.03604)
without uploading sensitive typing data to servers.

TFF enables developers to use the included federated learning algorithms with
their models and data, as well as to experiment with novel algorithms. The
building blocks provided by TFF can also be used to implement non-learning
computations, such as aggregated analytics over decentralized data.

TFF's interfaces are organized in two layers:

*   [Federated Learning (FL) API](https://github.com/google-parfait/tensorflow-federated/blob/main/docs/federated_learning.md)
    The `tff.learning` layer offers a set of high-level interfaces that allow
    developers to apply the included implementations of federated training and
    evaluation to their existing TensorFlow models.

*   [Federated Core (FC) API](https://github.com/google-parfait/tensorflow-federated/blob/main/docs/federated_core.md)
    At the core of the system is a set of lower-level interfaces for concisely
    expressing novel federated algorithms by combining TensorFlow with
    distributed communication operators within a strongly-typed functional
    programming environment. This layer also serves as the foundation upon which
    we've built `tff.learning`.

TFF enables developers to declaratively express federated computations, so they
could be deployed to diverse runtime environments. Included with TFF is a
single-machine simulation runtime for experiments. Please visit the tutorials
and try it out yourself!

## Installation

See the
[install](https://github.com/google-parfait/tensorflow-federated/blob/main/docs/install.md)
documentation for instructions on how to install TensorFlow Federated as a
package or build TensorFlow Federated from source.

## Getting Started

See the
[get started](https://github.com/google-parfait/tensorflow-federated/blob/main/docs/get_started.md)
documentation for instructions on how to use TensorFlow Federated.

## Contributing

There are a number of ways to contribute depending on what you're interested in:

*   If you are interested in developing new federated learning algorithms, the
    best way to start would be to study the implementations of federated
    averaging and evaluation in `tff.learning`, and to think of extensions to
    the existing implementation (or alternative approaches). If you have a
    proposal for a new algorithm, we recommend starting by staging your project
    in the `research` directory and including a colab notebook to showcase the
    new features.

    You may want to also develop new algorithms in your own repository. We are
    happy to feature pointers to academic publications and/or repos using TFF on
    [tensorflow.org/federated](http://www.tensorflow.org/federated).

*   If you are interested in applying federated learning, consider contributing
    a tutorial, a new federated dataset, or an example model that others could
    use for experiments and testing, or writing helper classes that others can
    use in setting up simulations.

*   If you are interested in helping us improve the developer experience, the
    best way to start would be to study the implementations behind the
    `tff.learning` API, and to reflect on how we could make the code more
    streamlined. You could contribute helper classes that build upon the FC API
    or suggest extensions to the FC API itself.

*   If you are interested in helping us develop runtime infrastructure for
    simulations and beyond, please wait for a future release in which we will
    introduce interfaces and guidelines for contributing to a simulation
    infrastructure.

Please be sure to review the
[contribution](https://github.com/google-parfait/tensorflow-federated/blob/main/CONTRIBUTING.md#guidelines)
guidelines on how to contribute.

## Issues

Use
[GitHub issues](https://github.com/google-parfait/tensorflow-federated/issues)
for tracking requests and bugs.

## Questions

Please direct questions to [Stack Overflow](https://stackoverflow.com) using the
[tensorflow-federated](https://stackoverflow.com/questions/tagged/tensorflow-federated)
tag.
