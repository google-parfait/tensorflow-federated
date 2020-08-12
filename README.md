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

*   [Federated Learning (FL) API](docs/federated_learning.md) The
    `tff.learning` layer offers a set of high-level interfaces that allow
    developers to apply the included implementations of federated training and
    evaluation to their existing TensorFlow models.

*   [Federated Core (FC) API](docs/federated_core.md) At the core of the system
    is a set of lower-level interfaces for concisely expressing novel federated
    algorithms by combining TensorFlow with distributed communication operators
    within a strongly-typed functional programming environment. This layer also
    serves as the foundation upon which we've built `tff.learning`.

TFF enables developers to declaratively express federated computations, so they
could be deployed to diverse runtime environments. Included with TFF is a
single-machine simulation runtime for experiments. Please visit the tutorials
and try it out yourself!

## Installation

See the [install](docs/install.md) documentation for instructions on how to
install TensorFlow Federated as a package or build TensorFlow Federated from
source.

## Getting Started

See the [get started](docs/get_started.md) documentation for instructions on
how to use TensorFlow Federated.

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
[contribution guidelines](CONTRIBUTING.md#code-style-guidelines-and-best-practices)
for guidelines on how to contribute.

## Compatibility

### TensorFlow

The following table describes the compatibility between the TensorFlow Federated
and TensorFlow Python packages, meaning that a version of the TensorFlow
Federated package was tested against a version of the TensorFlow before it was
released. It is possible that a newer version of the TensorFlow Federated
package will not work with an older version of the TensorFlow package and vice
versa.

TensorFlow Federated                                                  | TensorFlow
--------------------------------------------------------------------- | ----------
[0.16.1](https://github.com/tensorflow/federated/tree/v0.16.1)        | [tensorflow 2.2.0](https://pypi.org/project/tensorflow/2.2.0/)
[0.16.0](https://github.com/tensorflow/federated/tree/v0.16.0)        | [tensorflow 2.2.0](https://pypi.org/project/tensorflow/2.2.0/)
[0.15.0](https://github.com/tensorflow/federated/tree/v0.15.0)        | [tensorflow 2.2.0](https://pypi.org/project/tensorflow/2.2.0/)
[0.14.0](https://github.com/tensorflow/federated/tree/v0.14.0)        | [tensorflow 2.2.0](https://pypi.org/project/tensorflow/2.2.0/)
[0.13.1](https://github.com/tensorflow/federated/tree/v0.13.1)        | [tensorflow 2.1.0](https://pypi.org/project/tensorflow/2.1.0/)
[0.13.0](https://github.com/tensorflow/federated/tree/v0.13.0)        | [tensorflow 2.1.0](https://pypi.org/project/tensorflow/2.1.0/)
[0.12.0](https://github.com/tensorflow/federated/tree/v0.12.0)        | [tensorflow 2.1.0](https://pypi.org/project/tensorflow/2.1.0/)
[0.11.0](https://github.com/tensorflow/federated/tree/v0.11.0)        | [tensorflow 2.0.0](https://pypi.org/project/tensorflow/2.0.0/)
[0.10.1](https://github.com/tensorflow/federated/tree/v0.10.1)        | [tensorflow 2.0.0](https://pypi.org/project/tensorflow/2.0.0/)
[0.10.0](https://github.com/tensorflow/federated/tree/v0.10.0)        | [tensorflow 2.0.0](https://pypi.org/project/tensorflow/2.0.0/)
[0.9.0](https://github.com/tensorflow/federated/tree/v0.9.0)          | [tf-nightly 2.1.0.dev20191005](https://pypi.org/project/tf-nightly/2.1.0.dev20191005/)
~~[0.8.0](https://github.com/tensorflow/federated/tree/v0.8.0)~~ <sup>[1](#footnote1)</sup> | [tf-nightly 1.15.0.dev20190805](https://pypi.org/project/tf-nightly/1.15.0.dev20190805/)
[0.7.0](https://github.com/tensorflow/federated/tree/v0.7.0)          | [tf-nightly 1.15.0.dev20190711](https://pypi.org/project/tf-nightly/1.15.0.dev20190711/)
[0.6.0](https://github.com/tensorflow/federated/tree/v0.6.0)          | [tf-nightly 1.15.0.dev20190626](https://pypi.org/project/tf-nightly/1.15.0.dev20190626/)
[0.5.0](https://github.com/tensorflow/federated/tree/v0.5.0)          | [tf-nightly 1.14.1.dev20190528](https://pypi.org/project/tf-nightly/1.14.1.dev20190528/)
[0.4.0](https://github.com/tensorflow/federated/tree/v0.4.0)          | [tensorflow 1.13.1](https://pypi.org/project/tensorflow/1.13.1)
[0.3.0](https://github.com/tensorflow/federated/tree/v0.3.0)          | [tensorflow 1.13.1](https://pypi.org/project/tensorflow/1.13.1)
[0.2.0](https://github.com/tensorflow/federated/tree/v0.2.0)          | [tensorflow 1.13.1](https://pypi.org/project/tensorflow/1.13.1)
[0.1.0](https://github.com/tensorflow/federated/tree/v0.1.0)          | [tensorflow 1.13.0rc2](https://pypi.org/project/tensorflow/1.13.0rc0/)

### Python

See the `Programming Language` classifiers on the PyPI
[tensorflow-federated](https://pypi.org/project/tensorflow-federated/) project
for information about the compatibility between TensorFlow Federated and Python
versions.

## Issues

Use [GitHub issues](https://github.com/tensorflow/federated/issues) for tracking
requests and bugs.

## Questions

Please direct questions to [Stack Overflow](https://stackoverflow.com) using the
[tensorflow-federated](https://stackoverflow.com/questions/tagged/tensorflow-federated)
tag.

<sup id="footnote1">1</sup> TensorFlow Federated
    [0.8.0](https://github.com/tensorflow/federated/tree/v0.8.0) fails to pip
    install `pip install tensorflow_federated==0.8.0` because it requires
    `tf-nightly==1.15.0.dev20190805`, which is no longer available.
