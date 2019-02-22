<!-- Please keep the content of this file in sync with docs/_index.yaml -->

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

* [Federated Learning (FL) API](docs/federated_learning.md)
  The `tff.learning` layer offers a set of high-level interfaces that allow
  developers to apply the included implementations of federated training and
  evaluation to their existing TensorFlow models.

* [Federated Core (FC) API](docs/federated_core.md)
  At the core of the system is a set of lower-level interfaces for concisely
  expressing novel federated algorithms by combining TensorFlow with distributed
  communication operators within a strongly-typed functional programming
  environment. This layer also serves as the foundation upon which we've built
  `tff.learning`.

TFF enables developers to declaratively express federated computations, so they
could be deployed to diverse runtime environments. Included with TFF is a
single-machine simulation runtime for experiments. Please visit the
tutorials and try it out yourself!

## Installation

See the [install](docs/install.md) documentation for instructions on how to
install TensorFlow Federated as a package or build TensorFlow Federated from
source.

## Getting Started

See the [get started](docs/get_started.md) documentation for instructions on
how to use TensorFlow Federated.

The
[Code Style, Guidelines, and Best Practice](CONTRIBUTING.md#code-style-guidelines-and-best-practices)
for developers may also be useful to review.

## Contributing

If you want to contribute to TensorFlow Federated, be sure to review the
[contribution guidelines](CONTRIBUTING.md).

## Issues

Use [GitHub issues](https://github.com/tensorflow/federated/issues) for tracking
requests and bugs.

## Questions

Please direct questions to [Stack Overflow](https://stackoverflow.com) using the
[tensorflow-federated](https://stackoverflow.com/questions/tagged/tensorflow-federated)
tag.
