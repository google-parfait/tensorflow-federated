<!-- Please keep the content of this file in sync with docs/_index.yaml -->

# TensorFlow Federated

TensorFlow Federated (TFF) is an open-source framework for collaborative
computations on distributed data that does not require collecting data at a
centralized location.

The framework has initially been developed to facilitate open research and
experimentation with
[Federated Learning] (https://ai.googleblog.com/2017/04/federated-learning-collaborative.html),
a technology that enables devices owned by end users to collaboratively learn a
shared prediction model while keeping potentially sensitive training data on the
devices, thus decoupling the ability to do machine learning from the need to
collect and store the data in the cloud.

With the interfaces provided by TFF, developers can test existing federated
learning algorithms on their models and data, or design new experimental
algorithms and run them on existing models and data, all within the same open
source environment. The framework has been designed with compositionality in
mind, and can be used to combine independently-developed techniques and
components that offer complementary capabilities into larger systems.

Beyond this, TFF also provides a set of building blocks that can be used to
implement a variety of custom non-learning algorithms, such as analytics over
sensitive distributed on-device data.

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
