# TensorFlow Federated

Welcome!

TFF consists of two layers described in dedicated guide pages:

* [Federated Learning (FL)](federated_learning.md), high-level interfaces that
  allow users to plug existing Keras or non-Keras machine learning models into
  the framework, and perform basic tasks, such as federated training or
  evaluation, without having to study the details of federated learning
  algorithms.

* [Federated Core (FC)](federated_core.md), lower-level interfaces that allow
  users to concisely express their own custom federated algorithms by combining
  TensorFlow with distributed communication operators within a strongly-typed
  functional programming environment.

We recommend that you start your journey by reviewing the tutorials listed
below, as they walk you through the main concepts and APIs offered by TFF
using practical examples. Before you start, make sure to
follow the [installation instructions](install.md) to configure your
environment for use with TFF.

* [Federated Learning for Image Classification]
  (tutorials/federated_learning_for_image_classification.ipynb) introduces
  some the key parts of the Federated Learning (FL) API, and demonstrates how
  to use TFF to simulate federated learning on federated MNIST-like data.

* [Federated Learning for Text Generation]
  (tutorials/federated_learning_for_text_generation.ipynb) further demonstrates
  how to use TFF's FL API to refine a serialized pre-trained model for a
  language modeling task.

* [Custom Federated Algorithms with the Federated Core API]
  (tutorials/custom_federated_algorithms.ipynb) introduces the key concepts and
  interfaces offered by the Federated Core API (FC API), and demonstrates how to
  implement a simple federated averating training algorithm as well as how to
  perform federated evaluation.
