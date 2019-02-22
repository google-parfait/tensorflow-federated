# TensorFlow Federated

The TensorFlow Federated (TFF) platform consists of two layers:

*   [Federated Learning (FL)](federated_learning.md), high-level interfaces to
    plug existing Keras or non-Keras machine learning models into the TFF
    framework. You can perform basic tasks, such as federated training or
    evaluation, without having to study the details of federated learning
    algorithms.
*   [Federated Core (FC)](federated_core.md), lower-level interfaces to concisely
    express custom federated algorithms by combining TensorFlow with distributed
    communication operators within a strongly-typed functional programming
    environment.

Start by reading the following tutorials that walk you through the main TFF
concepts and APIs using practical examples. Make sure to follow the
[installation instructions](install.md) to configure your environment for use
with TFF.

*   [Federated Learning for image classification](tutorials/federated_learning_for_image_classification.ipynb)
    introduces the key parts of the Federated Learning (FL) API, and
    demonstrates how to use TFF to simulate federated learning on federated
    MNIST-like data.
*   [Federated Learning for text generation](tutorials/federated_learning_for_text_generation.ipynb)
    further demonstrates how to use TFF's FL API to refine a serialized
    pre-trained model for a language modeling task.
*   [Custom Federated Algorithms, Part 1: Introduction to the Federated Core](tutorials/custom_federated_algorithms_1.ipynb)
    and
    [Part 2: Implementing Federated Averaging](tutorials/custom_federated_algorithms_2.ipynb)
    introduce the key concepts and interfaces offered by the Federated Core API
    (FC API), and demonstrate how to implement a simple federated averaging
    training algorithm as well as how to perform federated evaluation.
