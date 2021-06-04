# TensorFlow Federated Tutorials

These [colab-based](https://colab.research.google.com/) tutorials walk you
through the main TFF concepts and APIs using practical examples. Reference
documentation can be found in the [TFF guides](../get_started.md).

**Getting started with federated learning**

*   [Federated Learning for image classification](federated_learning_for_image_classification.ipynb)
    introduces the key parts of the Federated Learning (FL) API, and
    demonstrates how to use TFF to simulate federated learning on federated
    MNIST-like data.
*   [Federated Learning for text generation](federated_learning_for_text_generation.ipynb)
    further demonstrates how to use TFF's FL API to refine a serialized
    pre-trained model for a language modeling task.

*   [Tuning recommended aggregations for learning](tuning_recommended_aggregators.ipynb)
    shows how the basic FL computations in `tff.learning` can be combined with
    specialized aggregation routines offering robustness, differential privacy,
    compression, and more.

**Getting started writing custom federated computations**

*   [Building Your Own Federated Learning Algorithm](building_your_own_federated_learning_algorithm.ipynb)
    shows how to use the TFF Core APIs to implement federated learning
    algorithms, using Federated Averaging as an example.

**Simulation best practices**

*   [High-performance simulations with TFF](simulations.ipynb) describes how to
    setup and configure the high performance TFF runtime.

*   [TFF simulation with accelerators (GPU)](simulations_with_accelerators.ipynb)
    shows how TFF's high-performance runtime can be used with GPUs.

*   [Working with ClientData](working_with_client_data.ipynb) gives best
    practices for integrating TFF's
    [ClientData](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/ClientData)-based
    simulation datasets into TFF computations.

**Intermediate and advanced tutorials**

*   [Implementing Custom Aggregations](custom_aggregators.ipynb) explains the
    design principles behind the `tff.aggregators` module and best practices for
    implementing custom aggregation of values from clients to server.

*   [Random noise generation](random_noise_generation.ipynb) points out some
    subtlities with using randomness in decentralized computations, and proposes
    best practices and recommend patterns.

*   [Sending Different Data To Particular Clients With tff.federated_select](federated_select.ipynb)
    introduces the `tff.federated_select` operator and gives a simple example of
    a custom federated algorithm that sends different data to different clients.

*   [Client-efficient large-model federated learning via federated_select and
    sparse aggregation](sparse_federated_learning.ipynb) shows how TFF can be
    used to train a very large model where each client device only downloads and
    updates a small part of the model, using `tff.federated_select` and sparse
    aggregation.

*   [TFF for Federated Learning Research: Model and Update Compression](tff_for_federated_learning_research_compression.ipynb)
    demonstrates how custom aggregations building on the
    [tensor_encoding API](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding)
    can be used in TFF.

*   [Custom Federated Algorithms, Part 1: Introduction to the Federated Core](custom_federated_algorithms_1.ipynb)
    and
    [Part 2: Implementing Federated Averaging](custom_federated_algorithms_2.ipynb)
    introduce the key concepts and interfaces offered by the Federated Core API
    (FC API).

*   [Experimental support for JAX in TFF](../experimental/tutorials/jax_support.ipynb)
    shows how [JAX](https://github.com/google/jax) computations can be used in
    TFF, demonstrating how TFF is designed to be able to interoperate with other
    frontend and backend ML frameworks.
