# Simple Personalization Experiment on EMNIST-62

This directory shows an example of how to use the
[`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval)
API in experiments.

## Summary

The experiment follows from the setup in paper
[Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/abs/1909.12488).
The
[EMNIST-62 dataset](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)
is an image classification dataset with 62 class labels. It has 3400 clients.
Each client has two datasets: train and test. The experiment first splits the
clients into two groups: one has 2500 clients and the other has 900 clients.

*   We use the [Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm
    to train a global model using the training examples from the first 2500
    clients. This step is done by the standard
    [`tff.learning.build_federated_averaging_process`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_federated_averaging_process)
    API.
*   Given a trained global model, we now evaluate multiple personalization
    strategies using (50 clients randomly selected from) the rest 900 clients. A
    personalization strategy defines the process of training a personalized
    model (more about personalization strategies can be found below). To
    evaluate a personalization strategy, each client trains a personalized model
    using the local training data, and then evaluate the personalized model on
    the local test data. The metrics of every personalization strategy are
    returned (more about evaluation metrics can be found below). This step is
    done by the
    [`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval)
    API.

## Personalization Strategies

A personalization strategy is a `tf.function`-decorated function that takes a
`tff.learning.Model`, an unbatched `tf.data.Dataset` for train, an unbatched
`tf.data.Dataset` for test, and an extra `context` object, trains a personalized
model, and returns the evaluation metrics. Users can define whatever
personalization strategies they like. An example of fine-tuning based
personalization strategy is given by `build_personalize_fn` in
[`p13n_utils`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/examples/personalization/p13n_utils.py).

The
[`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval)
API allows users to evaluate multiple personalization strategies at the same
time, starting from the same global model. Specifically, users define a
`collections.OrderedDict` that maps string names to personalization strategies,
and then pass it to the API as the `personalize_fn_dict` argument.

In
[our experiment](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/examples/personalization/emnist_p13n_main.py),
we define and evaluate two fine-tuning based personalization strategies: one
uses SGD and other other uses Adam optimizer.

## Returned Metrics

The
[`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval)
API has an argument `max_num_samples` with a default value 100. Metrics from at
most `max_num_samples` clients (the clients are sampled without replacement)
will be collected and returned. In
[our experiment](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/examples/personalization/emnist_p13n_main.py),
we set this value to be larger than the number of clients in the federated
dataset, which means that metrics from all clients will be returned.

The returned metrics is a `collections.OrderedDict`. It maps the key
'baseline_metrics' to the evaluation metrics of the initial model, and maps
strategy names to the evaluation metrics of the corresponding personalization
strategies. Each returned metric contains a list of scalars (each scalar comes
from one sampled client). Metric values at the same position, e.g.,
`metric_1[i]`, `metric_2[i]`, ..., corresponds to the same client. In
[our experiment](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/examples/personalization/emnist_p13n_main.py),
the baseline evaluation function is given by `evaluate_fn` in
[`p13n_utils`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/examples/personalization/p13n_utils.py),
which is the same evaluation function used by the personalization strategies.
