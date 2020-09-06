# Simple Personalization Experiment on EMNIST-62

Note: This directory is a work-in-progress.

## Summary

This directory provides a simple stand-alone example of using
[`tff.learning.build_personalization_eval`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/personalization_eval.py)
API on the EMNIST-62 dataset. The experimental setting follows from the paper
[Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/abs/1909.12488):

*   The first 2500 clients are used to train a global model via the standard
    [Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm.
*   The rest 900 clients then fine-tune the global model locally using their
    local training sets.

The
[`tff.learning.build_personalization_eval`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/personalization_eval.py)
API allows users to evaluate multiple fine-tuning methods (aka personalization
strategies) at the same time, starting from the same global model. Specifically,

*   Multiple personalization strategies are represented as a single
    `OrderedDict`, which is passed to the API via the `personalize_fn_dict`
    argument.
*   A personalization strategy is represented as a `tf.function` that takes a
    `tff.learning.Model`, an unbatched `tf.data.Dataset` for train, an unbatched
    `tf.data.Dataset` for test, and an extra `context` object, trains a
    personalized model, and returns the evaluation metrics. Users can define
    their own personalization strategies. An example is given by
    `build_personalize_fn` in the file
    [p13n_utils.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated_research/personalization/p13n_utils.py).
