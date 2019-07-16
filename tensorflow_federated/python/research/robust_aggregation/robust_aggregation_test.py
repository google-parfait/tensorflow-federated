# Lint as: python3
# TODO: copyright notice


import collections
import numpy as np
import six
from six.moves import range
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent

import tensorflow_federated as tff
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.tensorflow_libs import tensor_utils

from tensorflow_federated.python.research.robust_aggregation import build_stateless_robust_aggregation

dim = 500
num_data_points = 10

def setup_toy_data():
    rng = np.random.RandomState(0)
    data = rng.rand(num_data_points, dim).astype(np.float32)
    labels = rng.rand(num_data_points, 1).astype(np.float32)

    return [tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([('x', data[i:i+1]), ('y', labels[i:i+1])])).batch(1)
                        for i in range(data.shape[0])]


def get_model_fn():
    def create_compiled_keras_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                1, activation=None,
                kernel_initializer='zeros',
                use_bias=False, input_shape=(dim,))])

        model.compile(loss='mse', optimizer=gradient_descent.SGD(learning_rate=1e-20))
        # only to get the model to compile. We will not actually train this model
        return model

    sample_dataset = setup_toy_data()[0]
    sample_batch = tf.nest.map_structure(
        lambda x: x.numpy(), iter(sample_dataset).next()
    )
    def model_fn():
        keras_model = create_compiled_keras_model()
        return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

    return model_fn


class DummyClientComputation(optimizer_utils.ClientDeltaFn):
    """Client TensorFlow logic for example.
    Designed to mimic the class `ClientFedAvg` from federated_averaging.py
    """

    def __init__(self, model, client_weight_fn=None):
        """Creates the client computation for Federated Averaging.
        Args:
            model: A `tff.learning.TrainableModel`.
            client_weight_fn: Optional argument is ignored
        """
        del client_weight_fn
        self._model = model_utils.enhance(model)
        py_typecheck.check_type(self._model, model_utils.EnhancedTrainableModel)
        self._client_weight_fn = None

    @property
    def variables(self):
        return []

    @tf.function
    def __call__(self, dataset, initial_weights):
        del initial_weights
        model = self._model

        @tf.function
        def reduce_fn_num_examples(num_examples_sum, batch):
            """Count number of examples"""
            num_examples_in_batch = tf.shape(batch['x'])[0]
            return num_examples_sum + num_examples_in_batch

        @tf.function
        def reduce_fn_dataset_mean(sum_vector, batch):
            """Sum all the examples in the local dataset"""
            sum_batch = tf.reshape(tf.reduce_sum(batch['x'], [0]), (-1, 1))
            return sum_vector + sum_batch

        num_examples_sum = dataset.reduce(
                initial_state=tf.constant(0), reduce_func=reduce_fn_num_examples)

        example_vector_sum = dataset.reduce(
                initial_state=tf.zeros((dim, 1)), reduce_func=reduce_fn_dataset_mean)

        # create an ordered dictionary with the same type as model.trainable 
        # containing a mean of all the examples in the local dataset
        # Note: this works for a linear model only (as in the example above)
        key = list(model.weights.trainable.keys())[0]
        weights_delta = collections.OrderedDict(
                {key: example_vector_sum/tf.cast(num_examples_sum, tf.float32)}
        )

        aggregated_outputs = model.report_local_outputs()
        weights_delta, has_non_finite_delta = (
                tensor_utils.zero_all_if_any_non_finite(weights_delta)
        )

        weights_delta_weight = tf.cast(num_examples_sum, tf.float32)

        return optimizer_utils.ClientOutput(
                weights_delta, weights_delta_weight, aggregated_outputs,
                tensor_utils.to_odict({
                    'num_examples': num_examples_sum,
                    'has_non_finite_delta': has_non_finite_delta,
                }))


def build_federated_process_for_test(model_fn, num_passes=5, tolerance=1e-6):
    """ Analogue of `build_federated_averaging_process`, but with
    client_fed_avg replaced by the dummy mean computation defined above.
    """
    server_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)

    def client_fed_avg(model_fn):
        return DummyClientComputation(model_fn(), client_weight_fn=None)

    # Build robust aggregation function
    with tf.Graph().as_default():
        # workaround since keras automatically appends "_n" to the nth call of `model_fn`
        model_type = tff.framework.type_from_tensors(model_fn().weights.trainable)

    stateful_delta_aggregate_fn = build_stateless_robust_aggregation(
            model_type, num_communication_passes=num_passes, tolerance=tolerance
    )

    stateful_model_broadcast_fn = optimizer_utils.build_stateless_broadcaster()


    return optimizer_utils.build_model_delta_optimizer_process(
            model_fn, client_fed_avg, server_optimizer_fn,
            stateful_delta_aggregate_fn, stateful_model_broadcast_fn)


def get_mean(dataset):
    """Compute mean of given instance of `tf.data.dataset`."""
    mean = tf.zeros(dim)
    count = 0
    for ex in dataset:
        x = ex['x']
        mean += tf.reshape(x, [-1])
        count += 1
    return mean / count, count


def get_means_and_weights(federated_train_data):
    """Compute mean of each client's dataset and stack them into a matrix.
    Also return weights proportional to the number of datapoints in each dataset."""
    outs = [get_mean(ds) for ds in federated_train_data]
    means, counts = zip(*outs)
    weights = np.asarray(counts, dtype=np.float32) / sum(counts)
    means = np.array(means, dtype=np.float32)
    return means, weights


def aggregation_fn_np(value, weight, num_communication_passes=5, tolerance=1e-6):
    """Robust aggregation function of rows of `value` in numpy."""
    tolerance = np.float32(tolerance)
    aggr = np.average(value, axis=0, weights=weight)
    for iteration in range(num_communication_passes-1):
        aggr = np.average(
                value, axis=0,
                weights=[weight[i] /
                    np.maximum(tolerance, np.linalg.norm(aggr - value[i, :]))
                    for i in range(value.shape[0])]
        )
    return aggr


class RobustAggregationTest(tf.test.TestCase):
    def test_all(self):
        model_fn = get_model_fn()
        federated_train_data = setup_toy_data()
        means, weights = get_means_and_weights(federated_train_data)
        for num_passes in [2, 3, 5, 10]:
            for tolerance in [1e-4, 1e-6, 1e-8]:
                iterative_process = build_federated_process_for_test(model_fn, num_passes, tolerance)
                state = iterative_process.initialize()
                state, _ = iterative_process.next(state, federated_train_data)
                median_tff = state[0][0][0].reshape(-1)
                median_np = aggregation_fn_np(
                        means, weights,
                        num_communication_passes=num_passes,
                        tolerance=tolerance
                )
                print(num_passes, tolerance, tf.reduce_max(tf.abs(median_tff - median_np)))
                self.assertAllClose(
                        median_tff, median_np,
                        msg="""TFF median and np median do not agree for num_passes = {}
                        and tolerance = {}""".format(num_passes, tolerance)
                )


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()
