# Copyright 2019, Krishna Pillutla and Sham M. Kakade and Zaid Harchaoui.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.research.robust_aggregation import robust_federated_aggregation as rfa
from tensorflow_federated.python.tensorflow_libs import tensor_utils

DIM = 500
NUM_DATA_POINTS = 10


def setup_toy_data():
  rng = np.random.RandomState(0)
  data = rng.rand(NUM_DATA_POINTS, DIM).astype(np.float32)
  labels = rng.rand(NUM_DATA_POINTS, 1).astype(np.float32)

  def build_dataset(i):
    return tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(x=data[i:i + 1], y=labels[i:i + 1])).batch(1)

  return [build_dataset(i) for i in range(data.shape[0])]


def get_model_fn():
  """Return a function which creates a TFF model."""
  sample_dataset = setup_toy_data()[0]

  def model_fn():
    keras_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(DIM,)),
        tf.keras.layers.Dense(1, kernel_initializer='zeros', use_bias=False)
    ])
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=sample_dataset.element_spec,
        loss=tf.keras.losses.MeanSquaredError())

  return model_fn


class DummyClientComputation(tff.learning.framework.ClientDeltaFn):
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
    self._model = tff.learning.framework.enhance(model)
    py_typecheck.check_type(self._model, tff.learning.framework.EnhancedModel)
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
      """Count number of examples."""
      num_examples_in_batch = tf.shape(batch['x'])[0]
      return num_examples_sum + num_examples_in_batch

    @tf.function
    def reduce_fn_dataset_mean(sum_vector, batch):
      """Sum all the examples in the local dataset."""
      sum_batch = tf.reshape(tf.reduce_sum(batch['x'], [0]), (-1, 1))
      return sum_vector + sum_batch

    num_examples_sum = dataset.reduce(
        initial_state=tf.constant(0), reduce_func=reduce_fn_num_examples)
    example_vector_sum = dataset.reduce(
        initial_state=tf.zeros((DIM, 1)), reduce_func=reduce_fn_dataset_mean)

    # create a list with the same structure and type as model.trainable
    # containing a mean of all the examples in the local dataset. Note: this
    # works for a linear model only (as in the example above)
    weights_delta = [example_vector_sum / tf.cast(num_examples_sum, tf.float32)]
    aggregated_outputs = model.report_local_outputs()
    weights_delta, has_non_finite_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    weights_delta_weight = tf.cast(num_examples_sum, tf.float32)

    return tff.learning.framework.ClientOutput(
        weights_delta, weights_delta_weight, aggregated_outputs,
        collections.OrderedDict(
            num_examples=num_examples_sum,
            has_non_finite_delta=has_non_finite_delta,
        ))


def build_federated_process_for_test(model_fn, num_passes=5, tolerance=1e-6):
  """Build a test FedAvg process with a dummy client computation.

  Analogue of `build_federated_averaging_process`, but with client_fed_avg
  replaced by the dummy mean computation defined above.

  Args:
    model_fn: callable that returns a `tff.learning.Model`.
    num_passes: integer number  of communication rounds in the smoothed
      Weiszfeld algorithm (min. 1).
    tolerance: float smoothing parameter of smoothed Weiszfeld algorithm.
      Default 1e-6.

  Returns:
    A `tff.templates.IterativeProcess`.
  """
  server_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)

  def client_fed_avg(model_fn):
    return DummyClientComputation(model_fn(), client_weight_fn=None)

  # Build robust aggregation function
  with tf.Graph().as_default():
    # workaround since keras automatically appends "_n" to the nth call of
    # `model_fn`
    model_type = tff.framework.type_from_tensors(model_fn().weights.trainable)

    stateful_delta_aggregate_fn = rfa.build_stateless_robust_aggregation(
        model_type, num_communication_passes=num_passes, tolerance=tolerance)

    return tff.learning.framework.build_model_delta_optimizer_process(
        model_fn, client_fed_avg, server_optimizer_fn,
        stateful_delta_aggregate_fn)


def get_mean(dataset):
  """Compute mean of given instance of `tf.data.dataset`."""
  mean = tf.zeros(DIM)
  count = 0
  for ex in dataset:
    x = ex['x']
    mean += tf.reshape(x, [-1])
    count += 1
  return mean / count, count


def get_means_and_weights(federated_train_data):
  """Return mean of each client's dataset and weight for each client."""
  outs = [get_mean(ds) for ds in federated_train_data]
  means, counts = list(zip(*outs))
  weights = np.asarray(counts, dtype=np.float32) / sum(counts)
  means = np.array(means, dtype=np.float32)
  return means, weights


def aggregation_fn_np(value,
                      weight,
                      num_communication_passes=5,
                      tolerance=1e-6):
  """Robust aggregation function of rows of `value` in numpy."""
  tolerance = np.float32(tolerance)
  aggr = np.average(value, axis=0, weights=weight)
  for _ in range(num_communication_passes - 1):
    aggr = np.average(
        value,
        axis=0,
        weights=[
            weight[i] /
            np.maximum(tolerance, np.linalg.norm(aggr - value[i, :]))
            for i in range(value.shape[0])
        ])
  return aggr


class RobustAggregationTest(tf.test.TestCase):
  """Class to test Robust Aggregation."""

  def test_all(self):
    """Main test for Robust Aggregation."""
    model_fn = get_model_fn()
    federated_train_data = setup_toy_data()
    means, weights = get_means_and_weights(federated_train_data)
    for num_passes in [3, 5]:
      for tolerance in [1e-4, 1e-6]:
        iterative_process = build_federated_process_for_test(
            model_fn, num_passes, tolerance)
        state = iterative_process.initialize()
        state, _ = iterative_process.next(state, federated_train_data)
        median_tff = state[0][0][0].reshape(-1)
        median_np = aggregation_fn_np(
            means,
            weights,
            num_communication_passes=num_passes,
            tolerance=tolerance)
        self.assertAllClose(
            median_tff,
            median_np,
            msg="""TFF median and np median do not agree for num_passes = {}
            and tolerance = {}""".format(num_passes, tolerance))


if __name__ == '__main__':
  tf.test.main()
