# Lint as: python3
# Copyright 2020, The TensorFlow Federated Authors.
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

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning import personalization_eval as p13n_eval


@tf.function
def _evaluate_fn(model, dataset):
  """Evaluates a `tff.learning.Model` on the given dataset."""
  # Reset the local variables so that the returned metrics are computed using
  # the given data. Similar to the `reset_states` method of `tf.metrics.Metric`.
  for var in model.local_variables:
    if var.initial_value is not None:
      var.assign(var.initial_value)
    else:
      var.assign(tf.zeros_like(var))

  def eval_fn(dummy_state, batch):
    """Evaluates the model on a batch."""
    model.forward_pass(batch, training=False)
    return dummy_state

  # Evaluate on the dataset.
  dataset.reduce(initial_state=0, reduce_func=eval_fn)

  # Obtain the metrics.
  results = collections.OrderedDict()
  local_outputs = model.report_local_outputs()
  for name, metric in local_outputs.items():
    if isinstance(metric, list) and (len(metric) == 2):
      # Some metrics returned by `report_local_outputs()` can have two scalars:
      # one represents `sum`, and the other represents `count`. Ideally, we want
      # to return a single scalar for each metric.
      results[name] = metric[0] / metric[1]
    else:
      results[name] = metric[0] if isinstance(metric, list) else metric
  return results


def _build_personalize_fn(optimizer_fn):
  """Builds a personalization function given an optimizer constructor."""
  optimizer = optimizer_fn()

  @tf.function
  def personalize_fn(model, train_data, test_data, context=None):

    def train_fn(num_examples_sum, batch):
      """Runs gradient descent on a batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)

      grads = tape.gradient(output.loss, model.trainable_variables)
      optimizer.apply_gradients(
          zip(
              tf.nest.flatten(grads),
              tf.nest.flatten(model.trainable_variables)))
      return num_examples_sum + output.num_examples

    # Train a personalized model.
    num_examples_sum = train_data.reduce(initial_state=0, reduce_func=train_fn)

    # For test coverage, this example uses an optional `int32` as `context`.
    if context is not None:
      num_examples_sum = num_examples_sum + context

    results = collections.OrderedDict()
    results['num_examples'] = num_examples_sum
    results['test_outputs'] = _evaluate_fn(model, test_data)
    return results

  return personalize_fn


def _create_p13n_fn_dict(learning_rate):
  """Creates a dictionary containing two personalization strategies."""
  p13n_fn_dict = collections.OrderedDict()

  adam_opt_fn = lambda: tf.keras.optimizers.Adam(learning_rate=learning_rate)
  p13n_fn_dict['adam_opt'] = lambda: _build_personalize_fn(adam_opt_fn)

  sgd_opt_fn = lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)
  p13n_fn_dict['sgd_opt'] = lambda: _build_personalize_fn(sgd_opt_fn)

  return p13n_fn_dict


def _create_dataset(batch_size):
  """Constructs a batched dataset with three datapoints."""
  ds = collections.OrderedDict([('x', [[-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]),
                                ('y', [[1.0], [1.0], [1.0]])])
  # Note: batching is needed here as it creates the required batch dimension.
  # The batch size can be re-set (by `unbatch()` first) in personalization.
  return tf.data.Dataset.from_tensor_slices(ds).batch(batch_size)


def _create_client_input(train_batch_size, test_batch_size, context=None):
  """Constructs client datasets for personalization."""
  client_input = collections.OrderedDict()
  client_input['train_data'] = _create_dataset(train_batch_size)
  client_input['test_data'] = _create_dataset(test_batch_size)
  if context is not None:
    client_input['context'] = context
  return client_input


def _create_zero_model_weights(model_fn):
  """Creates the model weights with all zeros."""
  dummy_model = model_utils.enhance(model_fn())
  return tf.nest.map_structure(tf.zeros_like, dummy_model.weights)


class PersonalizationEvalTest(test.TestCase):

  def test_failure_with_invalid_model_fn(self):
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)
    with self.assertRaises(TypeError):
      # `model_fn` should be a callable.
      bad_model_fn = 6
      p13n_eval.build_personalization_eval(bad_model_fn, p13n_fn_dict,
                                           _evaluate_fn)

    with self.assertRaises(TypeError):
      # `model_fn` should be a callable that returns a `tff.learning.Model`.
      bad_model_fn = lambda: 6
      p13n_eval.build_personalization_eval(bad_model_fn, p13n_fn_dict,
                                           _evaluate_fn)

  def test_failure_with_invalid_p13n_fns(self):

    def model_fn():
      return model_examples.LinearRegression(feature_dim=2)

    with self.assertRaises(TypeError):
      # `personalize_fn_dict` should be a `OrderedDict`.
      bad_p13n_fn_dict = {'a': 6}
      p13n_eval.build_personalization_eval(model_fn, bad_p13n_fn_dict,
                                           _evaluate_fn)

    with self.assertRaises(TypeError):
      # `personalize_fn_dict` should be a `OrderedDict` that maps a `string` to
      # a `callable`.
      bad_p13n_fn_dict = collections.OrderedDict([('a', 6)])
      p13n_eval.build_personalization_eval(model_fn, bad_p13n_fn_dict,
                                           _evaluate_fn)

    with self.assertRaises(TypeError):
      # `personalize_fn_dict` should be a `OrderedDict` that maps a `string` to
      # a `callable` that when called, gives another `callable`.
      bad_p13n_fn_dict = collections.OrderedDict([('a', lambda: 2)])
      p13n_eval.build_personalization_eval(model_fn, bad_p13n_fn_dict,
                                           _evaluate_fn)

    with self.assertRaises(ValueError):
      # `personalize_fn_dict` should not use `baseline_metrics` as a key.
      bad_p13n_fn_dict = collections.OrderedDict([('baseline_metrics',
                                                   lambda: 2)])
      p13n_eval.build_personalization_eval(model_fn, bad_p13n_fn_dict,
                                           _evaluate_fn)

  def test_failure_with_invalid_baseline_eval_fn(self):

    def model_fn():
      return model_examples.LinearRegression(feature_dim=2)

    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)

    with self.assertRaises(TypeError):
      # `baseline_evaluate_fn` should be a callable.
      bad_baseline_evaluate_fn = 6
      p13n_eval.build_personalization_eval(model_fn, p13n_fn_dict,
                                           bad_baseline_evaluate_fn)

  def test_success_with_directly_constructed_model(self):

    def model_fn():
      return model_examples.LinearRegression(feature_dim=2)

    zero_model_weights = _create_zero_model_weights(model_fn)
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)

    federated_p13n_eval = p13n_eval.build_personalization_eval(
        model_fn, p13n_fn_dict, _evaluate_fn)

    # Perform p13n eval on two clients with different batch sizes.
    results = federated_p13n_eval(
        zero_model_weights,
        [_create_client_input(1, 1),
         _create_client_input(2, 3)])
    results = results._asdict(recursive=True)

    # Check if the baseline metrics are correct.
    baseline_metrics = results['baseline_metrics']
    # Average loss is 0.5 * (1 + 1 + 1)/3 = 0.5.
    self.assertAllEqual(baseline_metrics['loss'], [0.5, 0.5])
    # Number of test examples is 3 for both clients.
    self.assertAllEqual(baseline_metrics['num_examples'], [3, 3])
    # Number of test batches is 3 and 1.
    # Note: the order is not preserved due to `federated_sample`.
    self.assertAllEqual(sorted(baseline_metrics['num_batches']), [1, 3])
    if baseline_metrics['num_batches'][0] == 3:
      client_1_idx, client_2_idx = 0, 1
    else:
      client_1_idx, client_2_idx = 1, 0

    # Check if the metrics of `sgd_opt` are correct.
    sgd_metrics = results['sgd_opt']
    # Number of training examples is 3 for both clients.
    self.assertAllEqual(sgd_metrics['num_examples'], [3, 3])
    sgd_test_outputs = sgd_metrics['test_outputs']
    # Number of test examples is also 3 for both clients.
    self.assertAllEqual(sgd_test_outputs['num_examples'], [3, 3])
    # Client 1's weights become [-3, -3, -1], which gives average loss 24.
    # Client 2's weights become [0, 0, 1], which gives average loss 0.
    self.assertAlmostEqual(sgd_test_outputs['loss'][client_1_idx], 24.0)
    self.assertAlmostEqual(sgd_test_outputs['loss'][client_2_idx], 0.0)
    # Number of test batches should have the same order as baseline metrics.
    self.assertAllEqual(sgd_test_outputs['num_batches'],
                        baseline_metrics['num_batches'])

    # Check if the metrics of `adam_opt` are correct.
    adam_metrics = results['adam_opt']
    # Number of training examples is 3 for both clients.
    self.assertAllEqual(adam_metrics['num_examples'], [3, 3])
    adam_test_outputs = adam_metrics['test_outputs']
    # Number of test examples is also 3 for both clients.
    self.assertAllEqual(adam_test_outputs['num_examples'], [3, 3])
    # Number of test batches should have the same order as baseline metrics.
    self.assertAllEqual(adam_test_outputs['num_batches'],
                        baseline_metrics['num_batches'])

  def test_success_with_model_constructed_from_keras(self):

    def model_fn():
      inputs = tf.keras.Input(shape=(2,))  # feature dim = 2
      outputs = tf.keras.layers.Dense(1)(inputs)
      keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
      dummy_batch = collections.OrderedDict([
          ('x', np.zeros([1, 2], dtype=np.float32)),
          ('y', np.zeros([1, 1], dtype=np.float32))
      ])
      return keras_utils.from_keras_model(keras_model, dummy_batch,
                                          tf.keras.losses.MeanSquaredError())

    zero_model_weights = _create_zero_model_weights(model_fn)
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=0.5)

    federated_p13n_eval = p13n_eval.build_personalization_eval(
        model_fn, p13n_fn_dict, _evaluate_fn)

    # Perform p13n eval on two clients with different batch sizes.
    results = federated_p13n_eval(
        zero_model_weights,
        [_create_client_input(1, 1),
         _create_client_input(2, 3)])
    results = results._asdict(recursive=True)

    # Check if the baseline metrics are correct.
    baseline_metrics = results['baseline_metrics']
    # MeanSquredError(MSE) is (1 + 1 + 1)/3 = 1.0.
    self.assertAllEqual(baseline_metrics['loss'], [1.0, 1.0])

    # Check if the metrics of `sgd_opt` are correct.
    sgd_metrics = results['sgd_opt']
    # Number of training examples is 3 for both clients.
    self.assertAllEqual(sgd_metrics['num_examples'], [3, 3])
    sgd_test_outputs = sgd_metrics['test_outputs']
    # Client 1's weights become [-3, -3, -1], which gives MSE 48.
    # Client 2's weights become [0, 0, 1], which gives MSE 0.
    self.assertAlmostEqual(sorted(sgd_test_outputs['loss']), [0.0, 48.0])

    # Check if the metrics of `adam_opt` are correct.
    adam_metrics = results['adam_opt']
    # Number of training examples is 3 for both clients.
    self.assertAllEqual(adam_metrics['num_examples'], [3, 3])

  def test_failure_with_invalid_context_type(self):

    def model_fn():
      return model_examples.LinearRegression(feature_dim=2)

    zero_model_weights = _create_zero_model_weights(model_fn)
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)

    with self.assertRaises(TypeError):
      # `tf.int32` is not a `tff.Type`.
      bad_context_tff_type = tf.int32
      federated_p13n_eval = p13n_eval.build_personalization_eval(
          model_fn,
          p13n_fn_dict,
          _evaluate_fn,
          context_tff_type=bad_context_tff_type)

    with self.assertRaises(TypeError):
      # `context_tff_type` is provided but `context` is not provided.
      context_tff_type = tff.to_type(tf.int32)
      federated_p13n_eval = p13n_eval.build_personalization_eval(
          model_fn,
          p13n_fn_dict,
          _evaluate_fn,
          context_tff_type=context_tff_type)
      federated_p13n_eval(zero_model_weights, [
          _create_client_input(1, 1, context=None),
          _create_client_input(2, 3, context=None)
      ])

  def test_success_with_valid_context(self):

    def model_fn():
      return model_examples.LinearRegression(feature_dim=2)

    zero_model_weights = _create_zero_model_weights(model_fn)
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)

    # Build the p13n eval with an extra `context` argument.
    context_tff_type = tff.to_type(tf.int32)
    federated_p13n_eval = p13n_eval.build_personalization_eval(
        model_fn, p13n_fn_dict, _evaluate_fn, context_tff_type=context_tff_type)

    # Perform p13n eval on two clients with different `context` values.
    results = federated_p13n_eval(zero_model_weights, [
        _create_client_input(1, 1, context=2),
        _create_client_input(2, 3, context=5)
    ])
    results = results._asdict(recursive=True)

    sgd_metrics = results['sgd_opt']
    adam_metrics = results['adam_opt']

    # Number of training examples is `3 + context` for both clients.
    # Note: the order is not preserved due to `federated_sample`, but the order
    # should be consistent across different personalization strategies.
    self.assertAllEqual(sorted(sgd_metrics['num_examples']), [5, 8])
    self.assertAllEqual(sgd_metrics['num_examples'],
                        adam_metrics['num_examples'])

  def test_failure_with_invalid_sample_size(self):

    def model_fn():
      return model_examples.LinearRegression(feature_dim=2)

    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)

    with self.assertRaises(TypeError):
      # `max_num_samples` should be an `int`.
      bad_num_samples = 1.0
      p13n_eval.build_personalization_eval(
          model_fn, p13n_fn_dict, _evaluate_fn, max_num_samples=bad_num_samples)

    with self.assertRaises(ValueError):
      # `max_num_samples` should be a positive `int`.
      bad_num_samples = 0
      p13n_eval.build_personalization_eval(
          model_fn, p13n_fn_dict, _evaluate_fn, max_num_samples=bad_num_samples)

  def test_success_with_small_sample_size(self):

    def model_fn():
      return model_examples.LinearRegression(feature_dim=2)

    zero_model_weights = _create_zero_model_weights(model_fn)
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)

    federated_p13n_eval = p13n_eval.build_personalization_eval(
        model_fn, p13n_fn_dict, _evaluate_fn, max_num_samples=1)

    # Perform p13n eval on two clients with different batch sizes.
    results = federated_p13n_eval(
        zero_model_weights,
        [_create_client_input(1, 1),
         _create_client_input(2, 3)])
    results = results._asdict(recursive=True)

    # The results should only contain metrics from one client.
    self.assertAllEqual(len(results['baseline_metrics']['loss']), 1)
    self.assertAllEqual(len(results['sgd_opt']['test_outputs']['loss']), 1)
    self.assertAllEqual(len(results['adam_opt']['test_outputs']['loss']), 1)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  test.main()
