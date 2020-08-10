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

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning import personalization_eval as p13n_eval


@tf.function
def _evaluate_fn(model, dataset, batch_size=1):
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
  batched_dataset = dataset.batch(batch_size)
  batched_dataset.reduce(initial_state=0, reduce_func=eval_fn)

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


def _build_personalize_fn(optimizer_fn, train_batch_size, test_batch_size):
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
    batched_train_data = train_data.batch(train_batch_size)
    num_examples_sum = batched_train_data.reduce(
        initial_state=0, reduce_func=train_fn)

    # For test coverage, this example uses an optional `int32` as `context`.
    if context is not None:
      num_examples_sum = num_examples_sum + context

    results = collections.OrderedDict()
    results['num_examples'] = num_examples_sum
    results['test_outputs'] = _evaluate_fn(model, test_data, test_batch_size)
    return results

  return personalize_fn


def _create_p13n_fn_dict(learning_rate):
  """Creates a dictionary containing two personalization strategies."""
  p13n_fn_dict = collections.OrderedDict()

  opt_fn = lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)
  # The two personalization strategies use different training batch sizes.
  p13n_fn_dict['batch_size_1'] = lambda: _build_personalize_fn(opt_fn, 1, 3)
  p13n_fn_dict['batch_size_2'] = lambda: _build_personalize_fn(opt_fn, 2, 3)

  return p13n_fn_dict


def _create_dataset(scale):
  """Constructs a dataset with three datapoints."""
  x = np.array([[-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]) * scale
  y = np.array([[1.0], [1.0], [1.0]]) * scale
  ds = collections.OrderedDict([('x', x.astype(np.float32)),
                                ('y', y.astype(np.float32))])
  # Note: batching is not needed here as the preprocessing of dataset is done
  # inside the personalization function.
  return tf.data.Dataset.from_tensor_slices(ds)


def _create_client_input(train_scale, test_scale, context=None):
  """Constructs client datasets for personalization."""
  client_input = collections.OrderedDict()
  client_input['train_data'] = _create_dataset(train_scale)
  client_input['test_data'] = _create_dataset(test_scale)
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

    # Perform p13n eval on two clients: their train data are equivalent, but the
    # test data have different scales.
    results = federated_p13n_eval(zero_model_weights, [
        _create_client_input(train_scale=1.0, test_scale=1.0),
        _create_client_input(train_scale=1.0, test_scale=2.0)
    ])
    results = results._asdict(recursive=True)

    # Check if the baseline metrics are correct.
    baseline_metrics = results['baseline_metrics']
    # Number of test examples is 3 for both clients.
    self.assertAllEqual(baseline_metrics['num_examples'], [3, 3])
    # Number of test batches is 3 for both clients, because the function that
    # evaluates the baseline metrics `_evaluate_fn` uses a default batch size 1.
    self.assertAllEqual(sorted(baseline_metrics['num_batches']), [3, 3])
    # The initial weights are all zeros. The average loss can be computed as:
    # Client 1, 0.5*(1 + 1 + 1)/3 = 0.5; Client 2, 0.5*(4 + 4 + 4)/3 = 2.0.
    # Note: the order is not preserved due to `federated_sample`.
    self.assertAllEqual(sorted(baseline_metrics['loss']), [0.5, 2.0])
    if baseline_metrics['loss'][0] == 0.5:
      client_1_idx, client_2_idx = 0, 1
    else:
      client_1_idx, client_2_idx = 1, 0

    # Check if the metrics of `batch_size_1` are correct.
    bs1_metrics = results['batch_size_1']
    # Number of training examples is 3 for both clients.
    self.assertAllEqual(bs1_metrics['num_examples'], [3, 3])
    bs1_test_outputs = bs1_metrics['test_outputs']
    # Number of test examples is also 3 for both clients.
    self.assertAllEqual(bs1_test_outputs['num_examples'], [3, 3])
    # Number of test batches is 1 for both clients since test batch size is 3.
    self.assertAllEqual(bs1_test_outputs['num_batches'], [1, 1])
    # Both clients's weights become [-3, -3, -1] after training, which gives an
    # average loss 24 for Client 1 and 88.5 for Client 2.
    self.assertAlmostEqual(bs1_test_outputs['loss'][client_1_idx], 24.0)
    self.assertAlmostEqual(bs1_test_outputs['loss'][client_2_idx], 88.5)

    # Check if the metrics of `batch_size_2` are correct.
    bs2_metrics = results['batch_size_2']
    # Number of training examples is 3 for both clients.
    self.assertAllEqual(bs2_metrics['num_examples'], [3, 3])
    bs2_test_outputs = bs2_metrics['test_outputs']
    # Number of test examples is also 3 for both clients.
    self.assertAllEqual(bs2_test_outputs['num_examples'], [3, 3])
    # Number of test batches is 1 for both clients since test batch size is 3.
    self.assertAllEqual(bs2_test_outputs['num_batches'], [1, 1])
    # Both clients' weights become [0, 0, 1] after training, which gives an
    # average loss 0 for Client 1 and 0.5 for Client 2.
    self.assertAlmostEqual(bs2_test_outputs['loss'][client_1_idx], 0.0)
    self.assertAlmostEqual(bs2_test_outputs['loss'][client_2_idx], 0.5)

  def test_success_with_model_constructed_from_keras(self):

    def model_fn():
      inputs = tf.keras.Input(shape=(2,))  # feature dim = 2
      outputs = tf.keras.layers.Dense(1)(inputs)
      keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
      input_spec = collections.OrderedDict([
          ('x', tf.TensorSpec([None, 2], dtype=tf.float32)),
          ('y', tf.TensorSpec([None, 1], dtype=tf.float32))
      ])
      return keras_utils.from_keras_model(
          keras_model,
          input_spec=input_spec,
          loss=tf.keras.losses.MeanSquaredError())

    zero_model_weights = _create_zero_model_weights(model_fn)
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=0.5)

    federated_p13n_eval = p13n_eval.build_personalization_eval(
        model_fn, p13n_fn_dict, _evaluate_fn)

    # Perform p13n eval on two clients: their train data are equivalent, but the
    # test data have different scales.
    results = federated_p13n_eval(zero_model_weights, [
        _create_client_input(train_scale=1.0, test_scale=1.0),
        _create_client_input(train_scale=1.0, test_scale=2.0)
    ])
    results = results._asdict(recursive=True)

    # Check if the baseline metrics are correct.
    baseline_metrics = results['baseline_metrics']
    # The initial weights are all zeros. The MeanSquredError(MSE) is:
    # Client 1, (1 + 1 + 1)/3 = 1.0; Client 2, (4 + 4 + 4)/3 = 4.0.
    # Note: the order is not preserved due to `federated_sample`.
    self.assertAllEqual(sorted(baseline_metrics['loss']), [1.0, 4.0])

    # Check if the metrics of `batch_size_1` are correct.
    bs1_metrics = results['batch_size_1']
    # Number of training examples is 3 for both clients.
    self.assertAllEqual(bs1_metrics['num_examples'], [3, 3])
    bs1_test_outputs = bs1_metrics['test_outputs']
    # Both clients' weights become [-3, -3, -1] after training, which gives MSE
    # 48 for Client 1 and 177 for Client 2.
    self.assertAlmostEqual(sorted(bs1_test_outputs['loss']), [48.0, 177.0])

    # Check if the metrics of `batch_size_2` are correct.
    bs2_metrics = results['batch_size_2']
    # Number of training examples is 3 for both clients.
    self.assertAllEqual(bs2_metrics['num_examples'], [3, 3])
    bs2_test_outputs = bs2_metrics['test_outputs']
    # Both clients' weights become [0, 0, 1] after training, which gives MSE 0
    # for Client 1 and 1.0 for Client 2.
    self.assertAlmostEqual(sorted(bs2_test_outputs['loss']), [0.0, 1.0])

  def test_failure_with_batched_datasets(self):

    def model_fn():
      return model_examples.LinearRegression(feature_dim=2)

    zero_model_weights = _create_zero_model_weights(model_fn)
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)

    federated_p13n_eval = p13n_eval.build_personalization_eval(
        model_fn, p13n_fn_dict, _evaluate_fn)

    with self.assertRaises(TypeError):
      # client_input should not have batched datasets.
      bad_client_input = collections.OrderedDict([
          ('train_data', _create_dataset(scale=1.0).batch(1)),
          ('test_data', _create_dataset(scale=1.0).batch(1))
      ])
      federated_p13n_eval(zero_model_weights, [bad_client_input])

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
      context_tff_type = computation_types.to_type(tf.int32)
      federated_p13n_eval = p13n_eval.build_personalization_eval(
          model_fn,
          p13n_fn_dict,
          _evaluate_fn,
          context_tff_type=context_tff_type)
      federated_p13n_eval(zero_model_weights, [
          _create_client_input(train_scale=1.0, test_scale=1.0, context=None),
          _create_client_input(train_scale=1.0, test_scale=2.0, context=None)
      ])

  def test_success_with_valid_context(self):

    def model_fn():
      return model_examples.LinearRegression(feature_dim=2)

    zero_model_weights = _create_zero_model_weights(model_fn)
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)

    # Build the p13n eval with an extra `context` argument.
    context_tff_type = computation_types.to_type(tf.int32)
    federated_p13n_eval = p13n_eval.build_personalization_eval(
        model_fn, p13n_fn_dict, _evaluate_fn, context_tff_type=context_tff_type)

    # Perform p13n eval on two clients with different `context` values.
    results = federated_p13n_eval(zero_model_weights, [
        _create_client_input(train_scale=1.0, test_scale=1.0, context=2),
        _create_client_input(train_scale=1.0, test_scale=2.0, context=5)
    ])
    results = results._asdict(recursive=True)

    bs1_metrics = results['batch_size_1']
    bs2_metrics = results['batch_size_2']

    # Number of training examples is `3 + context` for both clients.
    # Note: the order is not preserved due to `federated_sample`, but the order
    # should be consistent across different personalization strategies.
    self.assertAllEqual(sorted(bs1_metrics['num_examples']), [5, 8])
    self.assertAllEqual(bs1_metrics['num_examples'],
                        bs2_metrics['num_examples'])

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

    # Perform p13n eval on two clients.
    results = federated_p13n_eval(zero_model_weights, [
        _create_client_input(train_scale=1.0, test_scale=1.0),
        _create_client_input(train_scale=1.0, test_scale=2.0)
    ])
    results = results._asdict(recursive=True)

    # The results should only contain metrics from one client.
    self.assertAllEqual(len(results['baseline_metrics']['loss']), 1)
    self.assertAllEqual(len(results['batch_size_1']['test_outputs']['loss']), 1)
    self.assertAllEqual(len(results['batch_size_2']['test_outputs']['loss']), 1)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
