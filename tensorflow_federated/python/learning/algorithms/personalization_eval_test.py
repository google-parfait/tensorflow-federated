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
from collections.abc import Mapping
from unittest import mock

from absl.testing import parameterized
import federated_language
import keras
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import cpp_execution_contexts
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning.algorithms import personalization_eval as p13n_eval
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.optimizers import sgdm

# TODO: b/160896627 - Switch to `dataset.reduce` once multi-GPU supports it.
dataset_reduce_fn = loop_builder.build_training_loop(
    loop_builder.LoopImplementation.DATASET_ITERATOR
)


def _build_functional_linear_regression(
    feature_dim: int = 2,
) -> functional.FunctionalModel:
  """Builds a linear regression FunctionalModel with 0.5 * MSE loss."""
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, feature_dim], tf.float32),
      y=tf.TensorSpec([None, 1], tf.float32),
  )
  initial_trainable_weights = (
      np.zeros([feature_dim, 1], dtype=np.float32),
      np.zeros([1], dtype=np.float32),
  )
  initial_weights = (initial_trainable_weights, ())

  @tf.function
  def predict_on_batch(weights, x, training=True):
    del training
    trainable_weights, _ = weights
    kernel, bias = trainable_weights
    return x @ kernel + bias

  def loss(output, label, sample_weight=None):
    del sample_weight
    return 0.5 * tf.math.reduce_mean(tf.math.pow(output - label, 2.0))

  return functional.FunctionalModel(
      initial_weights=initial_weights,
      predict_on_batch_fn=predict_on_batch,
      loss_fn=loss,
      input_spec=input_spec,
  )


@tf.function
def _evaluate_variable_model_fn(model, dataset, batch_size=1):
  """Evaluates a `tff.learning.models.VariableModel` on the given dataset."""
  # Reset the local variables so that the returned metrics are computed using
  # the given data. Similar to the `reset_states` method of `tf.metrics.Metric`.
  for var in model.local_variables:
    if var.initial_value is not None:
      var.assign(var.initial_value)
    else:
      var.assign(tf.zeros_like(var))

  def eval_fn(whimsy_state, batch):
    """Evaluates the model on a batch."""
    model.forward_pass(batch, training=False)
    return whimsy_state

  # Evaluate on the dataset.
  batched_dataset = dataset.batch(batch_size)
  dataset_reduce_fn(
      reduce_fn=eval_fn,
      dataset=batched_dataset,
      initial_state_fn=lambda: tf.constant(0),
  )

  # Obtain the metrics.
  results = collections.OrderedDict()
  local_outputs = model.report_local_unfinalized_metrics()
  for name, metric in local_outputs.items():
    if isinstance(metric, list) and (len(metric) == 2):
      # Some metrics returned by `report_local_unfinalized_metrics()` can have
      # two scalars: one represents `sum`, and the other represents `count`.
      # Ideally, we want to return a single scalar for each metric.
      results[name] = metric[0] / metric[1]
    else:
      results[name] = metric[0] if isinstance(metric, list) else metric
  return results


def _build_variable_model_personalize_fn(
    optimizer_fn, train_batch_size, test_batch_size
):
  """Builds a personalization function for `VariableModel`."""
  optimizer = optimizer_fn()

  @tf.function
  def personalize_fn(model, train_data, test_data):
    def train_fn(num_examples_sum, batch):
      """Runs gradient descent on a batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)

      grads = tape.gradient(output.loss, model.trainable_variables)
      optimizer.apply_gradients(
          zip(
              tf.nest.flatten(grads), tf.nest.flatten(model.trainable_variables)
          )
      )
      return num_examples_sum + output.num_examples

    # Train a personalized model.
    batched_train_data = train_data.batch(train_batch_size)
    num_examples_sum = dataset_reduce_fn(
        reduce_fn=train_fn,
        dataset=batched_train_data,
        initial_state_fn=lambda: tf.constant(0),
    )

    results = collections.OrderedDict()
    results['num_examples'] = num_examples_sum
    results['test_outputs'] = _evaluate_variable_model_fn(
        model, test_data, test_batch_size
    )
    return results

  return personalize_fn


def _create_variable_model_p13n_fn_dict(learning_rate):
  """Creates a dictionary containing two `VariableModel` strategies."""
  p13n_fn_dict = collections.OrderedDict()

  opt_fn = lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)
  # The two personalization strategies use different training batch sizes.
  p13n_fn_dict['batch_size_1'] = lambda: _build_variable_model_personalize_fn(
      opt_fn, 1, 3
  )
  p13n_fn_dict['batch_size_2'] = lambda: _build_variable_model_personalize_fn(
      opt_fn, 2, 3
  )

  return p13n_fn_dict


@tf.function
def _evaluate_fn(model, weights, dataset, batch_size=1):
  """Evaluates a `tff.learning.models.FunctionalModel` on the given dataset."""

  def eval_fn(state, batch):
    """Evaluates the model on a batch."""
    metrics_state, loss_sum, num_examples = state
    if isinstance(batch, Mapping):
      x = batch['x']
      y = batch['y']
    else:
      x, y = batch
    batch_output = model.predict_on_batch(weights, x, training=False)
    batch_loss = model.loss(output=batch_output, label=y)
    predictions = tf.nest.flatten(batch_output)[0]
    batch_num_examples = tf.shape(predictions)[0]
    metrics_state = model.update_metrics_state(
        metrics_state,
        labels=y,
        batch_output=variable.BatchOutput(
            loss=batch_loss,
            predictions=batch_output,
            num_examples=batch_num_examples,
        ),
    )
    loss_sum += batch_loss * tf.cast(batch_num_examples, tf.float32)
    num_examples += batch_num_examples
    return metrics_state, loss_sum, num_examples

  # Evaluate on the dataset.
  batched_dataset = dataset.batch(batch_size)
  metrics_state, loss_sum, num_examples = dataset_reduce_fn(
      reduce_fn=eval_fn,
      dataset=batched_dataset,
      initial_state_fn=lambda: (
          model.initialize_metrics_state(),
          tf.constant(0.0, tf.float32),
          tf.constant(0, tf.int32),
      ),
  )

  # Obtain the metrics.
  results = collections.OrderedDict(
      loss=tf.math.divide_no_nan(loss_sum, tf.cast(num_examples, tf.float32)),
      num_examples=num_examples,
  )
  results.update(model.finalize_metrics(metrics_state))
  return results


def _build_personalize_fn(optimizer_fn, train_batch_size, test_batch_size):
  """Builds a personalization function given an optimizer constructor."""
  optimizer = optimizer_fn()

  @tf.function
  def personalize_fn(model, weights, train_data, test_data):
    trainable_weights, non_trainable_weights = weights
    trainable_specs = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, trainable_weights
    )

    def train_fn(state, batch):
      """Runs gradient descent on a batch."""
      opt_state, current_trainable, num_examples_sum = state
      if isinstance(batch, Mapping):
        x = batch['x']
        y = batch['y']
      else:
        x, y = batch
      current_weights = (current_trainable, non_trainable_weights)
      with tf.GradientTape() as tape:
        tape.watch(current_trainable)
        predictions = model.predict_on_batch(current_weights, x, training=True)
        loss = model.loss(output=predictions, label=y)

      grads = tape.gradient(loss, current_trainable)
      opt_state, updated_trainable = optimizer.next(
          opt_state, current_trainable, grads
      )
      batch_num_examples = tf.shape(tf.nest.flatten(predictions)[0])[0]
      return (
          opt_state,
          updated_trainable,
          num_examples_sum + batch_num_examples,
      )

    # Train a personalized model.
    batched_train_data = train_data.batch(train_batch_size)
    _, trained_trainable, num_examples_sum = dataset_reduce_fn(
        reduce_fn=train_fn,
        dataset=batched_train_data,
        initial_state_fn=lambda: (
            optimizer.initialize(trainable_specs),
            trainable_weights,
            tf.constant(0),
        ),
    )

    trained_weights = (trained_trainable, non_trainable_weights)
    results = collections.OrderedDict()
    results['num_examples'] = num_examples_sum
    results['test_outputs'] = _evaluate_fn(
        model, trained_weights, test_data, test_batch_size
    )
    return results

  return personalize_fn


def _create_p13n_fn_dict(learning_rate):
  """Creates a dictionary containing two personalization strategies."""
  p13n_fn_dict = collections.OrderedDict()

  opt_fn = lambda: sgdm.build_sgdm(learning_rate=learning_rate)
  # The two personalization strategies use different training batch sizes.
  p13n_fn_dict['batch_size_1'] = lambda: _build_personalize_fn(opt_fn, 1, 3)
  p13n_fn_dict['batch_size_2'] = lambda: _build_personalize_fn(opt_fn, 2, 3)

  return p13n_fn_dict


def _create_dataset(scale):
  """Constructs a dataset with three datapoints."""
  x = np.array([[-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]) * scale
  y = np.array([[1.0], [1.0], [1.0]]) * scale
  ds = collections.OrderedDict(x=x.astype(np.float32), y=y.astype(np.float32))
  # Note: batching is not needed here as the preprocessing of dataset is done
  # inside the personalization function.
  return tf.data.Dataset.from_tensor_slices(ds)


def _create_client_input(train_scale, test_scale):
  """Constructs client datasets for personalization."""
  client_input = collections.OrderedDict()
  client_input['train_data'] = _create_dataset(train_scale)
  client_input['test_data'] = _create_dataset(test_scale)
  return client_input


def _create_zero_model_weights(model):
  """Creates the model weights with all zeros."""
  whimsy_model = model() if callable(model) else model
  return tf.nest.map_structure(
      np.zeros_like, model_weights.ModelWeights.from_model(whimsy_model)
  )


def _get_linear_regression_and_fns(use_functional_model: bool):
  """Returns `(model_or_fn, create_p13n_fn_dict, evaluate_fn)`."""
  if use_functional_model:
    return (
        _build_functional_linear_regression(feature_dim=2),
        _create_p13n_fn_dict,
        _evaluate_fn,
    )
  return (
      lambda: model_examples.LinearRegression(feature_dim=2),
      _create_variable_model_p13n_fn_dict,
      _evaluate_variable_model_fn,
  )


class PersonalizationEvalTest(tf.test.TestCase, parameterized.TestCase):

  def test_failure_with_invalid_model_fn(self):
    p13n_fn_dict = _create_p13n_fn_dict(learning_rate=1.0)
    with self.assertRaises(TypeError):
      # `model_fn` should be a `FunctionalModel` or callable.
      bad_model_fn = 6
      p13n_eval.build_personalization_eval_computation(
          bad_model_fn, p13n_fn_dict, _evaluate_fn
      )

    with self.assertRaises(TypeError):
      # `model_fn` if callable should return a `VariableModel`.
      bad_model_fn = lambda: 6
      p13n_eval.build_personalization_eval_computation(
          bad_model_fn, p13n_fn_dict, _evaluate_fn
      )

    with self.assertRaises(TypeError):
      # `FunctionalModel` must be passed as an instance, not a callable.
      p13n_eval.build_personalization_eval_computation(
          _build_functional_linear_regression, p13n_fn_dict, _evaluate_fn  # pyrefly: ignore[bad-argument-type]
      )

  @parameterized.named_parameters(
      ('variable_model', False),
      ('functional_model', True),
  )
  def test_failure_with_invalid_p13n_fns(self, use_functional_model):
    model, _, evaluate_fn = _get_linear_regression_and_fns(use_functional_model)

    with self.assertRaises(TypeError):
      # `personalize_fn_dict` should be a `OrderedDict`.
      bad_p13n_fn_dict = {'a': 6}
      p13n_eval.build_personalization_eval_computation(
          model, bad_p13n_fn_dict, evaluate_fn
      )

    with self.assertRaises(TypeError):
      # `personalize_fn_dict` should be a `OrderedDict` that maps a `string` to
      # a `callable`.
      bad_p13n_fn_dict = collections.OrderedDict(a=6)
      p13n_eval.build_personalization_eval_computation(
          model, bad_p13n_fn_dict, evaluate_fn
      )

    with self.assertRaises(TypeError):
      # `personalize_fn_dict` should be a `OrderedDict` that maps a `string` to
      # a `callable` that when called, gives another `callable`.
      bad_p13n_fn_dict = collections.OrderedDict(x=lambda: 2)
      p13n_eval.build_personalization_eval_computation(
          model, bad_p13n_fn_dict, evaluate_fn
      )

    with self.assertRaises(ValueError):
      # `personalize_fn_dict` should not use `baseline_metrics` as a key.
      bad_p13n_fn_dict = collections.OrderedDict(baseline_metrics=lambda: 2)
      p13n_eval.build_personalization_eval_computation(
          model, bad_p13n_fn_dict, evaluate_fn
      )

  @parameterized.named_parameters(
      ('variable_model', False),
      ('functional_model', True),
  )
  def test_failure_with_invalid_baseline_eval_fn(self, use_functional_model):
    model, create_p13n_fn_dict, _ = _get_linear_regression_and_fns(
        use_functional_model
    )

    p13n_fn_dict = create_p13n_fn_dict(learning_rate=1.0)

    with self.assertRaises(TypeError):
      # `baseline_evaluate_fn` should be a callable.
      bad_baseline_evaluate_fn = 6
      p13n_eval.build_personalization_eval_computation(
          model, p13n_fn_dict, bad_baseline_evaluate_fn
      )

  @parameterized.named_parameters(
      ('variable_model', False),
      ('functional_model', True),
  )
  def test_success_with_directly_constructed_model(self, use_functional_model):
    model, create_p13n_fn_dict, evaluate_fn = _get_linear_regression_and_fns(
        use_functional_model
    )

    zero_model_weights = _create_zero_model_weights(model)
    p13n_fn_dict = create_p13n_fn_dict(learning_rate=1.0)

    federated_p13n_eval = p13n_eval.build_personalization_eval_computation(
        model, p13n_fn_dict, evaluate_fn
    )

    # Perform p13n eval on two clients: their train data are equivalent, but the
    # test data have different scales.
    results = federated_p13n_eval(
        zero_model_weights,
        [
            _create_client_input(train_scale=1.0, test_scale=1.0),
            _create_client_input(train_scale=1.0, test_scale=2.0),
        ],
    )

    # Check if the baseline metrics are correct.
    baseline_metrics = results['baseline_metrics']
    # Number of test examples is 3 for both clients.
    self.assertAllEqual(baseline_metrics['num_examples'], [3, 3])
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
    # Both clients' weights become [0, 0, 1] after training, which gives an
    # average loss 0 for Client 1 and 0.5 for Client 2.
    self.assertAlmostEqual(bs2_test_outputs['loss'][client_1_idx], 0.0)
    self.assertAlmostEqual(bs2_test_outputs['loss'][client_2_idx], 0.5)

  @parameterized.product(
      use_functional_model=[False, True],
      input_spec=[
          collections.OrderedDict(
              x=tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
              y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
          ),
          federated_language.StructWithPythonType(
              collections.OrderedDict(
                  x=federated_language.TensorType(
                      shape=[None, 2], dtype=np.float32
                  ),
                  y=federated_language.TensorType(
                      shape=[None, 1], dtype=np.float32
                  ),
              ),
              container_type=collections.OrderedDict,
          ),
          federated_language.StructType(
              collections.OrderedDict(
                  x=federated_language.TensorType(
                      shape=[None, 2], dtype=np.float32
                  ),
                  y=federated_language.TensorType(
                      shape=[None, 1], dtype=np.float32
                  ),
              )
          ),
      ],
  )
  def test_success_with_model_constructed_from_keras(
      self, use_functional_model, input_spec
  ):
    if use_functional_model:
      inputs = keras.Input(shape=(2,))  # feature dim = 2
      outputs = keras.layers.Dense(1)(inputs)
      keras_model = keras.Model(inputs=inputs, outputs=outputs)
      model = functional.functional_model_from_keras(
          keras_model,
          input_spec=input_spec,
          loss_fn=keras.losses.MeanSquaredError(),
      )
      p13n_fn_dict = _create_p13n_fn_dict(learning_rate=0.5)
      evaluate_fn = _evaluate_fn
    else:

      def model_fn():
        inputs = tf.keras.Input(shape=(2,))  # feature dim = 2
        outputs = tf.keras.layers.Dense(1)(inputs)
        return keras_utils.from_keras_model(
            tf.keras.Model(inputs=inputs, outputs=outputs),
            input_spec=input_spec,
            loss=tf.keras.losses.MeanSquaredError(),
        )

      model = model_fn
      p13n_fn_dict = _create_variable_model_p13n_fn_dict(learning_rate=0.5)
      evaluate_fn = _evaluate_variable_model_fn

    zero_model_weights = _create_zero_model_weights(model)

    federated_p13n_eval = p13n_eval.build_personalization_eval_computation(
        model, p13n_fn_dict, evaluate_fn
    )

    # Perform p13n eval on two clients: their train data are equivalent, but the
    # test data have different scales.
    results = federated_p13n_eval(
        zero_model_weights,
        [
            _create_client_input(train_scale=1.0, test_scale=1.0),
            _create_client_input(train_scale=1.0, test_scale=2.0),
        ],
    )

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

  @parameterized.named_parameters(
      ('variable_model', False),
      ('functional_model', True),
  )
  def test_failure_with_batched_datasets(self, use_functional_model):
    model, create_p13n_fn_dict, evaluate_fn = _get_linear_regression_and_fns(
        use_functional_model
    )

    zero_model_weights = _create_zero_model_weights(model)
    p13n_fn_dict = create_p13n_fn_dict(learning_rate=1.0)

    federated_p13n_eval = p13n_eval.build_personalization_eval_computation(
        model, p13n_fn_dict, evaluate_fn
    )

    with self.assertRaises(TypeError):
      # client_input should not have batched datasets.
      bad_client_input = collections.OrderedDict(
          train_data=_create_dataset(scale=1.0).batch(1),
          test_data=_create_dataset(scale=1.0).batch(1),
      )
      federated_p13n_eval(zero_model_weights, [bad_client_input])

  @parameterized.named_parameters(
      ('variable_model', False),
      ('functional_model', True),
  )
  def test_failure_with_invalid_sample_size(self, use_functional_model):
    model, create_p13n_fn_dict, evaluate_fn = _get_linear_regression_and_fns(
        use_functional_model
    )

    p13n_fn_dict = create_p13n_fn_dict(learning_rate=1.0)

    with self.assertRaises(TypeError):
      # `max_num_clients` should be an `int`.
      bad_num_clients = 1.0
      p13n_eval.build_personalization_eval_computation(
          model, p13n_fn_dict, evaluate_fn, max_num_clients=bad_num_clients  # pyrefly: ignore[bad-argument-type]
      )

    with self.assertRaises(ValueError):
      # `max_num_clients` should be a positive `int`.
      bad_num_clients = 0
      p13n_eval.build_personalization_eval_computation(
          model, p13n_fn_dict, evaluate_fn, max_num_clients=bad_num_clients
      )

  @parameterized.named_parameters(
      ('variable_model', False),
      ('functional_model', True),
  )
  def test_success_with_small_sample_size(self, use_functional_model):
    model, create_p13n_fn_dict, evaluate_fn = _get_linear_regression_and_fns(
        use_functional_model
    )

    zero_model_weights = _create_zero_model_weights(model)
    p13n_fn_dict = create_p13n_fn_dict(learning_rate=1.0)

    federated_p13n_eval = p13n_eval.build_personalization_eval_computation(
        model, p13n_fn_dict, evaluate_fn, max_num_clients=1
    )

    # Perform p13n eval on two clients.
    results = federated_p13n_eval(
        zero_model_weights,
        [
            _create_client_input(train_scale=1.0, test_scale=1.0),
            _create_client_input(train_scale=1.0, test_scale=2.0),
        ],
    )
    # The results should only contain metrics from one client.
    self.assertAllEqual(len(results['baseline_metrics']['loss']), 1)
    self.assertAllEqual(len(results['batch_size_1']['test_outputs']['loss']), 1)
    self.assertAllEqual(len(results['batch_size_2']['test_outputs']['loss']), 1)

  def test_construction_calls_model_fn(self):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    p13n_fn_dict = _create_variable_model_p13n_fn_dict(learning_rate=1.0)
    p13n_eval.build_personalization_eval_computation(
        mock_model_fn,
        p13n_fn_dict,
        _evaluate_variable_model_fn,
        max_num_clients=1,
    )
    # TODO: b/186451541 - reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 3)


if __name__ == '__main__':
  cpp_execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
