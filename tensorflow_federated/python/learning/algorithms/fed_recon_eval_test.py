# Copyright 2020, Google LLC.
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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import attrs
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.templates import measured_process as measured_process_lib
from tensorflow_federated.python.learning.algorithms import fed_recon_eval
from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.models import reconstruction_model
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import learning_process as learning_process_lib


# Convenience aliases.
FunctionType = computation_types.FunctionType
SequenceType = computation_types.SequenceType
TensorType = computation_types.TensorType
LearningAlgorithmState = composers.LearningAlgorithmState
LearningProcessOutput = learning_process_lib.LearningProcessOutput


def _create_input_spec():
  return collections.OrderedDict(
      x=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
      y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
  )


@attrs.define(eq=False, frozen=True)
class LinearModelVariables:
  """Structure for variables in `LinearModel`."""

  weights: float
  bias: float


class LinearModel(reconstruction_model.ReconstructionModel):
  """An implementation of an MNIST `ReconstructionModel` without Keras."""

  def __init__(self):
    self._variables = LinearModelVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(1, 1)),
            name='weights',
            trainable=True,
        ),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(1, 1)),
            name='bias',
            trainable=True,
        ),
    )

  @property
  def global_trainable_variables(self):
    return [self._variables.weights]

  @property
  def global_non_trainable_variables(self):
    return []

  @property
  def local_trainable_variables(self):
    return [self._variables.bias]

  @property
  def local_non_trainable_variables(self):
    return []

  @property
  def input_spec(self):
    return _create_input_spec()

  @tf.function
  def forward_pass(self, batch, training=True):
    del training

    y = batch['x'] * self._variables.weights + self._variables.bias
    return reconstruction_model.ReconstructionBatchOutput(
        predictions=y, labels=batch['y'], num_examples=tf.size(batch['y'])
    )


class BiasLayer(tf.keras.layers.Layer):
  """Adds a bias to inputs."""

  def build(self, input_shape):
    self.bias = self.add_weight(
        'bias', shape=input_shape[1:], initializer='zeros', trainable=True
    )

  def call(self, x):
    return x + self.bias


def keras_linear_model_fn():
  """Should produce the same results as `LinearModel`."""
  inputs = tf.keras.layers.Input(shape=[1])
  scaled_input = tf.keras.layers.Dense(
      1, use_bias=False, kernel_initializer='zeros'
  )(inputs)
  outputs = BiasLayer()(scaled_input)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
  input_spec = _create_input_spec()
  return reconstruction_model.ReconstructionModel.from_keras_model_and_layers(
      keras_model=keras_model,
      global_layers=keras_model.layers[:-1],
      local_layers=keras_model.layers[-1:],
      input_spec=input_spec,
  )


class NumOverCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts examples greater than a constant.

  This metric counts label examples greater than a threshold.
  """

  def __init__(
      self, threshold: float, name: str = 'num_over', dtype=tf.float32
  ):
    super().__init__(name, dtype)
    self.threshold = threshold

  def update_state(self, y_true, y_pred, sample_weight=None):
    num_over = tf.reduce_sum(
        tf.cast(tf.greater(y_true, self.threshold), tf.float32)
    )
    return super().update_state(num_over)

  def get_config(self):
    config = {'threshold': self.threshold}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def create_client_data():
  client1_data = collections.OrderedDict(
      [('x', [[1.0], [2.0], [3.0]]), ('y', [[5.0], [6.0], [8.0]])]
  )
  client2_data = collections.OrderedDict(
      [('x', [[1.0], [2.0], [3.0]]), ('y', [[5.0], [5.0], [9.0]])]
  )

  client1_dataset = tf.data.Dataset.from_tensor_slices(client1_data).batch(1)
  client2_dataset = tf.data.Dataset.from_tensor_slices(client2_data).batch(1)

  return [client1_dataset, client2_dataset]


def _get_tff_optimizer(learning_rate=0.1):
  return sgdm.build_sgdm(learning_rate=learning_rate, momentum=0.5)


def _get_keras_optimizer_fn(learning_rate=0.1):
  return lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)


class FedreconEvaluationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('non_keras_model_with_keras_opt', LinearModel, _get_keras_optimizer_fn),
      ('non_keras_model_with_tff_opt', LinearModel, _get_tff_optimizer),
      (
          'keras_model_with_keras_opt',
          keras_linear_model_fn,
          _get_keras_optimizer_fn,
      ),
      ('keras_model_with_tff_opt', keras_linear_model_fn, _get_tff_optimizer),
  )
  def test_federated_reconstruction_no_split_data(self, model_fn, optimizer_fn):
    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [counters.NumExamplesCounter(), NumOverCounter(5.0)]

    dataset_split_fn = (
        reconstruction_model.ReconstructionModel.build_dataset_split_fn()
    )

    evaluate = fed_recon_eval.build_fed_recon_eval(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=optimizer_fn(),
        dataset_split_fn=dataset_split_fn,
    )
    global_weights_type = reconstruction_model.global_weights_type_from_model(
        model_fn()
    )
    state_type = computation_types.FederatedType(
        LearningAlgorithmState(
            global_model_weights=global_weights_type,
            distributor=(),
            client_work=(
                (),
                collections.OrderedDict(
                    num_examples=[np.int64],
                    num_over=[np.float32],
                    loss=[np.float32, np.float32],
                ),
            ),
            aggregator=collections.OrderedDict(
                value_sum_process=(), weight_sum_process=()
            ),
            finalizer=(),
        ),
        placements.SERVER,
    )
    type_test_utils.assert_types_identical(
        evaluate.next.type_signature,
        FunctionType(
            parameter=collections.OrderedDict(
                state=state_type,
                client_data=computation_types.FederatedType(
                    SequenceType(
                        collections.OrderedDict(
                            x=TensorType(np.float32, [None, 1]),
                            y=TensorType(np.float32, [None, 1]),
                        )
                    ),
                    placements.CLIENTS,
                ),
            ),
            result=LearningProcessOutput(
                state=state_type,
                metrics=computation_types.FederatedType(
                    collections.OrderedDict(
                        distributor=(),
                        client_work=collections.OrderedDict(
                            eval=collections.OrderedDict(
                                current_round_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                                total_rounds_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                            )
                        ),
                        aggregator=collections.OrderedDict(
                            mean_value=(), mean_weight=()
                        ),
                        finalizer=(),
                    ),
                    placements.SERVER,
                ),
            ),
        ),
    )

    state = evaluate.initialize()
    # Explicitly set weights for testing assertions below.
    state = evaluate.set_model_weights(
        state,
        reconstruction_model.model_weights.ModelWeights(
            trainable=(tf.convert_to_tensor([[1.0]]),), non_trainable=()
        ),
    )
    _, result = evaluate.next(state, create_client_data())
    eval_result = result['client_work']['eval']['current_round_metrics']

    # Ensure loss isn't equal to the value we'd expect if no reconstruction
    # happens. We can calculate this since the local bias is initialized at 0
    # and not reconstructed. MSE is (y - 1 * x)^2 for each example, for a mean
    # of (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3.
    self.assertNotAlmostEqual(eval_result['loss'], 19.666666)
    self.assertAlmostEqual(eval_result['num_examples'], 6.0)
    self.assertAlmostEqual(eval_result['num_over'], 3.0)

  @parameterized.named_parameters(
      ('non_keras_model_with_keras_opt', LinearModel, _get_keras_optimizer_fn),
      ('non_keras_model_with_tff_opt', LinearModel, _get_tff_optimizer),
      (
          'keras_model_with_keras_opt',
          keras_linear_model_fn,
          _get_keras_optimizer_fn,
      ),
      ('keras_model_with_tff_opt', keras_linear_model_fn, _get_tff_optimizer),
  )
  def test_federated_reconstruction_split_data(self, model_fn, optimizer_fn):
    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [counters.NumExamplesCounter(), NumOverCounter(5.0)]

    evaluate = fed_recon_eval.build_fed_recon_eval(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=optimizer_fn(),
    )

    global_weights_type = reconstruction_model.global_weights_type_from_model(
        model_fn()
    )
    state_type = computation_types.FederatedType(
        LearningAlgorithmState(
            global_model_weights=global_weights_type,
            distributor=(),
            client_work=(
                (),
                collections.OrderedDict(
                    num_examples=[np.int64],
                    num_over=[np.float32],
                    loss=[np.float32, np.float32],
                ),
            ),
            aggregator=collections.OrderedDict(
                value_sum_process=(), weight_sum_process=()
            ),
            finalizer=(),
        ),
        placements.SERVER,
    )
    type_test_utils.assert_types_identical(
        evaluate.next.type_signature,
        FunctionType(
            parameter=collections.OrderedDict(
                state=state_type,
                client_data=computation_types.FederatedType(
                    SequenceType(
                        collections.OrderedDict(
                            x=TensorType(np.float32, [None, 1]),
                            y=TensorType(np.float32, [None, 1]),
                        )
                    ),
                    placements.CLIENTS,
                ),
            ),
            result=LearningProcessOutput(
                state=state_type,
                metrics=computation_types.FederatedType(
                    collections.OrderedDict(
                        distributor=(),
                        client_work=collections.OrderedDict(
                            eval=collections.OrderedDict(
                                current_round_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                                total_rounds_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                            )
                        ),
                        aggregator=collections.OrderedDict(
                            mean_value=(), mean_weight=()
                        ),
                        finalizer=(),
                    ),
                    placements.SERVER,
                ),
            ),
        ),
    )

    state = evaluate.initialize()
    _, result = evaluate.next(state, create_client_data())
    eval_result = result['client_work']['eval']['current_round_metrics']

    self.assertAlmostEqual(eval_result['num_examples'], 2.0)
    self.assertAlmostEqual(eval_result['num_over'], 1.0)

  @parameterized.named_parameters(
      ('non_keras_model_with_keras_opt', LinearModel, _get_keras_optimizer_fn),
      ('non_keras_model_with_tff_opt', LinearModel, _get_tff_optimizer),
      (
          'keras_model_with_keras_opt',
          keras_linear_model_fn,
          _get_keras_optimizer_fn,
      ),
      ('keras_model_with_tff_opt', keras_linear_model_fn, _get_tff_optimizer),
  )
  def test_federated_reconstruction_split_data_multiple_epochs(
      self, model_fn, optimizer_fn
  ):
    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [counters.NumExamplesCounter(), NumOverCounter(5.0)]

    dataset_split_fn = (
        reconstruction_model.ReconstructionModel.build_dataset_split_fn(
            recon_epochs=2,
            post_recon_epochs=10,
            post_recon_steps_max=7,
            split_dataset=True,
        )
    )

    evaluate = fed_recon_eval.build_fed_recon_eval(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=optimizer_fn(),
        dataset_split_fn=dataset_split_fn,
    )
    global_weights_type = reconstruction_model.global_weights_type_from_model(
        model_fn()
    )
    state_type = computation_types.FederatedType(
        LearningAlgorithmState(
            global_model_weights=global_weights_type,
            distributor=(),
            client_work=(
                (),
                collections.OrderedDict(
                    num_examples=[np.int64],
                    num_over=[np.float32],
                    loss=[np.float32, np.float32],
                ),
            ),
            aggregator=collections.OrderedDict(
                value_sum_process=(), weight_sum_process=()
            ),
            finalizer=(),
        ),
        placements.SERVER,
    )
    type_test_utils.assert_types_identical(
        evaluate.next.type_signature,
        FunctionType(
            parameter=collections.OrderedDict(
                state=state_type,
                client_data=computation_types.FederatedType(
                    SequenceType(
                        collections.OrderedDict(
                            x=TensorType(np.float32, [None, 1]),
                            y=TensorType(np.float32, [None, 1]),
                        )
                    ),
                    placements.CLIENTS,
                ),
            ),
            result=LearningProcessOutput(
                state=state_type,
                metrics=computation_types.FederatedType(
                    collections.OrderedDict(
                        distributor=(),
                        client_work=collections.OrderedDict(
                            eval=collections.OrderedDict(
                                current_round_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                                total_rounds_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                            )
                        ),
                        aggregator=collections.OrderedDict(
                            mean_value=(), mean_weight=()
                        ),
                        finalizer=(),
                    ),
                    placements.SERVER,
                ),
            ),
        ),
    )

    state = evaluate.initialize()
    _, result = evaluate.next(state, create_client_data())
    eval_result = result['client_work']['eval']['current_round_metrics']

    self.assertAlmostEqual(eval_result['num_examples'], 14.0)
    self.assertAlmostEqual(eval_result['num_over'], 7.0)

  @parameterized.named_parameters(
      ('non_keras_model_with_keras_opt', LinearModel, _get_keras_optimizer_fn),
      ('non_keras_model_with_tff_opt', LinearModel, _get_tff_optimizer),
      (
          'keras_model_with_keras_opt',
          keras_linear_model_fn,
          _get_keras_optimizer_fn,
      ),
      ('keras_model_with_tff_opt', keras_linear_model_fn, _get_tff_optimizer),
  )
  def test_federated_reconstruction_recon_lr_0(self, model_fn, optimizer_fn):
    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [counters.NumExamplesCounter(), NumOverCounter(5.0)]

    dataset_split_fn = (
        reconstruction_model.ReconstructionModel.build_dataset_split_fn()
    )

    evaluate = fed_recon_eval.build_fed_recon_eval(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        # Set recon optimizer LR to 0 so reconstruction has no effect.
        reconstruction_optimizer_fn=optimizer_fn(0.0),
        dataset_split_fn=dataset_split_fn,
    )
    global_weights_type = reconstruction_model.global_weights_type_from_model(
        model_fn()
    )
    state_type = computation_types.FederatedType(
        LearningAlgorithmState(
            global_model_weights=global_weights_type,
            distributor=(),
            client_work=(
                (),
                collections.OrderedDict(
                    num_examples=[np.int64],
                    num_over=[np.float32],
                    loss=[np.float32, np.float32],
                ),
            ),
            aggregator=collections.OrderedDict(
                value_sum_process=(), weight_sum_process=()
            ),
            finalizer=(),
        ),
        placements.SERVER,
    )
    type_test_utils.assert_types_identical(
        evaluate.next.type_signature,
        FunctionType(
            parameter=collections.OrderedDict(
                state=state_type,
                client_data=computation_types.FederatedType(
                    SequenceType(
                        collections.OrderedDict(
                            x=TensorType(np.float32, [None, 1]),
                            y=TensorType(np.float32, [None, 1]),
                        )
                    ),
                    placements.CLIENTS,
                ),
            ),
            result=LearningProcessOutput(
                state=state_type,
                metrics=computation_types.FederatedType(
                    collections.OrderedDict(
                        distributor=(),
                        client_work=collections.OrderedDict(
                            eval=collections.OrderedDict(
                                current_round_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                                total_rounds_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                            )
                        ),
                        aggregator=collections.OrderedDict(
                            mean_value=(), mean_weight=()
                        ),
                        finalizer=(),
                    ),
                    placements.SERVER,
                ),
            ),
        ),
    )

    state = evaluate.initialize()
    state = evaluate.set_model_weights(
        state,
        reconstruction_model.model_weights.ModelWeights(
            trainable=(tf.convert_to_tensor([[1.0]]),), non_trainable=()
        ),
    )
    _, result = evaluate.next(state, create_client_data())
    eval_result = result['client_work']['eval']['current_round_metrics']

    # Now have an expectation for loss since the local bias is initialized at 0
    # and not reconstructed. MSE is (y - 1 * x)^2 for each example, for a mean
    # of (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3.
    self.assertAlmostEqual(eval_result['loss'], 19.666666)
    self.assertAlmostEqual(eval_result['num_examples'], 6.0)
    self.assertAlmostEqual(eval_result['num_over'], 3.0)

  @parameterized.named_parameters(
      ('non_keras_model_with_keras_opt', LinearModel, _get_keras_optimizer_fn),
      ('non_keras_model_with_tff_opt', LinearModel, _get_tff_optimizer),
      (
          'keras_model_with_keras_opt',
          keras_linear_model_fn,
          _get_keras_optimizer_fn,
      ),
      ('keras_model_with_tff_opt', keras_linear_model_fn, _get_tff_optimizer),
  )
  def test_federated_reconstruction_skip_recon(self, model_fn, optimizer_fn):
    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [counters.NumExamplesCounter(), NumOverCounter(5.0)]

    def dataset_split_fn(client_dataset):
      return client_dataset.repeat(0), client_dataset.repeat(2)

    evaluate = fed_recon_eval.build_fed_recon_eval(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=optimizer_fn(),
        dataset_split_fn=dataset_split_fn,
    )
    global_weights_type = reconstruction_model.global_weights_type_from_model(
        model_fn()
    )
    state_type = computation_types.FederatedType(
        LearningAlgorithmState(
            global_model_weights=global_weights_type,
            distributor=(),
            client_work=(
                (),
                collections.OrderedDict(
                    num_examples=[np.int64],
                    num_over=[np.float32],
                    loss=[np.float32, np.float32],
                ),
            ),
            aggregator=collections.OrderedDict(
                value_sum_process=(), weight_sum_process=()
            ),
            finalizer=(),
        ),
        placements.SERVER,
    )
    type_test_utils.assert_types_identical(
        evaluate.next.type_signature,
        FunctionType(
            parameter=collections.OrderedDict(
                state=state_type,
                client_data=computation_types.FederatedType(
                    SequenceType(
                        collections.OrderedDict(
                            x=TensorType(np.float32, [None, 1]),
                            y=TensorType(np.float32, [None, 1]),
                        )
                    ),
                    placements.CLIENTS,
                ),
            ),
            result=LearningProcessOutput(
                state=state_type,
                metrics=computation_types.FederatedType(
                    collections.OrderedDict(
                        distributor=(),
                        client_work=collections.OrderedDict(
                            eval=collections.OrderedDict(
                                current_round_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                                total_rounds_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                            )
                        ),
                        aggregator=collections.OrderedDict(
                            mean_value=(), mean_weight=()
                        ),
                        finalizer=(),
                    ),
                    placements.SERVER,
                ),
            ),
        ),
    )

    state = evaluate.initialize()
    state = evaluate.set_model_weights(
        state,
        reconstruction_model.model_weights.ModelWeights(
            trainable=(tf.convert_to_tensor([[1.0]]),), non_trainable=()
        ),
    )
    _, result = evaluate.next(state, create_client_data())
    eval_result = result['client_work']['eval']['current_round_metrics']

    # Now have an expectation for loss since the local bias is initialized at 0
    # and not reconstructed. MSE is (y - 1 * x)^2 for each example, for a mean
    # of (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3
    self.assertAlmostEqual(eval_result['loss'], 19.666666)
    self.assertAlmostEqual(eval_result['num_examples'], 12.0)
    self.assertAlmostEqual(eval_result['num_over'], 6.0)

  @parameterized.named_parameters(
      ('non_keras_model_with_keras_opt', LinearModel, _get_keras_optimizer_fn),
      ('non_keras_model_with_tff_opt', LinearModel, _get_tff_optimizer),
      (
          'keras_model_with_keras_opt',
          keras_linear_model_fn,
          _get_keras_optimizer_fn,
      ),
      ('keras_model_with_tff_opt', keras_linear_model_fn, _get_tff_optimizer),
  )
  def test_federated_reconstruction_metrics_none_loss_decreases(
      self, model_fn, optimizer_fn
  ):
    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    dataset_split_fn = (
        reconstruction_model.ReconstructionModel.build_dataset_split_fn(
            recon_epochs=3
        )
    )

    evaluate = fed_recon_eval.build_fed_recon_eval(
        model_fn,
        loss_fn=loss_fn,
        reconstruction_optimizer_fn=optimizer_fn(0.01),
        dataset_split_fn=dataset_split_fn,
    )
    global_weights_type = reconstruction_model.global_weights_type_from_model(
        model_fn()
    )
    state_type = computation_types.FederatedType(
        LearningAlgorithmState(
            global_model_weights=global_weights_type,
            distributor=(),
            client_work=(
                (),
                collections.OrderedDict(
                    loss=[np.float32, np.float32],
                ),
            ),
            aggregator=collections.OrderedDict(
                value_sum_process=(), weight_sum_process=()
            ),
            finalizer=(),
        ),
        placements.SERVER,
    )
    type_test_utils.assert_types_identical(
        evaluate.next.type_signature,
        FunctionType(
            parameter=collections.OrderedDict(
                state=state_type,
                client_data=computation_types.FederatedType(
                    SequenceType(
                        collections.OrderedDict(
                            x=TensorType(np.float32, [None, 1]),
                            y=TensorType(np.float32, [None, 1]),
                        )
                    ),
                    placements.CLIENTS,
                ),
            ),
            result=LearningProcessOutput(
                state=state_type,
                metrics=computation_types.FederatedType(
                    collections.OrderedDict(
                        distributor=(),
                        client_work=collections.OrderedDict(
                            eval=collections.OrderedDict(
                                current_round_metrics=collections.OrderedDict(
                                    loss=np.float32,
                                ),
                                total_rounds_metrics=collections.OrderedDict(
                                    loss=np.float32,
                                ),
                            )
                        ),
                        aggregator=collections.OrderedDict(
                            mean_value=(), mean_weight=()
                        ),
                        finalizer=(),
                    ),
                    placements.SERVER,
                ),
            ),
        ),
    )

    state = evaluate.initialize()
    state = evaluate.set_model_weights(
        state,
        reconstruction_model.model_weights.ModelWeights(
            trainable=(tf.convert_to_tensor([[1.0]]),), non_trainable=()
        ),
    )
    _, result = evaluate.next(state, create_client_data())
    eval_result = result['client_work']['eval']['current_round_metrics']

    # Ensure loss decreases from reconstruction vs. initializing the bias to 0.
    # MSE is (y - 1 * x)^2 for each example, for a mean of
    # (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3.
    self.assertLess(eval_result['loss'], 19.666666)

  @parameterized.named_parameters(
      ('non_keras_model', LinearModel), ('keras_model', keras_linear_model_fn)
  )
  def test_fed_recon_eval_custom_stateful_broadcaster(self, model_fn):
    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [counters.NumExamplesCounter(), NumOverCounter(5.0)]

    model_weights_type = reconstruction_model.global_weights_type_from_model(
        model_fn()
    )

    def build_custom_distributor(
        model_weights_type,
    ) -> distributors.DistributionProcess:
      """Builds a `DistributionProcess` that wraps `tff.federated_broadcast`."""

      @federated_computation.federated_computation()
      def test_server_initialization():
        # Count the number of calls.
        return intrinsics.federated_value(0, placements.SERVER)

      @federated_computation.federated_computation(
          computation_types.FederatedType(np.int32, placements.SERVER),
          computation_types.FederatedType(
              model_weights_type, placements.SERVER
          ),
      )
      def stateful_broadcast(state, value):
        test_metrics = intrinsics.federated_value(3.0, placements.SERVER)
        new_state = intrinsics.federated_map(
            tensorflow_computation.tf_computation(lambda x: x + 1),
            state,
        )
        return measured_process_lib.MeasuredProcessOutput(
            state=new_state,
            result=intrinsics.federated_broadcast(value),
            measurements=test_metrics,
        )

      return distributors.DistributionProcess(
          initialize_fn=test_server_initialization, next_fn=stateful_broadcast
      )

    evaluate = fed_recon_eval.build_fed_recon_eval(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        model_distributor=build_custom_distributor(
            model_weights_type=model_weights_type
        ),
    )
    global_weights_type = reconstruction_model.global_weights_type_from_model(
        model_fn()
    )
    state_type = computation_types.FederatedType(
        LearningAlgorithmState(
            global_model_weights=global_weights_type,
            distributor=np.int32,
            client_work=(
                (),
                collections.OrderedDict(
                    num_examples=[np.int64],
                    num_over=[np.float32],
                    loss=[np.float32, np.float32],
                ),
            ),
            aggregator=collections.OrderedDict(
                value_sum_process=(), weight_sum_process=()
            ),
            finalizer=(),
        ),
        placements.SERVER,
    )
    type_test_utils.assert_types_identical(
        evaluate.next.type_signature,
        FunctionType(
            parameter=collections.OrderedDict(
                state=state_type,
                client_data=computation_types.FederatedType(
                    SequenceType(
                        collections.OrderedDict(
                            x=TensorType(np.float32, [None, 1]),
                            y=TensorType(np.float32, [None, 1]),
                        )
                    ),
                    placements.CLIENTS,
                ),
            ),
            result=LearningProcessOutput(
                state=state_type,
                metrics=computation_types.FederatedType(
                    collections.OrderedDict(
                        distributor=np.float32,
                        client_work=collections.OrderedDict(
                            eval=collections.OrderedDict(
                                current_round_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                                total_rounds_metrics=collections.OrderedDict(
                                    num_examples=np.int64,
                                    num_over=np.float32,
                                    loss=np.float32,
                                ),
                            )
                        ),
                        aggregator=collections.OrderedDict(
                            mean_value=(), mean_weight=()
                        ),
                        finalizer=(),
                    ),
                    placements.SERVER,
                ),
            ),
        ),
    )

    state = evaluate.initialize()
    _, result = evaluate.next(state, create_client_data())

    self.assertEqual(result['distributor'], 3.0)

  def test_evaluation_construction_calls_model_fn(self):
    # Assert that the evaluation building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=keras_linear_model_fn)

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    fed_recon_eval.build_fed_recon_eval(model_fn=mock_model_fn, loss_fn=loss_fn)
    # TODO: b/186451541 - Reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 2)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  absltest.main()
