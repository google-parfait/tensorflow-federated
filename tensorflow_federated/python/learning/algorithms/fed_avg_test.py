# Copyright 2018, The TensorFlow Federated Authors.
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
import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
import keras
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory_utils
from tensorflow_federated.python.core.backends.native import cpp_execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_backend import serialization_utils
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning.algorithms import fed_avg
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import test_models
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import model_delta_client_work


class FedAvgTest(parameterized.TestCase):
  """Tests construction of the FedAvg training process."""

  @parameterized.product(
      optimizer_fn=[
          sgdm.build_sgdm(learning_rate=0.1),
      ],
      aggregation_factory=[
          model_update_aggregator.robust_aggregator,
          model_update_aggregator.secure_aggregator,
      ],
  )
  def test_construction_calls_model_fn(self, optimizer_fn, aggregation_factory):
    # Assert that the process building does not call `model_fn` too many times.
    # `model_fn` can potentially be expensive (loading weights, processing, etc
    # ).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    fed_avg.build_weighted_fed_avg(
        model_fn=mock_model_fn,
        client_optimizer_fn=optimizer_fn,
        model_aggregator=aggregation_factory(),
    )
    self.assertEqual(mock_model_fn.call_count, 3)

  @parameterized.named_parameters(
      ('dataset_reduce', loop_builder.LoopImplementation.DATASET_REDUCE),
      ('dataset_iterator', loop_builder.LoopImplementation.DATASET_ITERATOR),
  )
  @mock.patch.object(
      loop_builder,
      'build_training_loop',
      wraps=loop_builder.build_training_loop,
  )
  def test_client_tf_dataset_reduce_fn(self, loop_implementation, mock_method):
    fed_avg.build_weighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
        loop_implementation=loop_implementation,
    )
    mock_method.assert_called_once_with(loop_implementation=loop_implementation)

  @mock.patch.object(fed_avg, 'build_weighted_fed_avg')
  def test_build_weighted_fed_avg_called_by_unweighted_fed_avg(
      self, mock_fed_avg
  ):
    fed_avg.build_unweighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
    )
    self.assertEqual(mock_fed_avg.call_count, 1)

  @mock.patch.object(fed_avg, 'build_weighted_fed_avg')
  @mock.patch.object(factory_utils, 'as_weighted_aggregator')
  def test_aggregation_wrapper_called_by_unweighted(self, _, mock_as_weighted):
    fed_avg.build_unweighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
    )
    self.assertEqual(mock_as_weighted.call_count, 1)

  def test_raises_on_callable_non_model_fn(self):
    with self.assertRaisesRegex(TypeError, 'callable returned type:'):
      fed_avg.build_weighted_fed_avg(
          model_fn=lambda: 0, client_optimizer_fn=sgdm.build_sgdm()
      )
    with self.assertRaisesRegex(TypeError, 'callable returned type:'):
      fed_avg.build_unweighted_fed_avg(
          model_fn=lambda: 0, client_optimizer_fn=sgdm.build_sgdm()
      )

  def test_raises_on_invalid_client_weighting(self):
    with self.assertRaisesRegex(TypeError, 'client_weighting'):
      fed_avg.build_weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          client_weighting='uniform',
      )

  def test_weighted_fed_avg_raises_on_unweighted_aggregator(self):
    model_aggregator = model_update_aggregator.robust_aggregator(weighted=False)
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      fed_avg.build_weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_aggregator=model_aggregator,
      )

  def test_unweighted_fed_avg_raises_on_weighted_aggregator(self):
    model_aggregator = model_update_aggregator.robust_aggregator(weighted=True)
    with self.assertRaisesRegex(TypeError, 'UnweightedAggregationFactory'):
      fed_avg.build_unweighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_aggregator=model_aggregator,
      )

  def test_weighted_fed_avg_with_only_secure_aggregation(self):
    model_fn = model_examples.LinearRegression
    learning_process = fed_avg.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=sgdm.build_sgdm(),
        model_aggregator=model_update_aggregator.secure_aggregator(
            weighted=True
        ),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    self.assertFalse(
        federated_language.framework.computation_contains_unsecure_aggregation(
            learning_process.next
        )
    )

  def test_unweighted_fed_avg_with_only_secure_aggregation(self):
    model_fn = model_examples.LinearRegression
    learning_process = fed_avg.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=sgdm.build_sgdm(),
        model_aggregator=model_update_aggregator.secure_aggregator(
            weighted=False
        ),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    self.assertFalse(
        federated_language.framework.computation_contains_unsecure_aggregation(
            learning_process.next
        )
    )


class FunctionalFedAvgTest(parameterized.TestCase):
  """Tests construction of the FedAvg training process."""

  @parameterized.named_parameters(
      ('weighted', fed_avg.build_weighted_fed_avg),
      ('unweighted', fed_avg.build_unweighted_fed_avg),
  )
  def test_raises_on_non_callable_or_functional_model(self, constructor):
    with self.assertRaisesRegex(TypeError, 'is not a callable'):
      constructor(
          model_fn=0, client_optimizer_fn=sgdm.build_sgdm(learning_rate=0.1)
      )

  @parameterized.named_parameters(
      ('weighted', fed_avg.build_weighted_fed_avg),
      ('unweighted', fed_avg.build_unweighted_fed_avg),
  )
  def test_weighted_fed_avg_with_only_secure_aggregation(self, constructor):
    model = test_models.build_functional_linear_regression()
    learning_process = constructor(
        model_fn=model,
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=0.1),
        server_optimizer_fn=sgdm.build_sgdm(),
        model_aggregator=model_update_aggregator.secure_aggregator(
            weighted=constructor is fed_avg.build_weighted_fed_avg
        ),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    self.assertFalse(
        federated_language.framework.computation_contains_unsecure_aggregation(
            learning_process.next
        )
    )


_BATCH_SIZE = 3
_FEATURE_DIM = 2
_LEARNING_RATE = 0.1
_NUM_ROUNDS = 2
# Substrings (lower case) of op types that must not be in the graphs of a
# computation that uses tensor client datasets.
_FORBIDDEN_OP_TYPE_SUBSTRINGS = ('dataset', 'iterator', 'variant')
_WHILE_OP_TYPES = frozenset(['While', 'StatelessWhile'])
# Pairs of `(num_examples, num_batches)` of the clients. If `num_batches` is
# `None`, the tensor client dataset has no padding batches. The TFF runtime
# does not accept a `tf.data.Dataset` argument with a partial batch or with no
# elements, so the clients that are compared with a `tf.data.Dataset` have only
# full batches.
_CLIENT_CASES = ((6, 4), (6, None), (3, None), (9, 5))
# Clients with partial batches and with no examples. `model_delta_client_work`
# tests compare these cases with `tf.data.Dataset` client updates.
_PARTIAL_CLIENT_CASES = ((7, 5), (2, None), (0, 2), (0, None))


def _create_examples(num_examples: int) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed=num_examples)
  x = rng.uniform(size=[num_examples, _FEATURE_DIM])
  y = rng.uniform(size=[num_examples, 1])
  return x.astype(np.float32), y.astype(np.float32)


def _create_batched_dataset(x: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(x=x, y=y)
  ).batch(_BATCH_SIZE, drop_remainder=False)


def _create_tensor_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_batches: int | None = None,
) -> collections.OrderedDict[str, object]:
  """Returns the tensor client dataset of `_create_batched_dataset(x, y)`."""
  num_examples = x.shape[0]
  num_valid_batches = -(-num_examples // _BATCH_SIZE)
  if num_batches is None:
    num_batches = num_valid_batches
  num_padded_examples = num_batches * _BATCH_SIZE

  def pad_and_reshape(array):
    padded = np.zeros(
        [num_padded_examples, *array.shape[1:]], dtype=array.dtype
    )
    padded[:num_examples] = array
    return padded.reshape([num_batches, _BATCH_SIZE, *array.shape[1:]])

  valid_example_mask = np.arange(num_padded_examples) < num_examples
  return collections.OrderedDict([
      (
          model_delta_client_work.TENSOR_DATASET_DATA_KEY,
          collections.OrderedDict(x=pad_and_reshape(x), y=pad_and_reshape(y)),
      ),
      (
          model_delta_client_work.TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY,
          valid_example_mask.reshape([num_batches, _BATCH_SIZE]),
      ),
      (
          model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY,
          np.int32(num_valid_batches),
      ),
  ])


def _create_functional_model() -> functional.FunctionalModel:

  def build_metrics_fn():
    return collections.OrderedDict(
        loss=keras.metrics.MeanSquaredError(),
        num_examples=counters.NumExamplesCounter(),
        num_batches=counters.NumBatchesCounter(),
    )

  return functional.functional_model_from_keras(
      model_examples.build_linear_regression_keras_functional_model(
          feature_dims=_FEATURE_DIM
      ),
      loss_fn=keras.losses.MeanSquaredError(),
      input_spec=collections.OrderedDict(
          x=tf.TensorSpec([None, _FEATURE_DIM], tf.float32),
          y=tf.TensorSpec([None, 1], tf.float32),
      ),
      metrics_constructor=build_metrics_fn,
  )


def _get_tensorflow_op_types(
    comp: federated_language.framework.ConcreteComputation,
) -> set[str]:
  """Returns the op types of all TensorFlow graphs in `comp`."""
  op_types = set()

  def visit(building_block):
    if not isinstance(
        building_block, federated_language.framework.CompiledComputation
    ):
      return
    proto = building_block.to_proto()
    if proto.WhichOneof('computation') != 'tensorflow':
      return
    graph_def = serialization_utils.unpack_graph_def(proto.tensorflow.graph_def)
    nodes = itertools.chain(
        graph_def.node, *[f.node_def for f in graph_def.library.function]
    )
    op_types.update(node.op for node in nodes)

  federated_language.framework.visit_postorder(comp.to_building_block(), visit)
  return op_types


class TensorDatasetFedAvgTest(tf.test.TestCase, parameterized.TestCase):
  """Tests FedAvg with `use_tensor_dataset=True`."""

  def test_raises_on_variable_model_fn(self):
    with self.assertRaisesRegex(TypeError, 'use_tensor_dataset'):
      fed_avg.build_weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(learning_rate=0.1),
          use_tensor_dataset=True,
      )

  def test_does_not_raise_on_variable_model_fn_without_tensor_dataset(self):
    fed_avg.build_weighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=0.1),
        use_tensor_dataset=False,
    )

  def test_next_has_expected_client_data_type(self):
    model = _create_functional_model()

    tensor_process = fed_avg.build_weighted_fed_avg(
        model_fn=model,
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=_LEARNING_RATE),
        use_tensor_dataset=True,
    )
    dataset_process = fed_avg.build_weighted_fed_avg(
        model_fn=model,
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=_LEARNING_RATE),
    )

    expected_client_data_type = federated_language.FederatedType(
        model_delta_client_work.tensor_dataset_type_from_input_spec(
            model.input_spec
        ),
        federated_language.CLIENTS,
    )
    tensor_next_type = tensor_process.next.type_signature
    dataset_next_type = dataset_process.next.type_signature
    self.assertTrue(
        tensor_next_type.parameter[1].is_equivalent_to(
            expected_client_data_type
        ),
        msg=f'{tensor_next_type.parameter[1]} != {expected_client_data_type}',
    )
    # Only the type of the client data is different.
    self.assertTrue(
        tensor_next_type.parameter[0].is_equivalent_to(
            dataset_next_type.parameter[0]
        )
    )
    self.assertTrue(
        tensor_next_type.result.is_equivalent_to(dataset_next_type.result)
    )
    self.assertTrue(
        tensor_process.initialize.type_signature.is_equivalent_to(
            dataset_process.initialize.type_signature
        )
    )

  def test_graph_has_while_loop_and_no_dataset_ops(self):
    process = fed_avg.build_weighted_fed_avg(
        model_fn=_create_functional_model(),
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=_LEARNING_RATE),
        use_tensor_dataset=True,
    )

    op_types = _get_tensorflow_op_types(process.next)

    forbidden_op_types = [
        op_type
        for op_type in op_types
        if any(s in op_type.lower() for s in _FORBIDDEN_OP_TYPE_SUBSTRINGS)
    ]
    self.assertEmpty(forbidden_op_types)
    self.assertNotEmpty(op_types & _WHILE_OP_TYPES)

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_tensor_dataset_matches_dataset(self, weighting):
    model = _create_functional_model()
    tensor_process = fed_avg.build_weighted_fed_avg(
        model_fn=model,
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=_LEARNING_RATE),
        client_weighting=weighting,
        use_tensor_dataset=True,
    )
    dataset_process = fed_avg.build_weighted_fed_avg(
        model_fn=model,
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=_LEARNING_RATE),
        client_weighting=weighting,
    )
    examples = [_create_examples(n) for n, _ in _CLIENT_CASES]
    tensor_client_data = [
        _create_tensor_dataset(x, y, num_batches=num_batches)
        for (x, y), (_, num_batches) in zip(examples, _CLIENT_CASES)
    ]
    dataset_client_data = [_create_batched_dataset(x, y) for x, y in examples]
    tensor_state = tensor_process.initialize()
    dataset_state = dataset_process.initialize()
    initial_model_weights = tensor_process.get_model_weights(tensor_state)

    for _ in range(_NUM_ROUNDS):
      tensor_output = tensor_process.next(tensor_state, tensor_client_data)
      dataset_output = dataset_process.next(dataset_state, dataset_client_data)
      tensor_state = tensor_output.state
      dataset_state = dataset_output.state

      self.assertAllClose(
          tensor_process.get_model_weights(tensor_state),
          dataset_process.get_model_weights(dataset_state),
      )
      self.assertAllClose(
          tensor_output.metrics['client_work'],
          dataset_output.metrics['client_work'],
      )
      train_metrics = tensor_output.metrics['client_work']['train']
      self.assertEqual(
          train_metrics['num_examples'], sum(n for n, _ in _CLIENT_CASES)
      )
      self.assertEqual(
          train_metrics['num_batches'],
          sum(-(-n // _BATCH_SIZE) for n, _ in _CLIENT_CASES),
      )
    # Make sure that the training changed the model.
    self.assertNotAllClose(
        tensor_process.get_model_weights(tensor_state).trainable,
        initial_model_weights.trainable,
    )

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_tensor_dataset_with_partial_and_empty_clients(self, weighting):
    model = _create_functional_model()
    process = fed_avg.build_weighted_fed_avg(
        model_fn=model,
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=_LEARNING_RATE),
        client_weighting=weighting,
        use_tensor_dataset=True,
    )
    client_data = [
        _create_tensor_dataset(
            *_create_examples(num_examples), num_batches=num_batches
        )
        for num_examples, num_batches in _PARTIAL_CLIENT_CASES
    ]
    state = process.initialize()
    initial_model_weights = process.get_model_weights(state)

    output = process.next(state, client_data)

    train_metrics = output.metrics['client_work']['train']
    self.assertEqual(
        train_metrics['num_examples'], sum(n for n, _ in _PARTIAL_CLIENT_CASES)
    )
    self.assertEqual(
        train_metrics['num_batches'],
        sum(-(-n // _BATCH_SIZE) for n, _ in _PARTIAL_CLIENT_CASES),
    )
    self.assertEqual(train_metrics['num_non_finite_clients'], 0)
    self.assertEqual(output.metrics['finalizer']['update_non_finite'], 0)
    model_weights = process.get_model_weights(output.state)
    for weight in tf.nest.flatten(model_weights.trainable):
      self.assertTrue(np.all(np.isfinite(weight)), msg=f'{weight}')
    self.assertNotAllClose(
        model_weights.trainable, initial_model_weights.trainable
    )


if __name__ == '__main__':
  cpp_execution_contexts.set_sync_local_cpp_execution_context()
  absltest.main()
