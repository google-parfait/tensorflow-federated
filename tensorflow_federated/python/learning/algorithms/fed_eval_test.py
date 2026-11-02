# Copyright 2022, The TensorFlow Federated Authors.
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
import functools
import itertools
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
import keras
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import cpp_execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_backend import serialization_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_test_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import type_conversions
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.algorithms import fed_eval
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.metrics import sum_aggregation_factory
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.learning.templates import model_delta_client_work

# Convenience aliases.
FederatedType = federated_language.FederatedType
FunctionType = federated_language.FunctionType
SequenceType = federated_language.SequenceType
StructType = federated_language.StructType
TensorType = federated_language.TensorType


class TestModel(variable.VariableModel):

  def __init__(self):
    self._variables = collections.namedtuple('Vars', 'max_temp num_over')(
        max_temp=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=[]),
            name='max_temp',
            trainable=True,
        ),
        num_over=tf.Variable(0.0, name='num_over', trainable=False),
    )

  @property
  def trainable_variables(self):
    return [self._variables.max_temp]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [self._variables.num_over]

  @property
  def input_spec(self):
    return collections.OrderedDict(temp=tf.TensorSpec([None], tf.float32))

  @tf.function
  def predict_on_batch(self, batch, training=True):
    del training  # Unused.
    return tf.zeros_like(batch['temp'])

  @tf.function
  def forward_pass(self, batch, training=True):
    assert not training
    num_over = tf.reduce_sum(
        tf.cast(tf.greater(batch['temp'], self._variables.max_temp), tf.float32)
    )
    self._variables.num_over.assign_add(num_over)
    loss = tf.constant(0.0)
    predictions = self.predict_on_batch(batch, training)
    return variable.BatchOutput(
        loss=loss,
        predictions=predictions,
        num_examples=tf.shape(predictions)[0],
    )

  @tf.function
  def report_local_unfinalized_metrics(self):
    return collections.OrderedDict(num_over=self._variables.num_over)

  def metric_finalizers(self):
    return collections.OrderedDict(num_over=tf.function(func=lambda x: x))

  @tf.function
  def reset_metrics(self):
    """Resets metrics variables to initial value."""
    for var in self.local_variables:
      var.assign(tf.zeros_like(var))


# TODO: b/319261270 - Avoid the need for inferring types here, if possible.
def _get_metrics_type(metrics: collections.OrderedDict[str, Any]):
  def _tensor_spec_from_tensor_like(x):
    x_as_tensor = tf.convert_to_tensor(x)
    return tensorflow_types.to_type((x_as_tensor.dtype, x_as_tensor.shape))

  finalizer_spec = tf.nest.map_structure(_tensor_spec_from_tensor_like, metrics)
  return federated_language.StructWithPythonType(
      finalizer_spec, collections.OrderedDict
  )


def _create_custom_metrics_aggregation_process(
    metric_finalizers, local_unfinalized_metrics_type
):
  """Creates an `AggregationProcess` gets maximum and finalizes metrics."""

  @tensorflow_computation.tf_computation
  def create_all_zero_state():
    return type_conversions.structure_from_tensor_type_tree(
        lambda t: tf.zeros(shape=t.shape, dtype=t.dtype),
        local_unfinalized_metrics_type,
    )

  @federated_language.federated_computation()
  def init_fn():
    return federated_language.federated_eval(
        create_all_zero_state, federated_language.SERVER
    )

  @tensorflow_computation.tf_computation(
      local_unfinalized_metrics_type, local_unfinalized_metrics_type
  )
  def get_max_unfinalized_metrics(
      unfinalized_metrics, new_max_unfinalized_metrics
  ):
    return tf.nest.map_structure(
        tf.math.maximum, unfinalized_metrics, new_max_unfinalized_metrics
    )

  @federated_language.federated_computation(
      init_fn.type_signature.result,
      federated_language.FederatedType(
          local_unfinalized_metrics_type, federated_language.CLIENTS
      ),
  )
  def next_fn(state, unfinalized_metrics):
    max_unfinalized_metrics = federated_language.federated_max(
        unfinalized_metrics
    )

    state = federated_language.federated_map(
        get_max_unfinalized_metrics, (state, max_unfinalized_metrics)
    )

    @tensorflow_computation.tf_computation(local_unfinalized_metrics_type)
    def finalizer_computation(unfinalized_metrics):
      if callable(metric_finalizers):
        return metric_finalizers(unfinalized_metrics)
      finalized_metrics = collections.OrderedDict()
      for metric_name, metric_finalizer in metric_finalizers.items():
        finalized_metrics[metric_name] = metric_finalizer(
            unfinalized_metrics[metric_name]
        )
      return finalized_metrics

    current_round_metrics = federated_language.federated_map(
        finalizer_computation, max_unfinalized_metrics
    )
    total_rounds_metrics = federated_language.federated_map(
        finalizer_computation, state
    )

    return measured_process.MeasuredProcessOutput(
        state=state,
        result=federated_language.federated_zip(
            (current_round_metrics, total_rounds_metrics)
        ),
        measurements=federated_language.federated_value(
            (), federated_language.SERVER
        ),
    )

  return aggregation_process.AggregationProcess(init_fn, next_fn)


class FedEvalProcessTest(tf.test.TestCase):

  def test_fed_eval_process_type_properties(self):
    model_fn = TestModel
    test_model = model_fn()
    model_weights_type = model_weights_lib.weights_type_from_model(test_model)

    unfinalized_metrics = test_model.report_local_unfinalized_metrics()
    local_unfinalized_metrics_type = _get_metrics_type(unfinalized_metrics)

    metric_finalizers = test_model.metric_finalizers()
    finalized_metrics = collections.OrderedDict()
    for metric, finalizer in metric_finalizers.items():
      finalized_metrics[metric] = finalizer(unfinalized_metrics[metric])
    finalized_metrics_type = _get_metrics_type(finalized_metrics)

    metics_aggregator = sum_aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers
    ).create(local_unfinalized_metrics_type)
    metics_aggregator_state_type = (
        metics_aggregator.initialize.type_signature.result.member
    )

    eval_process = fed_eval.build_fed_eval(model_fn)
    self.assertIsInstance(eval_process, learning_process.LearningProcess)

    expected_state_type = federated_language.FederatedType(
        composers.LearningAlgorithmState(
            global_model_weights=model_weights_type,
            distributor=(),
            client_work=metics_aggregator_state_type,
            aggregator=collections.OrderedDict(
                value_sum_process=(), weight_sum_process=()
            ),
            finalizer=(),
        ),
        federated_language.SERVER,
    )
    expected_metrics_type = federated_language.FederatedType(
        collections.OrderedDict(
            distributor=(),
            client_work=collections.OrderedDict(
                eval=collections.OrderedDict(
                    current_round_metrics=finalized_metrics_type,
                    total_rounds_metrics=finalized_metrics_type,
                )
            ),
            aggregator=collections.OrderedDict(mean_value=(), mean_weight=()),
            finalizer=(),
        ),
        federated_language.SERVER,
    )
    self.assertTrue(
        eval_process.initialize.type_signature.is_equivalent_to(
            FunctionType(parameter=None, result=expected_state_type),
        )
    )
    self.assertTrue(
        eval_process.next.type_signature.is_equivalent_to(
            FunctionType(
                parameter=StructType([
                    ('state', expected_state_type),
                    (
                        'client_data',
                        federated_language.FederatedType(
                            SequenceType(
                                StructType([(
                                    'temp',
                                    TensorType(dtype=np.float32, shape=[None]),
                                )])
                            ),
                            federated_language.CLIENTS,
                        ),
                    ),
                ]),
                result=learning_process.LearningProcessOutput(
                    state=expected_state_type, metrics=expected_metrics_type
                ),
            )
        )
    )
    self.assertTrue(
        eval_process.get_model_weights.type_signature.is_equivalent_to(
            FunctionType(
                parameter=expected_state_type.member, result=model_weights_type
            )
        )
    )
    self.assertTrue(
        eval_process.set_model_weights.type_signature.is_equivalent_to(
            FunctionType(
                parameter=StructType([
                    ('state', expected_state_type.member),
                    ('model_weights', model_weights_type),
                ]),
                result=expected_state_type.member,
            )
        )
    )

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_fed_eval_process_execution(self):
    eval_process = fed_eval.build_fed_eval(TestModel)

    # Update the state with the model weights to be evaluated, and verify that
    # the `get_model_weights` method returns the same model weights.
    state = eval_process.initialize()
    model_weights = model_weights_lib.ModelWeights(
        trainable=[5.0], non_trainable=[]
    )
    new_state = eval_process.set_model_weights(
        state, model_weights_lib.ModelWeights(trainable=[5.0], non_trainable=[])
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        model_weights,
        eval_process.get_model_weights(new_state),
    )

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    clients_data = [
        [_temp_dict([1.0, 10.0, 2.0, 7.0]), _temp_dict([6.0, 11.0, 0.0, 0.0])],
        [_temp_dict([9.0, 12.0, 13.0, 0.0])],
        [_temp_dict([1.0, 0.0, 0.0, 0.0]), _temp_dict([22.0, 23.0, 0.0, 0.0])],
    ]

    output = eval_process.next(new_state, clients_data)
    tf.nest.map_structure(
        self.assertAllEqual,
        model_weights,
        eval_process.get_model_weights(output.state),
    )
    _, accumulated_unfinalized_metrics = output.state.client_work
    self.assertEqual(
        accumulated_unfinalized_metrics,
        collections.OrderedDict([('num_over', 9.0)]),
    )
    self.assertEqual(
        output.metrics['client_work'],
        collections.OrderedDict(
            eval=collections.OrderedDict(
                current_round_metrics=collections.OrderedDict(num_over=9.0),
                total_rounds_metrics=collections.OrderedDict(num_over=9.0),
            )
        ),
    )

  def test_fed_eval_with_model_distributor(self):
    model_weights_type = model_weights_lib.weights_type_from_model(TestModel)

    def test_distributor():

      @federated_language.federated_computation()
      def init_fn():
        return federated_language.federated_value((), federated_language.SERVER)

      @federated_language.federated_computation(
          init_fn.type_signature.result,
          federated_language.FederatedType(
              model_weights_type, federated_language.SERVER
          ),
      )
      def next_fn(state, value):
        return measured_process.MeasuredProcessOutput(
            state,
            federated_language.federated_broadcast(value),
            federated_language.federated_value((), federated_language.SERVER),
        )

      return distributors.DistributionProcess(init_fn, next_fn)

    eval_process = fed_eval.build_fed_eval(TestModel, test_distributor())
    self.assertTrue(
        eval_process.initialize.type_signature.result.member.distributor.is_equivalent_to(
            test_distributor().initialize.type_signature.result.member
        )
    )
    self.assertTrue(
        eval_process.next.type_signature.result.state.member.distributor.is_equivalent_to(
            test_distributor().next.type_signature.result.state.member
        )
    )
    self.assertTrue(
        eval_process.next.type_signature.result.metrics.member.distributor.is_equivalent_to(
            test_distributor().next.type_signature.result.measurements.member
        )
    )

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_fed_eval_with_metrics_aggregation_process(self):
    model_fn = TestModel
    test_model = model_fn()
    unfinalized_metrics = test_model.report_local_unfinalized_metrics()
    local_unfinalized_metrics_type = _get_metrics_type(unfinalized_metrics)
    metric_finalizers = test_model.metric_finalizers()

    eval_process = fed_eval.build_fed_eval(
        model_fn,
        metrics_aggregation_process=_create_custom_metrics_aggregation_process(
            metric_finalizers, local_unfinalized_metrics_type
        ),
    )
    state = eval_process.initialize()
    self.assertEqual(
        state.client_work, collections.OrderedDict([('num_over', 0.0)])
    )

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    clients_data = [
        [_temp_dict([1.0, 10.0, 2.0, 7.0]), _temp_dict([6.0, 11.0, 0.0, 0.0])],
        [_temp_dict([9.0, 12.0, 13.0, 0.0])],
        [_temp_dict([1.0, 0.0, 0.0, 0.0]), _temp_dict([22.0, 23.0, 0.0, 0.0])],
    ]

    output = eval_process.next(state, clients_data)
    self.assertEqual(
        output.state.client_work, collections.OrderedDict([('num_over', 6.0)])
    )
    self.assertEqual(
        output.metrics['client_work'],
        collections.OrderedDict(
            eval=collections.OrderedDict(
                current_round_metrics=collections.OrderedDict(num_over=6.0),
                total_rounds_metrics=collections.OrderedDict(num_over=6.0),
            )
        ),
    )

  def test_invalid_metrics_aggregation_process_raises(self):
    test_model = TestModel()
    metrics_aggregator = aggregator.sum_then_finalize(
        test_model.metric_finalizers()
    )
    with self.assertRaisesRegex(TypeError, 'AggregationProcess'):
      fed_eval.build_fed_eval(
          TestModel, metrics_aggregation_process=metrics_aggregator
      )

  def test_invalid_model_distributor_raises(self):
    model_distributor = tensorflow_computation.tf_computation(lambda x: x)
    with self.assertRaisesRegex(TypeError, 'DistributionProcess'):
      fed_eval.build_fed_eval(TestModel, model_distributor=model_distributor)

  def test_construction_calls_model_fn(self):
    # Assert that the process building does not call `model_fn` too many times.
    # `model_fn` can potentially be expensive (loading weights, processing, etc
    # ).
    mock_model_fn = mock.Mock(side_effect=TestModel)
    fed_eval.build_fed_eval(mock_model_fn)
    self.assertEqual(mock_model_fn.call_count, 3)


class FunctionalFedEvalProcessTest(tf.test.TestCase):

  def create_test_datasets(self) -> tf.data.Dataset:
    dataset1 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]],
        )
    )
    dataset2 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            y=[[1.0], [2.0], [3.0], [4.0]],
        )
    )
    return [dataset1.repeat(3).batch(3), dataset2.repeat(3).batch(3)]

  def test_raises_on_non_callable_or_functional_model(self):
    with self.assertRaisesRegex(TypeError, 'is not a callable'):
      fed_eval.build_fed_eval(model_fn=0)

  @tensorflow_test_utils.skip_test_for_gpu
  def test_functional_evaluation_matches_non_functional(self):
    datasets = self.create_test_datasets()
    batch_type = tensorflow_types.to_type(datasets[0].element_spec)
    loss_fn = keras.losses.MeanSquaredError
    keras_model_fn = functools.partial(
        model_examples.build_linear_regression_keras_functional_model,
        feature_dims=2,
    )

    # Defining artifacts using `tff.learning.models.VariableModel`
    def tff_model_fn():
      keras_model = keras_model_fn()
      return keras_utils.from_keras_model(
          keras_model, loss=loss_fn(), input_spec=batch_type
      )

    eval_process = fed_eval.build_fed_eval(tff_model_fn)
    eval_state = eval_process.initialize()
    eval_output = eval_process.next(eval_state, datasets)

    # Defining artifacts using `tff.learning.models.FunctionalModel`
    def build_metrics_fn():
      return collections.OrderedDict(
          loss=keras.metrics.MeanSquaredError(),
          num_examples=counters.NumExamplesCounter(),
          num_batches=counters.NumBatchesCounter(),
      )

    functional_model = functional.functional_model_from_keras(
        keras_model=keras_model_fn(),
        loss_fn=loss_fn(),
        input_spec=batch_type,
        metrics_constructor=build_metrics_fn,
    )
    functional_eval_process = fed_eval.build_fed_eval(functional_model)
    functional_eval_state = functional_eval_process.initialize()
    functional_eval_output = functional_eval_process.next(
        functional_eval_state, datasets
    )
    self.assertDictEqual(eval_output.metrics, functional_eval_output.metrics)

  @tensorflow_test_utils.skip_test_for_gpu
  def test_functional_evaluation_with_metrics_aggregation_process_matches_non_functional(
      self,
  ):
    datasets = self.create_test_datasets()
    batch_type = tensorflow_types.to_type(datasets[0].element_spec)
    loss_fn = keras.losses.MeanSquaredError
    keras_model_fn = functools.partial(
        model_examples.build_linear_regression_keras_functional_model,
        feature_dims=2,
    )

    def tff_model_fn():
      keras_model = keras_model_fn()
      return keras_utils.from_keras_model(
          keras_model, loss=loss_fn(), input_spec=batch_type
      )

    with tf.Graph().as_default():
      test_model = tff_model_fn()
      unfinalized_metrics = test_model.report_local_unfinalized_metrics()
      local_unfinalized_metrics_type = _get_metrics_type(unfinalized_metrics)
      metric_finalizers = test_model.metric_finalizers()

    eval_process = fed_eval.build_fed_eval(
        tff_model_fn,
        metrics_aggregation_process=_create_custom_metrics_aggregation_process(
            metric_finalizers, local_unfinalized_metrics_type
        ),
    )
    eval_state = eval_process.initialize()
    eval_output = eval_process.next(eval_state, datasets)

    def build_metrics_fn():
      return collections.OrderedDict(
          loss=keras.metrics.MeanSquaredError(),
          num_examples=counters.NumExamplesCounter(),
          num_batches=counters.NumBatchesCounter(),
      )

    functional_model = functional.functional_model_from_keras(
        keras_model=keras_model_fn(),
        loss_fn=loss_fn(),
        input_spec=batch_type,
        metrics_constructor=build_metrics_fn,
    )
    functional_unfinalized_metrics_type = tensorflow_computation.tf_computation(
        functional_model.initialize_metrics_state
    ).type_signature.result
    functional_eval_process = fed_eval.build_fed_eval(
        functional_model,
        metrics_aggregation_process=_create_custom_metrics_aggregation_process(
            functional_model.finalize_metrics,
            functional_unfinalized_metrics_type,
        ),
    )
    functional_eval_state = functional_eval_process.initialize()
    functional_eval_output = functional_eval_process.next(
        functional_eval_state, datasets
    )
    self.assertDictEqual(eval_output.metrics, functional_eval_output.metrics)

  def test_invalid_metrics_aggregation_process_raises(self):
    datasets = self.create_test_datasets()
    batch_type = tensorflow_types.to_type(datasets[0].element_spec)
    functional_model = functional.functional_model_from_keras(
        keras_model=model_examples.build_linear_regression_keras_functional_model(
            feature_dims=2
        ),
        loss_fn=keras.losses.MeanSquaredError(),
        input_spec=batch_type,
    )
    with self.assertRaisesRegex(TypeError, 'AggregationProcess'):
      fed_eval.build_fed_eval(
          functional_model, metrics_aggregation_process='not_a_process'
      )


_BATCH_SIZE = 3
_FEATURE_DIM = 2
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
# Clients with partial batches and with no examples. These are compared with
# metrics that numpy computes.
_PARTIAL_CLIENT_CASES = ((7, 5), (2, None), (0, 2), (0, None), (6, None))
_MODEL_WEIGHT_VALUE = 0.5


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
    pad_value: float = 0.0,
) -> collections.OrderedDict[str, object]:
  """Returns the tensor client dataset of `_create_batched_dataset(x, y)`."""
  num_examples = x.shape[0]
  num_valid_batches = -(-num_examples // _BATCH_SIZE)
  if num_batches is None:
    num_batches = num_valid_batches
  num_padded_examples = num_batches * _BATCH_SIZE

  def pad_and_reshape(array):
    padded = np.full(
        [num_padded_examples, *array.shape[1:]], pad_value, dtype=array.dtype
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


def _get_forbidden_op_types(op_types: set[str]) -> list[str]:
  return sorted(
      op_type
      for op_type in op_types
      if any(s in op_type.lower() for s in _FORBIDDEN_OP_TYPE_SUBSTRINGS)
  )


class TensorDatasetFedEvalProcessTest(tf.test.TestCase, parameterized.TestCase):
  """Tests federated evaluation with `use_tensor_dataset=True`."""

  def test_raises_on_variable_model_fn(self):
    with self.assertRaisesRegex(TypeError, 'use_tensor_dataset'):
      fed_eval.build_fed_eval(TestModel, use_tensor_dataset=True)

  def test_does_not_raise_on_variable_model_fn_without_tensor_dataset(self):
    fed_eval.build_fed_eval(TestModel, use_tensor_dataset=False)

  def test_next_has_expected_client_data_type(self):
    model = _create_functional_model()

    tensor_process = fed_eval.build_fed_eval(model, use_tensor_dataset=True)
    dataset_process = fed_eval.build_fed_eval(model)

    tensor_next_type = tensor_process.next.type_signature
    dataset_next_type = dataset_process.next.type_signature
    client_data_type = tensor_next_type.parameter['client_data']
    self.assertIsInstance(client_data_type, FederatedType)
    self.assertEqual(client_data_type.placement, federated_language.CLIENTS)
    expected_member_type = StructType([
        (
            model_delta_client_work.TENSOR_DATASET_DATA_KEY,
            StructType([
                ('x', TensorType(np.float32, [None, None, _FEATURE_DIM])),
                ('y', TensorType(np.float32, [None, None, 1])),
            ]),
        ),
        (
            model_delta_client_work.TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY,
            TensorType(np.bool_, [None, None]),
        ),
        (
            model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY,
            TensorType(np.int32),
        ),
    ])
    self.assertTrue(
        client_data_type.member.is_equivalent_to(expected_member_type),
        msg=f'{client_data_type.member} != {expected_member_type}',
    )
    # Only the type of the client data is different.
    self.assertTrue(
        tensor_next_type.parameter['state'].is_equivalent_to(
            dataset_next_type.parameter['state']
        )
    )
    self.assertTrue(
        tensor_next_type.result.is_equivalent_to(dataset_next_type.result)
    )

  def test_graph_has_while_loop_and_no_dataset_ops(self):
    process = fed_eval.build_fed_eval(
        _create_functional_model(), use_tensor_dataset=True
    )

    op_types = _get_tensorflow_op_types(process.next)

    self.assertEmpty(_get_forbidden_op_types(op_types))
    self.assertNotEmpty(op_types & _WHILE_OP_TYPES)

  def test_dataset_graph_has_dataset_ops(self):
    # Make sure that `_get_tensorflow_op_types` finds the ops of the client
    # computation. Thus the test above can detect dataset ops.
    process = fed_eval.build_fed_eval(_create_functional_model())

    op_types = _get_tensorflow_op_types(process.next)

    self.assertNotEmpty(_get_forbidden_op_types(op_types))

  @parameterized.named_parameters(
      ('zero_padding', 0.0), ('nan_padding', np.nan)
  )
  def test_tensor_dataset_metrics_match_dataset_metrics(self, pad_value):
    model = _create_functional_model()
    tensor_process = fed_eval.build_fed_eval(model, use_tensor_dataset=True)
    dataset_process = fed_eval.build_fed_eval(model)
    examples = [_create_examples(n) for n, _ in _CLIENT_CASES]
    tensor_client_data = [
        _create_tensor_dataset(
            x, y, num_batches=num_batches, pad_value=pad_value
        )
        for (x, y), (_, num_batches) in zip(examples, _CLIENT_CASES)
    ]
    dataset_client_data = [_create_batched_dataset(x, y) for x, y in examples]
    # Use weights that are not zero, so that the loss depends on `x`.
    model_weights = tf.nest.map_structure(
        lambda w: np.full_like(w, _MODEL_WEIGHT_VALUE),
        tensor_process.get_model_weights(tensor_process.initialize()),
    )
    tensor_state = tensor_process.set_model_weights(
        tensor_process.initialize(), model_weights
    )
    dataset_state = dataset_process.set_model_weights(
        dataset_process.initialize(), model_weights
    )

    tensor_output = tensor_process.next(tensor_state, tensor_client_data)
    dataset_output = dataset_process.next(dataset_state, dataset_client_data)

    self.assertAllClose(tensor_output.metrics, dataset_output.metrics)
    eval_metrics = tensor_output.metrics['client_work']['eval'][
        'current_round_metrics'
    ]
    self.assertEqual(
        eval_metrics['num_examples'], sum(n for n, _ in _CLIENT_CASES)
    )
    self.assertEqual(
        eval_metrics['num_batches'],
        sum(-(-n // _BATCH_SIZE) for n, _ in _CLIENT_CASES),
    )
    self.assertGreater(eval_metrics['loss'], 0.0)

  def _evaluate_tensor_datasets(self, client_cases, pad_value):
    """Returns the eval metrics and the examples of the clients."""
    model = _create_functional_model()
    process = fed_eval.build_fed_eval(model, use_tensor_dataset=True)
    examples = [_create_examples(n) for n, _ in client_cases]
    client_data = [
        _create_tensor_dataset(
            x, y, num_batches=num_batches, pad_value=pad_value
        )
        for (x, y), (_, num_batches) in zip(examples, client_cases)
    ]
    model_weights = tf.nest.map_structure(
        lambda w: np.full_like(w, _MODEL_WEIGHT_VALUE),
        process.get_model_weights(process.initialize()),
    )
    state = process.set_model_weights(process.initialize(), model_weights)

    output = process.next(state, client_data)

    eval_metrics = output.metrics['client_work']['eval'][
        'current_round_metrics'
    ]
    return eval_metrics, examples

  @parameterized.named_parameters(
      ('zero_padding', 0.0), ('nan_padding', np.nan)
  )
  def test_tensor_dataset_metrics_with_partial_and_empty_clients(
      self, pad_value
  ):
    eval_metrics, examples = self._evaluate_tensor_datasets(
        _PARTIAL_CLIENT_CASES, pad_value
    )

    # The linear model has all weights equal to `_MODEL_WEIGHT_VALUE`. The
    # summed MSE metric is the MSE over the examples of all clients.
    x = np.concatenate([x for x, _ in examples])
    y = np.concatenate([y for _, y in examples])
    predictions = (
        np.sum(x, axis=1, keepdims=True) * _MODEL_WEIGHT_VALUE
        + _MODEL_WEIGHT_VALUE
    )
    expected_loss = np.mean(np.square(predictions - y))
    self.assertAllClose(eval_metrics['loss'], expected_loss, rtol=1e-5)
    self.assertEqual(
        eval_metrics['num_examples'], sum(n for n, _ in _PARTIAL_CLIENT_CASES)
    )
    self.assertEqual(
        eval_metrics['num_batches'],
        sum(-(-n // _BATCH_SIZE) for n, _ in _PARTIAL_CLIENT_CASES),
    )

  @parameterized.named_parameters(
      ('no_batches', None, 0.0),
      ('zero_padding_batches', 2, 0.0),
      ('nan_padding_batches', 2, np.nan),
  )
  def test_tensor_dataset_metrics_with_only_empty_clients(
      self, num_batches, pad_value
  ):
    eval_metrics, _ = self._evaluate_tensor_datasets(
        ((0, num_batches), (0, num_batches)), pad_value
    )

    self.assertAllClose(eval_metrics['loss'], 0.0)
    self.assertEqual(eval_metrics['num_examples'], 0)
    self.assertEqual(eval_metrics['num_batches'], 0)


if __name__ == '__main__':
  cpp_execution_contexts.set_sync_local_cpp_execution_context()
  absltest.main()
