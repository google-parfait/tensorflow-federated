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
from typing import Any
from unittest import mock

from absl.testing import absltest
import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_test_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import type_conversions
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.algorithms import fed_eval
from tensorflow_federated.python.learning.algorithms import fed_multi_model_eval
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


class FedMultiModelEvalProcessTest(tf.test.TestCase):

  def test_fed_multi_model_eval_process_type_properties(self):
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

    eval_process = fed_multi_model_eval.build_fed_multi_model_eval(
        (TestModel, TestModel)
    )
    self.assertIsInstance(eval_process, learning_process.LearningProcess)

    expected_model_weights_type = {
        'a': model_weights_type,
        'b': model_weights_type,
    }
    expected_state_type = federated_language.FederatedType(
        composers.LearningAlgorithmState(
            global_model_weights=expected_model_weights_type,
            distributor=(),
            client_work={
                'a': metics_aggregator_state_type,
                'b': metics_aggregator_state_type,
            },
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
            client_work={
                'a': collections.OrderedDict(
                    eval=collections.OrderedDict(
                        current_round_metrics=finalized_metrics_type,
                        total_rounds_metrics=finalized_metrics_type,
                    )
                ),
                'b': collections.OrderedDict(
                    eval=collections.OrderedDict(
                        current_round_metrics=finalized_metrics_type,
                        total_rounds_metrics=finalized_metrics_type,
                    )
                ),
            },
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
                parameter=expected_state_type.member,
                result=expected_model_weights_type,
            )
        )
    )
    self.assertTrue(
        eval_process.set_model_weights.type_signature.is_equivalent_to(
            FunctionType(
                parameter=StructType([
                    ('state', expected_state_type.member),
                    ('model_weights', expected_model_weights_type),
                ]),
                result=expected_state_type.member,
            )
        )
    )

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_fed_multi_model_eval_process_execution(self):
    eval_process = fed_multi_model_eval.build_fed_multi_model_eval(
        (TestModel, TestModel)
    )

    # Update the state with the model weights to be evaluated, and verify that
    # the `get_model_weights` method returns the same model weights.
    state = eval_process.initialize()
    model_weights = {
        'a': model_weights_lib.ModelWeights(trainable=[5.0], non_trainable=[]),
        'b': model_weights_lib.ModelWeights(trainable=[10.0], non_trainable=[]),
    }
    new_state = eval_process.set_model_weights(state, model_weights)
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
    self.assertSequenceEqual(list(output.state.client_work.keys()), ['a', 'b'])
    self.assertSequenceEqual(
        list(output.metrics['client_work'].keys()), ['a', 'b']
    )
    for (
        client_work_state_per_model,
        client_work_metrics_per_model,
        expected_num_over,
    ) in zip(
        output.state.client_work.values(),
        output.metrics['client_work'].values(),
        [9.0, 5.0],
    ):
      _, accumulated_unfinalized_metrics = client_work_state_per_model
      self.assertEqual(
          accumulated_unfinalized_metrics,
          collections.OrderedDict([('num_over', expected_num_over)]),
      )
      self.assertEqual(
          client_work_metrics_per_model,
          collections.OrderedDict(
              eval=collections.OrderedDict(
                  current_round_metrics=collections.OrderedDict(
                      num_over=expected_num_over
                  ),
                  total_rounds_metrics=collections.OrderedDict(
                      num_over=expected_num_over
                  ),
              )
          ),
      )

  def test_fed_multi_model_eval_with_model_distributor(self):
    model_weights_type = model_weights_lib.weights_type_from_model(TestModel)
    model_weights_type = {
        'a': model_weights_type,
        'b': model_weights_type,
    }

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

    eval_process = fed_multi_model_eval.build_fed_multi_model_eval(
        (TestModel, TestModel), test_distributor()
    )
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
  def test_fed_multi_model_eval_with_metrics_aggregation_process(self):
    model_fn = TestModel
    test_model = model_fn()
    unfinalized_metrics = test_model.report_local_unfinalized_metrics()
    local_unfinalized_metrics_type = _get_metrics_type(unfinalized_metrics)
    metric_finalizers = test_model.metric_finalizers()
    metrics_aggregation_process = _create_custom_metrics_aggregation_process(
        metric_finalizers, local_unfinalized_metrics_type
    )

    eval_process = fed_multi_model_eval.build_fed_multi_model_eval(
        (TestModel, TestModel),
        metrics_aggregation_processes=(
            metrics_aggregation_process,
            metrics_aggregation_process,
        ),
    )
    state = eval_process.initialize()
    self.assertEqual(
        state.client_work,
        {
            'a': collections.OrderedDict([('num_over', 0.0)]),
            'b': collections.OrderedDict([('num_over', 0.0)]),
        },
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
        output.state.client_work,
        {
            'a': collections.OrderedDict([('num_over', 6.0)]),
            'b': collections.OrderedDict([('num_over', 6.0)]),
        },
    )
    self.assertEqual(
        output.metrics['client_work'],
        {
            'a': collections.OrderedDict(
                eval=collections.OrderedDict(
                    current_round_metrics=collections.OrderedDict(num_over=6.0),
                    total_rounds_metrics=collections.OrderedDict(num_over=6.0),
                )
            ),
            'b': collections.OrderedDict(
                eval=collections.OrderedDict(
                    current_round_metrics=collections.OrderedDict(num_over=6.0),
                    total_rounds_metrics=collections.OrderedDict(num_over=6.0),
                )
            ),
        },
    )

  def test_invalid_metrics_aggregation_process_raises(self):
    test_model = TestModel()
    metrics_aggregator = aggregator.sum_then_finalize(
        test_model.metric_finalizers()
    )
    with self.assertRaisesRegex(TypeError, 'AggregationProcess'):
      fed_multi_model_eval.build_fed_multi_model_eval(
          (TestModel, TestModel),
          metrics_aggregation_processes=(
              metrics_aggregator,
              metrics_aggregator,
          ),
      )

  def test_invalid_model_distributor_raises(self):
    model_distributor = tensorflow_computation.tf_computation(lambda x: x)
    with self.assertRaisesRegex(TypeError, 'DistributionProcess'):
      fed_multi_model_eval.build_fed_multi_model_eval(
          (TestModel, TestModel), model_distributor=model_distributor
      )

  def test_construction_calls_model_fn(self):
    # Assert that the process building does not call `model_fn` too many times.
    # `model_fn` can potentially be expensive (loading weights, processing, etc
    # ).
    mock_model_fn = mock.Mock(side_effect=TestModel)
    fed_multi_model_eval.build_fed_multi_model_eval(
        (mock_model_fn, mock_model_fn)
    )
    self.assertEqual(mock_model_fn.call_count, 3 * 2)


class FunctionalFedMultiModelEvalProcessTest(tf.test.TestCase):

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
    with self.assertRaisesRegex(TypeError, 'FunctionalModel'):
      fed_multi_model_eval.build_fed_multi_model_eval(model_fns=(0,))

  @tensorflow_test_utils.skip_test_for_gpu
  def test_functional_evaluation_matches_non_functional(self):
    datasets = self.create_test_datasets()
    batch_type = tensorflow_types.to_type(datasets[0].element_spec)
    loss_fn = tf.keras.losses.MeanSquaredError
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

    eval_process = fed_multi_model_eval.build_fed_multi_model_eval((
        tff_model_fn,
    ))
    eval_state = eval_process.initialize()
    eval_output = eval_process.next(eval_state, datasets)

    # Defining artifacts using `tff.learning.models.FunctionalModel`
    def build_metrics_fn():
      return collections.OrderedDict(
          loss=tf.keras.metrics.MeanSquaredError(),
          num_examples=counters.NumExamplesCounter(),
          num_batches=counters.NumBatchesCounter(),
      )

    functional_model = functional.functional_model_from_keras(
        keras_model=keras_model_fn,
        loss_fn=loss_fn(),
        input_spec=batch_type,
        metrics_constructor=build_metrics_fn,
    )
    functional_eval_process = fed_multi_model_eval.build_fed_multi_model_eval((
        functional_model,
    ))
    functional_eval_state = functional_eval_process.initialize()
    functional_eval_output = functional_eval_process.next(
        functional_eval_state, datasets
    )
    self.assertDictEqual(eval_output.metrics, functional_eval_output.metrics)


class FedMultiModelEvalProcessFromEvalProcessesTest(tf.test.TestCase):

  def test_multi_model_eval_process_type_properties(
      self,
  ):
    eval_processes = [fed_eval.build_fed_eval(TestModel) for _ in range(2)]
    eval_process = (
        fed_multi_model_eval.build_fed_multi_model_eval_from_processes(
            eval_processes
        )
    )
    self.assertIsInstance(eval_process, learning_process.LearningProcess)

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

    expected_model_weights_type = {
        'a': model_weights_type,
        'b': model_weights_type,
    }
    expected_state_type = federated_language.FederatedType(
        composers.LearningAlgorithmState(
            global_model_weights=expected_model_weights_type,
            distributor={'a': (), 'b': ()},
            client_work={
                'a': metics_aggregator_state_type,
                'b': metics_aggregator_state_type,
            },
            aggregator={
                'a': collections.OrderedDict(
                    value_sum_process=(), weight_sum_process=()
                ),
                'b': collections.OrderedDict(
                    value_sum_process=(), weight_sum_process=()
                ),
            },
            finalizer={'a': (), 'b': ()},
        ),
        federated_language.SERVER,
    )
    expected_metrics_type = federated_language.FederatedType(
        collections.OrderedDict(
            distributor={'a': (), 'b': ()},
            client_work={
                'a': collections.OrderedDict(
                    eval=collections.OrderedDict(
                        current_round_metrics=finalized_metrics_type,
                        total_rounds_metrics=finalized_metrics_type,
                    )
                ),
                'b': collections.OrderedDict(
                    eval=collections.OrderedDict(
                        current_round_metrics=finalized_metrics_type,
                        total_rounds_metrics=finalized_metrics_type,
                    )
                ),
            },
            aggregator={
                'a': collections.OrderedDict(mean_value=(), mean_weight=()),
                'b': collections.OrderedDict(mean_value=(), mean_weight=()),
            },
            finalizer={'a': (), 'b': ()},
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
                parameter=expected_state_type.member,
                result=expected_model_weights_type,
            )
        )
    )
    self.assertTrue(
        eval_process.set_model_weights.type_signature.is_equivalent_to(
            FunctionType(
                parameter=StructType([
                    ('state', expected_state_type.member),
                    ('model_weights', expected_model_weights_type),
                ]),
                result=expected_state_type.member,
            )
        )
    )

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_multi_model_eval_process_execution(self):
    eval_process = (
        fed_multi_model_eval.build_fed_multi_model_eval_from_processes(
            [fed_eval.build_fed_eval(TestModel) for _ in range(2)]
        )
    )

    # Update the state with the model weights to be evaluated, and verify that
    # the `get_model_weights` method returns the same model weights.
    state = eval_process.initialize()
    model_weights = {
        'a': model_weights_lib.ModelWeights(trainable=[5.0], non_trainable=[]),
        'b': model_weights_lib.ModelWeights(trainable=[10.0], non_trainable=[]),
    }
    new_state = eval_process.set_model_weights(state, model_weights)
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
    self.assertSequenceEqual(list(output.state.client_work.keys()), ['a', 'b'])
    self.assertSequenceEqual(
        list(output.metrics['client_work'].keys()), ['a', 'b']
    )
    for (
        client_work_state_per_model,
        client_work_metrics_per_model,
        expected_num_over,
    ) in zip(
        output.state.client_work.values(),
        output.metrics['client_work'].values(),
        [9.0, 5.0],
    ):
      _, accumulated_unfinalized_metrics = client_work_state_per_model
      self.assertEqual(
          accumulated_unfinalized_metrics,
          collections.OrderedDict([('num_over', expected_num_over)]),
      )
      self.assertEqual(
          client_work_metrics_per_model,
          collections.OrderedDict(
              eval=collections.OrderedDict(
                  current_round_metrics=collections.OrderedDict(
                      num_over=expected_num_over
                  ),
                  total_rounds_metrics=collections.OrderedDict(
                      num_over=expected_num_over
                  ),
              )
          ),
      )

  def test_multi_model_eval_process_raises_on_less_than_two_eval_processes(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError, 'At least two evaluation processes must be provided'
    ):
      fed_multi_model_eval.build_fed_multi_model_eval_from_processes(
          [fed_eval.build_fed_eval(TestModel)]
      )

  def test_multi_model_eval_process_raises_on_inconsistent_dataset_type(self):
    eval_processes = [
        fed_eval.build_fed_eval(TestModel),
        fed_eval.build_fed_eval(model_examples.LinearRegression),
    ]
    with self.assertRaisesRegex(ValueError, 'same dataset type'):
      fed_multi_model_eval.build_fed_multi_model_eval_from_processes(
          eval_processes
      )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  absltest.main()
