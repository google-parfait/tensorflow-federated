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
from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory_utils
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import dataset_reduce
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning.algorithms import fed_avg
from tensorflow_federated.python.learning.algorithms import mime
from tensorflow_federated.python.learning.metrics import aggregator as metrics_aggregator
from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import test_models
from tensorflow_federated.python.learning.optimizers import adagrad
from tensorflow_federated.python.learning.optimizers import adam
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import rmsprop
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.optimizers import yogi
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.tensorflow_libs import tensorflow_test_utils


class MimeLiteClientWorkComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_type_properties(self, weighting):
    model_fn = model_examples.LinearRegression
    optimizer = sgdm.build_sgdm(learning_rate=0.1, momentum=0.9)
    client_work_process = mime._build_mime_lite_client_work(
        model_fn, optimizer, weighting
    )
    self.assertIsInstance(client_work_process, client_works.ClientWorkProcess)

    mw_type = model_weights.ModelWeights(
        trainable=computation_types.to_type([(np.float32, (2, 1)), np.float32]),
        non_trainable=computation_types.to_type([np.float32]),
    )
    expected_param_model_weights_type = computation_types.FederatedType(
        mw_type, placements.CLIENTS
    )
    element_type = computation_types.tensorflow_to_type(model_fn().input_spec)
    expected_param_data_type = computation_types.FederatedType(
        computation_types.SequenceType(element_type), placements.CLIENTS
    )
    expected_result_type = computation_types.FederatedType(
        client_works.ClientResult(
            update=mw_type.trainable,
            update_weight=computation_types.TensorType(np.float32),
        ),
        placements.CLIENTS,
    )
    expected_optimizer_state_type = computation_types.StructWithPythonType(
        collections.OrderedDict(
            learning_rate=np.float32,
            momentum=np.float32,
            accumulator=mw_type.trainable,
        ),
        collections.OrderedDict,
    )
    expected_aggregator_type = computation_types.to_type(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=())
    )
    expected_state_type = computation_types.FederatedType(
        (expected_optimizer_state_type, expected_aggregator_type),
        placements.SERVER,
    )
    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(
            train=collections.OrderedDict(
                loss=np.float32, num_examples=np.int32
            )
        ),
        placements.SERVER,
    )

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type
    )
    expected_initialize_type.check_equivalent_to(
        client_work_process.initialize.type_signature
    )

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_model_weights_type,
            client_data=expected_param_data_type,
        ),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type,
            expected_result_type,
            expected_measurements_type,
        ),
    )
    expected_next_type.check_equivalent_to(
        client_work_process.next.type_signature
    )

  def test_created_model_raises(self):
    with self.assertRaises(TypeError):
      mime._build_mime_lite_client_work(
          model_examples.LinearRegression(),
          sgdm.build_sgdm(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      )

  def test_keras_optimizer_raises(self):
    with self.assertRaises(TypeError):
      mime._build_mime_lite_client_work(
          model_examples.LinearRegression,
          lambda: tf.keras.optimizers.SGD(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      )

  def test_unweighted_full_gradient_aggregator_raises(self):
    with self.assertRaises(TypeError):
      mime._build_mime_lite_client_work(
          model_examples.LinearRegression(),
          sgdm.build_sgdm(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
          full_gradient_aggregator=mean.UnweightedMeanFactory(),
      )


def _create_model():
  return model_examples.LinearRegression(feature_dim=2)


def _initial_weights():
  return model_weights.ModelWeights(
      trainable=[tf.zeros((2, 1)), tf.constant(0.0)], non_trainable=[0.0]
  )


def _create_dataset():
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
          y=[[0.0], [0.0], [1.0], [1.0]],
      )
  )
  # Repeat the dataset 2 times with batches of 3 examples, producing 3
  # minibatches (the last one with only 2 examples).  Note that `batch` is
  # required for this dataset to be useable, as it adds the batch dimension
  # which is expected by the model.
  return dataset.repeat(2).batch(3)


class MimeLiteClientWorkExecutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('non-simulation', False), ('simulation', True)
  )
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn,
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    process = mime._build_mime_lite_client_work(
        model_fn=_create_model,
        optimizer=sgdm.build_sgdm(learning_rate=0.1, momentum=0.9),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        use_experimental_simulation_loop=simulation,
    )
    client_data = [_create_dataset()]
    client_model_weights = [_initial_weights()]
    process.next(process.initialize(), client_model_weights, client_data)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  @parameterized.named_parameters(
      ('adagrad', adagrad.build_adagrad(0.1)),
      ('adam', adam.build_adam(0.1)),
      ('rmsprop', rmsprop.build_rmsprop(0.1)),
      ('sgd', sgdm.build_sgdm(0.1)),
      ('sgdm', sgdm.build_sgdm(0.1, momentum=0.9)),
      ('yogi', yogi.build_yogi(0.1)),
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_execution_with_optimizer(self, optimizer):
    process = mime._build_mime_lite_client_work(
        _create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    client_data = [_create_dataset()]
    client_model_weights = [_initial_weights()]
    state = process.initialize()
    output = process.next(state, client_model_weights, client_data)
    self.assertEqual(8, output.measurements['train']['num_examples'])

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_custom_metrics_aggregator(self):

    def sum_then_finalize_then_times_two(metric_finalizers):
      @federated_computation.federated_computation
      def aggregation_computation(client_local_unfinalized_metrics):
        unfinalized_metrics_sum = intrinsics.federated_sum(
            client_local_unfinalized_metrics
        )

        @tensorflow_computation.tf_computation
        def finalizer_computation(unfinalized_metrics):
          finalized_metrics = collections.OrderedDict()
          for metric_name, metric_finalizer in metric_finalizers.items():
            finalized_metrics[metric_name] = (
                metric_finalizer(unfinalized_metrics[metric_name]) * 2
            )
          return finalized_metrics

        return intrinsics.federated_map(
            finalizer_computation, unfinalized_metrics_sum
        )

      return aggregation_computation

    process = mime._build_mime_lite_client_work(
        model_fn=_create_model,
        optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        metrics_aggregator=sum_then_finalize_then_times_two,
    )
    client_model_weights = [_initial_weights()]
    client_data = [_create_dataset()]
    output = process.next(
        process.initialize(), client_model_weights, client_data
    )
    # Train metrics should be multiplied by two by the custom aggregator.
    self.assertEqual(output.measurements['train']['num_examples'], 16)


class MimeLiteFunctionalClientWorkExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('non-simulation', False), ('simulation', True)
  )
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn,
  )
  @tensorflow_test_utils.skip_test_for_gpu
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    model = test_models.build_functional_linear_regression()
    process = mime._build_mime_lite_functional_client_work(
        model=model,
        optimizer=sgdm.build_sgdm(learning_rate=0.1, momentum=0.9),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        use_experimental_simulation_loop=simulation,
    )
    client_data = [_create_dataset().map(lambda d: (d['x'], d['y']))]
    client_model_weights = [model.initial_weights]
    process.next(process.initialize(), client_model_weights, client_data)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  @parameterized.named_parameters(
      ('adagrad', adagrad.build_adagrad(0.1)),
      ('adam', adam.build_adam(0.1)),
      ('rmsprop', rmsprop.build_rmsprop(0.1)),
      ('sgd', sgdm.build_sgdm(0.1)),
      ('sgdm', sgdm.build_sgdm(0.1, momentum=0.9)),
      ('yogi', yogi.build_yogi(0.1)),
  )
  @tensorflow_test_utils.skip_test_for_gpu
  def test_execution_with_optimizer(self, optimizer):
    model = test_models.build_functional_linear_regression()
    process = mime._build_mime_lite_functional_client_work(
        model=model,
        optimizer=optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    client_data = [_create_dataset().map(lambda d: (d['x'], d['y']))]
    client_model_weights = [model.initial_weights]
    state = process.initialize()
    output = process.next(state, client_model_weights, client_data)
    self.assertEqual(8, output.measurements['train']['num_examples'])

  @tensorflow_test_utils.skip_test_for_gpu
  def test_custom_metrics_aggregator(self):
    def sum_then_finalize_then_times_two(
        metric_finalizers, local_unfinalized_metrics_type
    ):

      @federated_computation.federated_computation(
          computation_types.FederatedType(
              local_unfinalized_metrics_type, placements.CLIENTS
          )
      )
      def aggregation_computation(client_local_unfinalized_metrics):
        unfinalized_metrics_sum = intrinsics.federated_sum(
            client_local_unfinalized_metrics
        )

        @tensorflow_computation.tf_computation(local_unfinalized_metrics_type)
        def finalizer_computation(unfinalized_metrics):
          finalized_metrics = collections.OrderedDict(
              (metric_name, finalized_value * 2)
              for metric_name, finalized_value in metric_finalizers(
                  unfinalized_metrics
              ).items()
          )
          return finalized_metrics

        return intrinsics.federated_map(
            finalizer_computation, unfinalized_metrics_sum
        )

      return aggregation_computation

    model = test_models.build_functional_linear_regression()
    process = mime._build_mime_lite_functional_client_work(
        model=model,
        optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        metrics_aggregator=sum_then_finalize_then_times_two,
    )
    client_model_weights = [model.initial_weights]
    client_data = [_create_dataset().map(lambda d: (d['x'], d['y']))]
    output = process.next(
        process.initialize(), client_model_weights, client_data
    )
    # Train metrics should be multiplied by two by the custom aggregator.
    self.assertEqual(output.measurements['train']['num_examples'], 16)


class MimeLiteTest(tf.test.TestCase, parameterized.TestCase):
  """Tests construction of the Mime Lite training process."""

  def test_construction_calls_model_fn(self):
    # Assert that the process building does not call `model_fn` too many times.
    # `model_fn` can potentially be expensive (loading weights, processing, etc
    # ).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    mime.build_weighted_mime_lite(
        model_fn=mock_model_fn,
        base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
    )
    self.assertEqual(mock_model_fn.call_count, 3)

  @parameterized.named_parameters(
      ('non-simulation_tff_optimizer', False),
      ('simulation_tff_optimizer', True),
  )
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn,
  )
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    mime.build_weighted_mime_lite(
        model_fn=model_examples.LinearRegression,
        base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
        use_experimental_simulation_loop=simulation,
    )
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  @mock.patch.object(mime, 'build_weighted_mime_lite')
  def test_build_weighted_mime_lite_called_by_unweighted_mime_lite(
      self, mock_mime_lite
  ):
    mime.build_unweighted_mime_lite(
        model_fn=model_examples.LinearRegression,
        base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
    )
    self.assertEqual(mock_mime_lite.call_count, 1)

  @mock.patch.object(mime, 'build_weighted_mime_lite')
  @mock.patch.object(factory_utils, 'as_weighted_aggregator')
  def test_aggregation_wrapper_called_by_unweighted(self, _, mock_as_weighted):
    mime.build_unweighted_mime_lite(
        model_fn=model_examples.LinearRegression,
        base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
    )
    self.assertEqual(mock_as_weighted.call_count, 1)

  def test_raises_on_non_callable_model_fn(self):
    with self.assertRaises(TypeError):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression(),
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
      )

  def test_raises_on_invalid_client_weighting(self):
    with self.assertRaises(TypeError):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          client_weighting='uniform',
      )

  def test_raises_on_invalid_distributor(self):
    model_weights_type = model_weights.weights_type_from_model(
        model_examples.LinearRegression
    )
    distributor = distributors.build_broadcast_process(model_weights_type)
    invalid_distributor = iterative_process.IterativeProcess(
        distributor.initialize, distributor.next
    )
    with self.assertRaises(TypeError):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          model_distributor=invalid_distributor,
      )

  def test_weighted_mime_lite_raises_on_unweighted_aggregator(self):
    aggregator = model_update_aggregator.robust_aggregator(weighted=False)
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          model_aggregator=aggregator,
      )
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      mime.build_weighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          full_gradient_aggregator=aggregator,
      )

  def test_unweighted_mime_lite_raises_on_weighted_aggregator(self):
    aggregator = model_update_aggregator.robust_aggregator(weighted=True)
    with self.assertRaisesRegex(TypeError, 'UnweightedAggregationFactory'):
      mime.build_unweighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          model_aggregator=aggregator,
      )
    with self.assertRaisesRegex(TypeError, 'UnweightedAggregationFactory'):
      mime.build_unweighted_mime_lite(
          model_fn=model_examples.LinearRegression,
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          full_gradient_aggregator=aggregator,
      )

  def test_weighted_mime_lite_with_only_secure_aggregation(self):
    aggregator = model_update_aggregator.secure_aggregator(weighted=True)
    learning_process = mime.build_weighted_mime_lite(
        model_examples.LinearRegression,
        base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
        model_aggregator=aggregator,
        full_gradient_aggregator=aggregator,
        metrics_aggregator=metrics_aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
    )

  def test_unweighted_mime_lite_with_only_secure_aggregation(self):
    aggregator = model_update_aggregator.secure_aggregator(weighted=False)
    learning_process = mime.build_unweighted_mime_lite(
        model_examples.LinearRegression,
        base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
        model_aggregator=aggregator,
        full_gradient_aggregator=aggregator,
        metrics_aggregator=metrics_aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
    )

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_equivalent_to_vanilla_fed_avg(self):
    # Mime Lite with no-momentum SGD should reduce to FedAvg.
    mime_process = mime.build_weighted_mime_lite(
        model_fn=_create_model, base_optimizer=sgdm.build_sgdm(0.1)
    )
    fed_avg_process = fed_avg.build_weighted_fed_avg(
        model_fn=_create_model, client_optimizer_fn=sgdm.build_sgdm(0.1)
    )

    client_data = [_create_dataset()]
    mime_state = mime_process.initialize()
    fed_avg_state = fed_avg_process.initialize()

    for _ in range(3):
      mime_output = mime_process.next(mime_state, client_data)
      mime_state = mime_output.state
      mime_metrics = mime_output.metrics
      fed_avg_output = fed_avg_process.next(fed_avg_state, client_data)
      fed_avg_state = fed_avg_output.state
      fed_avg_metrics = fed_avg_output.metrics
      self.assertAllClose(
          tf.nest.flatten(mime_process.get_model_weights(mime_state)),
          tf.nest.flatten(fed_avg_process.get_model_weights(fed_avg_state)),
      )
      self.assertAllClose(
          mime_metrics['client_work']['train']['loss'],
          fed_avg_metrics['client_work']['train']['loss'],
      )
      self.assertAllClose(
          mime_metrics['client_work']['train']['num_examples'],
          fed_avg_metrics['client_work']['train']['num_examples'],
      )

  @parameterized.named_parameters(
      ('sgdm_sgd', sgdm.build_sgdm(0.1, 0.9), sgdm.build_sgdm(1.0)),
      ('sgdm_sgdm', sgdm.build_sgdm(0.1, 0.9), sgdm.build_sgdm(1.0, 0.9)),
      ('sgdm_adam', sgdm.build_sgdm(0.1, 0.9), adam.build_adam(1.0)),
      ('adagrad_sgdm', adagrad.build_adagrad(0.1), sgdm.build_sgdm(1.0, 0.9)),
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_execution_with_optimizers(self, base_optimizer, server_optimizer):
    learning_process = mime.build_weighted_mime_lite(
        model_fn=_create_model,
        base_optimizer=base_optimizer,
        server_optimizer=server_optimizer,
    )

    client_data = [_create_dataset()]
    state = learning_process.initialize()

    for _ in range(3):
      output = learning_process.next(state, client_data)
      state = output.state
      metrics = output.metrics
      self.assertEqual(8, metrics['client_work']['train']['num_examples'])


class ScheduledMimeLiteTest(tf.test.TestCase):
  """Tests the Mime Lite training process with learning rate scheduling."""

  def test_raises_on_non_callable_model_fn(self):
    with self.assertRaises(TypeError):
      mime.build_mime_lite_with_optimizer_schedule(
          model_fn=model_examples.LinearRegression(),
          learning_rate_fn=lambda x: tf.constant(0.1),
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
      )

  def test_raises_on_non_callable_learning_rate_fn(self):
    with self.assertRaises(TypeError):
      mime.build_mime_lite_with_optimizer_schedule(
          model_fn=model_examples.LinearRegression,
          learning_rate_fn=tf.constant(0.1),
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
      )

  def test_raises_on_invalid_client_weighting(self):
    with self.assertRaises(TypeError):
      mime.build_mime_lite_with_optimizer_schedule(
          model_fn=model_examples.LinearRegression,
          learning_rate_fn=lambda x: tf.constant(0.1),
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          client_weighting='uniform',
      )

  def test_raises_on_invalid_distributor(self):
    model_weights_type = model_weights.weights_type_from_model(
        model_examples.LinearRegression
    )
    distributor = distributors.build_broadcast_process(model_weights_type)
    invalid_distributor = iterative_process.IterativeProcess(
        distributor.initialize, distributor.next
    )
    with self.assertRaises(TypeError):
      mime.build_mime_lite_with_optimizer_schedule(
          model_fn=model_examples.LinearRegression,
          learning_rate_fn=lambda x: tf.constant(0.1),
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          model_distributor=invalid_distributor,
      )

  def test_raises_on_unweighted_aggregator(self):
    aggregator = model_update_aggregator.robust_aggregator(weighted=False)
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      mime.build_mime_lite_with_optimizer_schedule(
          model_fn=model_examples.LinearRegression,
          learning_rate_fn=lambda x: tf.constant(0.1),
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          model_aggregator=aggregator,
      )
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      mime.build_mime_lite_with_optimizer_schedule(
          model_fn=model_examples.LinearRegression,
          learning_rate_fn=lambda x: tf.constant(0.1),
          base_optimizer=sgdm.build_sgdm(learning_rate=0.01, momentum=0.9),
          full_gradient_aggregator=aggregator,
      )

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_construction_calls_learning_rate_fn(self):
    mock_learning_rate_fn = mock.Mock(
        side_effect=lambda x: tf.cast(x, tf.float32)
    )
    mime_process = mime.build_mime_lite_with_optimizer_schedule(
        model_fn=_create_model,
        learning_rate_fn=mock_learning_rate_fn,
        base_optimizer=sgdm.build_sgdm(0.1),
    )

    client_data = [_create_dataset()]
    mime_state = mime_process.initialize()
    self.assertEqual(
        mime_state.client_work[1][0][optimizer_base.LEARNING_RATE_KEY], 0.0
    )

    for x in range(2):
      mime_output = mime_process.next(mime_state, client_data)
      mime_state = mime_output.state
      self.assertEqual(
          mime_state.client_work[1][0][optimizer_base.LEARNING_RATE_KEY],
          x + 1.0,
      )

    self.assertEqual(mock_learning_rate_fn.call_count, 2)


class FunctionalMimeLiteTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('sgdm_sgd', sgdm.build_sgdm(0.1, 0.9), sgdm.build_sgdm(1.0)),
      ('sgdm_sgdm', sgdm.build_sgdm(0.1, 0.9), sgdm.build_sgdm(1.0, 0.9)),
      ('sgdm_adam', sgdm.build_sgdm(0.1, 0.9), adam.build_adam(1.0)),
      ('adagrad_sgdm', adagrad.build_adagrad(0.1), sgdm.build_sgdm(1.0, 0.9)),
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_functional_matches_variable_model(
      self, base_optimizer, server_optimizer
  ):
    dataset = _create_dataset()

    def create_keras_model():
      return tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape=(2,)),
          tf.keras.layers.Dense(
              units=1, kernel_initializer='zeros', bias_initializer='zeros'
          ),
      ])

    def create_functional_model():
      return functional.functional_model_from_keras(
          create_keras_model(),
          loss_fn=tf.keras.losses.MeanSquaredError(),
          input_spec=dataset.element_spec,
          metrics_constructor=collections.OrderedDict(
              loss=tf.keras.metrics.MeanSquaredError,
              num_examples=counters.NumExamplesCounter,
              num_batches=counters.NumBatchesCounter,
          ),
      )

    def create_tff_model():
      return keras_utils.from_keras_model(
          create_keras_model(),
          loss=tf.keras.losses.MeanSquaredError(),
          input_spec=dataset.element_spec,
      )

    functional_learning_process = mime.build_weighted_mime_lite(
        model_fn=create_functional_model(),
        base_optimizer=base_optimizer,
        server_optimizer=server_optimizer,
    )
    variable_learning_process = mime.build_weighted_mime_lite(
        model_fn=create_tff_model,
        base_optimizer=base_optimizer,
        server_optimizer=server_optimizer,
    )

    variable_state = variable_learning_process.initialize()
    functional_state = functional_learning_process.initialize()

    named_client_data = [dataset]
    for round_num in range(10):
      variable_output = variable_learning_process.next(
          variable_state, named_client_data
      )
      variable_state = variable_output.state
      functional_output = functional_learning_process.next(
          functional_state, named_client_data
      )
      functional_state = functional_output.state
      self.assertAllClose(
          variable_output.metrics,
          functional_output.metrics,
          msg=f'Round {round_num}',
      )
      self.assertAllClose(
          variable_state,
          functional_state,
          msg=f'Round {round_num}',
      )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
