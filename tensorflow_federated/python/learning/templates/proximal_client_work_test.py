# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import dataset_reduce
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import test_models
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import proximal_client_work


class ProximalClientWorkComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      (
          'tff_uniform',
          sgdm.build_sgdm(1.0),
          client_weight_lib.ClientWeighting.UNIFORM,
      ),
      (
          'tff_num_examples',
          sgdm.build_sgdm(1.0),
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
      (
          'keras_uniform',
          lambda: tf.keras.optimizers.SGD(1.0),
          client_weight_lib.ClientWeighting.UNIFORM,
      ),
      (
          'keras_num_examples',
          lambda: tf.keras.optimizers.SGD(1.0),
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
  )
  def test_type_properties(self, optimizer, weighting):
    model_fn = model_examples.LinearRegression
    client_work_process = proximal_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting, delta_l2_regularizer=0.1
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
    expected_state_type = computation_types.FederatedType((), placements.SERVER)
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

  def test_created_keras_optimizer_raises(self):
    with self.assertRaises(TypeError):
      proximal_client_work.build_model_delta_client_work(
          model_examples.LinearRegression,
          tf.keras.optimizers.SGD(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
          delta_l2_regularizer=0.1,
      )

  def test_created_model_raises(self):
    with self.assertRaises(TypeError):
      proximal_client_work.build_model_delta_client_work(
          model_examples.LinearRegression(),
          sgdm.build_sgdm(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
          delta_l2_regularizer=0.1,
      )

  def test_negative_proximal_strength_raises(self):
    with self.assertRaises(ValueError):
      proximal_client_work.build_model_delta_client_work(
          model_examples.LinearRegression,
          sgdm.build_sgdm(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
          delta_l2_regularizer=-1.0,
      )


def create_test_dataset() -> tf.data.Dataset:
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
          y=[[0.0], [0.0], [1.0], [1.0]],
      )
  )
  # Repeat the dataset 2 times with batches of 3 examples,
  # producing 3 minibatches (the last one with only 2 examples).
  # Note that `batch` is required for this dataset to be useable,
  # as it adds the batch dimension which is expected by the model.
  return dataset.repeat(2).batch(3)


def create_test_initial_weights() -> model_weights.ModelWeights:
  return model_weights.ModelWeights(
      trainable=[tf.zeros((2, 1)), tf.constant(0.0)], non_trainable=[0.0]
  )


def create_model():
  return model_examples.LinearRegression(feature_dim=2)


class ProximalClientWorkExecutionTest(tf.test.TestCase, parameterized.TestCase):
  """Tests of the client work of FedProx using a common model and data."""

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_keras_tff_client_work_equal(self, weighting):
    dataset = create_test_dataset()
    client_update_keras = (
        proximal_client_work.build_model_delta_update_with_keras_optimizer(
            model_fn=create_model, weighting=weighting, delta_l2_regularizer=0.1
        )
    )
    client_update_tff = (
        proximal_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model, weighting=weighting, delta_l2_regularizer=0.1
        )
    )
    keras_result = client_update_keras(
        tf.keras.optimizers.SGD(learning_rate=0.1),
        create_test_initial_weights(),
        dataset,
    )
    tff_result = client_update_tff(
        sgdm.build_sgdm(learning_rate=0.1),
        create_test_initial_weights(),
        dataset,
    )
    self.assertAllClose(keras_result[0].update, tff_result[0].update)
    self.assertEqual(keras_result[0].update_weight, tff_result[0].update_weight)
    self.assertAllClose(keras_result[1], tff_result[1])

  @parameterized.named_parameters(
      (
          'non-simulation_noclip_uniform',
          False,
          {},
          0.1,
          client_weight_lib.ClientWeighting.UNIFORM,
      ),
      (
          'non-simulation_noclip_num_examples',
          False,
          {},
          0.1,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
      (
          'simulation_noclip_uniform',
          True,
          {},
          0.1,
          client_weight_lib.ClientWeighting.UNIFORM,
      ),
      (
          'simulation_noclip_num_examples',
          True,
          {},
          0.1,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
      (
          'non-simulation_clipnorm',
          False,
          {'clipnorm': 0.2},
          0.05,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
      (
          'non-simulation_clipvalue',
          False,
          {'clipvalue': 0.1},
          0.02,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
  )
  def test_client_tf(
      self, simulation, optimizer_kwargs, expected_norm, weighting
  ):
    client_tf = (
        proximal_client_work.build_model_delta_update_with_keras_optimizer(
            model_fn=create_model,
            weighting=weighting,
            delta_l2_regularizer=0.1,
            use_experimental_simulation_loop=simulation,
        )
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs)
    dataset = create_test_dataset()
    client_result, model_output = self.evaluate(
        client_tf(optimizer, create_test_initial_weights(), dataset)
    )
    # Both trainable parameters should have been updated, and we don't return
    # the non-trainable variable.
    for trainable_param in client_result.update:
      self.assertAllGreater(np.linalg.norm(trainable_param), expected_norm)
    if weighting == client_weight_lib.ClientWeighting.UNIFORM:
      self.assertEqual(client_result.update_weight, 1.0)
    else:
      self.assertEqual(client_result.update_weight, 8.0)
    self.assertDictContainsSubset({'num_examples': 8}, model_output)
    self.assertBetween(model_output['loss'][0], np.finfo(np.float32).eps, 10.0)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    client_tf = (
        proximal_client_work.build_model_delta_update_with_keras_optimizer(
            model_fn=create_model,
            weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            delta_l2_regularizer=0.1,
        )
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = create_test_dataset()
    init_weights = create_test_initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(optimizer, init_weights, dataset)
    self.assertEqual(self.evaluate(client_outputs[0].update_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs[0].update), [[[0.0], [0.0]], 0.0]
    )

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

    process = proximal_client_work.build_model_delta_client_work(
        model_fn=create_model,
        optimizer=sgdm.build_sgdm(1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        delta_l2_regularizer=0.1,
        metrics_aggregator=sum_then_finalize_then_times_two,
    )
    client_model_weights = [create_test_initial_weights()]
    client_data = [create_test_dataset()]
    output = process.next(
        process.initialize(), client_model_weights, client_data
    )
    # Train metrics should be multiplied by two by the custom aggregator.
    self.assertEqual(output.measurements['train']['num_examples'], 16)

  @parameterized.named_parameters(
      ('non-simulation', False), ('simulation', True)
  )
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn,
  )
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    client_tf = (
        proximal_client_work.build_model_delta_update_with_keras_optimizer(
            model_fn=create_model,
            weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            delta_l2_regularizer=0.1,
            use_experimental_simulation_loop=simulation,
        )
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = create_test_dataset()
    client_tf(optimizer, create_test_initial_weights(), dataset)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  @parameterized.named_parameters(
      ('tff_optimizer', sgdm.build_sgdm(1.0)),
      ('keras_optimizer', lambda: tf.keras.optimizers.SGD(1.0)),
  )
  def test_delta_regularizer_yields_smaller_model_delta(self, optimizer):
    small_delta_process = proximal_client_work.build_model_delta_client_work(
        create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        delta_l2_regularizer=0.01,
    )
    large_delta_process = proximal_client_work.build_model_delta_client_work(
        create_model,
        sgdm.build_sgdm(1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        delta_l2_regularizer=1.0,
    )
    client_data = [create_test_dataset()]
    client_model_weights = [create_test_initial_weights()]

    small_delta_output = small_delta_process.next(
        small_delta_process.initialize(), client_model_weights, client_data
    )
    large_delta_output = large_delta_process.next(
        large_delta_process.initialize(), client_model_weights, client_data
    )

    small_delta_update_norm = tf.linalg.global_norm(
        tf.nest.flatten(small_delta_output.result[0].update)
    )
    large_delta_update_norm = tf.linalg.global_norm(
        tf.nest.flatten(large_delta_output.result[0].update)
    )
    self.assertGreater(small_delta_update_norm, large_delta_update_norm)

    self.assertEqual(
        small_delta_output.measurements['train']['num_examples'],
        large_delta_output.measurements['train']['num_examples'],
    )

  @parameterized.named_parameters(
      ('tff_simple', sgdm.build_sgdm(1.0)),
      ('tff_momentum', sgdm.build_sgdm(1.0, momentum=0.9)),
      ('keras_simple', lambda: tf.keras.optimizers.SGD(1.0)),
      ('keras_momentum', lambda: tf.keras.optimizers.SGD(1.0, momentum=0.9)),
  )
  def test_execution_with_optimizer(self, optimizer):
    client_work_process = proximal_client_work.build_model_delta_client_work(
        create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        delta_l2_regularizer=0.1,
    )
    client_data = [create_test_dataset()]
    client_model_weights = [create_test_initial_weights()]

    state = client_work_process.initialize()
    output = client_work_process.next(state, client_model_weights, client_data)

    self.assertCountEqual(output.measurements.keys(), ['train'])


def create_functional_model() -> functional.FunctionalModel:
  return test_models.build_functional_linear_regression(feature_dim=2)


class FunctionalProximalClientWorkExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):
  """Tests of the client work of FedProx using a functional model and data."""

  @parameterized.named_parameters(
      ('uniform_weighting', 0.1, client_weight_lib.ClientWeighting.UNIFORM),
      (
          'uniform_weighting_new_prox',
          0.2,
          client_weight_lib.ClientWeighting.UNIFORM,
      ),
      (
          'num_examples_weighting',
          0.1,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
      (
          'num_examples_weighting_new_prox',
          0.2,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
  )
  def test_client_tf(self, expected_norm, weighting):
    model = create_functional_model()
    client_tf = proximal_client_work._build_functional_model_delta_update(
        model=model, weighting=weighting, delta_l2_regularizer=expected_norm
    )
    optimizer = sgdm.build_sgdm(learning_rate=0.1)
    dataset = create_test_dataset()
    client_result, model_output = self.evaluate(
        client_tf(optimizer, model.initial_weights, dataset)
    )
    # Both trainable parameters should have been updated, and we don't return
    # the non-trainable variable.
    for trainable_param in client_result.update:
      self.assertAllGreater(np.linalg.norm(trainable_param), 0.1)
    if weighting == client_weight_lib.ClientWeighting.UNIFORM:
      self.assertEqual(client_result.update_weight, 1.0)
    else:
      self.assertEqual(client_result.update_weight, 8.0)
    self.assertDictContainsSubset({'num_examples': 8}, model_output)
    self.assertBetween(model_output['loss'], np.finfo(np.float32).eps, 10.0)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    model = create_functional_model()
    client_tf = proximal_client_work._build_functional_model_delta_update(
        model=model,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        delta_l2_regularizer=0.1,
    )
    optimizer = sgdm.build_sgdm(learning_rate=0.1)
    dataset = create_test_dataset()
    trainable_weights, non_trainable_weights = model.initial_weights
    # Replace the trainable bias with a non finitie value.
    init_weights = (
        (trainable_weights[0], tf.constant(bad_value)),
        non_trainable_weights,
    )
    client_outputs = client_tf(optimizer, init_weights, dataset)
    self.assertEqual(self.evaluate(client_outputs[0].update_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs[0].update), [[[0.0], [0.0]], 0.0]
    )

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
          return tf.nest.map_structure(
              lambda x: x * 2, metric_finalizers(unfinalized_metrics)
          )

        return intrinsics.federated_map(
            finalizer_computation, unfinalized_metrics_sum
        )

      return aggregation_computation

    model = create_functional_model()
    process = proximal_client_work.build_functional_model_delta_client_work(
        model=model,
        optimizer=sgdm.build_sgdm(1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        delta_l2_regularizer=0.1,
        metrics_aggregator=sum_then_finalize_then_times_two,
    )
    client_model_weights = [model.initial_weights]
    client_data = [create_test_dataset().map(lambda d: (d['x'], d['y']))]
    output = process.next(
        process.initialize(), client_model_weights, client_data
    )
    # Train metrics should be multiplied by two by the custom aggregator.
    self.assertEqual(output.measurements['train']['num_examples'], 16)

  def test_delta_regularizer_yields_smaller_model_delta(self):
    model = create_functional_model()
    small_delta_process = (
        proximal_client_work.build_functional_model_delta_client_work(
            model=model,
            optimizer=sgdm.build_sgdm(1.0),
            client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            delta_l2_regularizer=0.01,
        )
    )
    large_delta_process = (
        proximal_client_work.build_functional_model_delta_client_work(
            model=model,
            optimizer=sgdm.build_sgdm(1.0),
            client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            delta_l2_regularizer=1.0,
        )
    )
    client_data = [create_test_dataset().map(lambda d: (d['x'], d['y']))]
    client_model_weights = [model.initial_weights]

    small_delta_output = small_delta_process.next(
        small_delta_process.initialize(), client_model_weights, client_data
    )
    large_delta_output = large_delta_process.next(
        large_delta_process.initialize(), client_model_weights, client_data
    )

    small_delta_update_norm = tf.linalg.global_norm(
        tf.nest.flatten(small_delta_output.result[0].update)
    )
    large_delta_update_norm = tf.linalg.global_norm(
        tf.nest.flatten(large_delta_output.result[0].update)
    )
    self.assertGreater(small_delta_update_norm, large_delta_update_norm)
    self.assertEqual(
        small_delta_output.measurements['train']['num_examples'],
        large_delta_output.measurements['train']['num_examples'],
    )

  @parameterized.named_parameters(
      ('simple', sgdm.build_sgdm(1.0)),
      ('momentum', sgdm.build_sgdm(1.0, momentum=0.9)),
  )
  def test_execution_with_optimizer(self, optimizer):
    model = create_functional_model()
    client_work_process = (
        proximal_client_work.build_functional_model_delta_client_work(
            model=model,
            optimizer=optimizer,
            client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            delta_l2_regularizer=0.1,
        )
    )
    client_data = [create_test_dataset().map(lambda d: (d['x'], d['y']))]
    client_model_weights = [model.initial_weights]
    state = client_work_process.initialize()
    output = client_work_process.next(state, client_model_weights, client_data)
    self.assertCountEqual(output.measurements.keys(), ['train'])


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
