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
import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import model_delta_client_work


class ModelDeltaClientWorkComputationTest(tf.test.TestCase,
                                          parameterized.TestCase):

  @parameterized.named_parameters(
      ('tff_uniform', sgdm.build_sgdm(1.0),
       client_weight_lib.ClientWeighting.UNIFORM),
      ('tff_num_examples', sgdm.build_sgdm(1.0),
       client_weight_lib.ClientWeighting.NUM_EXAMPLES),
      ('keras_uniform', lambda: tf.keras.optimizers.SGD(1.0),
       client_weight_lib.ClientWeighting.UNIFORM),
      ('keras_num_examples', lambda: tf.keras.optimizers.SGD(1.0),
       client_weight_lib.ClientWeighting.NUM_EXAMPLES))
  def test_type_properties(self, optimizer, weighting):
    model_fn = model_examples.LinearRegression
    client_work_process = model_delta_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting)
    self.assertIsInstance(client_work_process, client_works.ClientWorkProcess)

    mw_type = model_utils.ModelWeights(
        trainable=computation_types.to_type([(tf.float32, (2, 1)), tf.float32]),
        non_trainable=computation_types.to_type([tf.float32]))
    expected_param_model_weights_type = computation_types.at_clients(mw_type)
    expected_param_data_type = computation_types.at_clients(
        computation_types.SequenceType(
            computation_types.to_type(model_fn().input_spec)))
    expected_result_type = computation_types.at_clients(
        client_works.ClientResult(
            update=mw_type.trainable,
            update_weight=computation_types.TensorType(tf.float32)))
    expected_state_type = computation_types.at_server(())
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(
            train=collections.OrderedDict(
                loss=tf.float32, num_examples=tf.int32)))

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    expected_initialize_type.check_equivalent_to(
        client_work_process.initialize.type_signature)

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_model_weights_type,
            client_data=expected_param_data_type),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, expected_result_type,
            expected_measurements_type))
    expected_next_type.check_equivalent_to(
        client_work_process.next.type_signature)

  def test_created_keras_optimizer_raises(self):
    with self.assertRaises(TypeError):
      model_delta_client_work.build_model_delta_client_work(
          model_examples.LinearRegression,
          tf.keras.optimizers.SGD(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)

  def test_created_model_raises(self):
    with self.assertRaises(TypeError):
      model_delta_client_work.build_model_delta_client_work(
          model_examples.LinearRegression(),
          sgdm.build_sgdm(1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)


def create_test_dataset() -> tf.data.Dataset:
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
          y=[[0.0], [0.0], [1.0], [1.0]]))
  # Repeat the dataset 2 times with batches of 3 examples,
  # producing 3 minibatches (the last one with only 2 examples).
  # Note that `batch` is required for this dataset to be useable,
  # as it adds the batch dimension which is expected by the model.
  return dataset.repeat(2).batch(3)


def create_test_initial_weights() -> model_utils.ModelWeights:
  return model_utils.ModelWeights(
      trainable=[tf.zeros((2, 1)), tf.constant(0.0)], non_trainable=[0.0])


def create_model():
  return model_examples.LinearRegression(feature_dim=2)


class ModelDeltaClientWorkExecutionTest(tf.test.TestCase,
                                        parameterized.TestCase):
  """Tests of the client work of FedAvg using a common model and data."""

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES))
  def test_keras_tff_client_work_equal(self, weighting):
    dataset = create_test_dataset()
    client_update_keras = model_delta_client_work.build_model_delta_update_with_keras_optimizer(
        model_fn=create_model, weighting=weighting)
    client_update_tff = model_delta_client_work.build_model_delta_update_with_tff_optimizer(
        model_fn=create_model, weighting=weighting)
    keras_result = client_update_keras(
        tf.keras.optimizers.SGD(learning_rate=0.1),
        create_test_initial_weights(), dataset)
    tff_result = client_update_tff(
        sgdm.build_sgdm(learning_rate=0.1), create_test_initial_weights(),
        dataset)
    self.assertAllClose(keras_result[0].update, tff_result[0].update)
    self.assertEqual(keras_result[0].update_weight, tff_result[0].update_weight)
    self.assertAllClose(keras_result[1], tff_result[1])

  @parameterized.named_parameters(
      ('non-simulation_noclip_uniform', False, {}, 0.1,
       client_weight_lib.ClientWeighting.UNIFORM),
      ('non-simulation_noclip_num_examples', False, {}, 0.1,
       client_weight_lib.ClientWeighting.NUM_EXAMPLES),
      ('simulation_noclip_uniform', True, {}, 0.1,
       client_weight_lib.ClientWeighting.UNIFORM),
      ('simulation_noclip_num_examples', True, {}, 0.1,
       client_weight_lib.ClientWeighting.NUM_EXAMPLES),
      ('non-simulation_clipnorm', False, {
          'clipnorm': 0.2
      }, 0.05, client_weight_lib.ClientWeighting.NUM_EXAMPLES),
      ('non-simulation_clipvalue', False, {
          'clipvalue': 0.1
      }, 0.02, client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_client_tf(self, simulation, optimizer_kwargs, expected_norm,
                     weighting):
    client_tf = model_delta_client_work.build_model_delta_update_with_keras_optimizer(
        model_fn=create_model,
        weighting=weighting,
        use_experimental_simulation_loop=simulation)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs)
    dataset = create_test_dataset()
    client_result, model_output = self.evaluate(
        client_tf(optimizer, create_test_initial_weights(), dataset))
    # Both trainable parameters should have been updated, and we don't return
    # the non-trainable variable.
    self.assertAllGreater(
        np.linalg.norm(client_result.update, axis=-1), expected_norm)
    if weighting == client_weight_lib.ClientWeighting.UNIFORM:
      self.assertEqual(client_result.update_weight, 1.0)
    else:
      self.assertEqual(client_result.update_weight, 8.0)
    self.assertDictContainsSubset(
        {
            'num_examples': 8,
        }, model_output)
    self.assertBetween(model_output['loss'][0], np.finfo(np.float32).eps, 10.0)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    client_tf = model_delta_client_work.build_model_delta_update_with_keras_optimizer(
        model_fn=create_model,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = create_test_dataset()
    init_weights = create_test_initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(optimizer, init_weights, dataset)
    self.assertEqual(self.evaluate(client_outputs[0].update_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs[0].update), [[[0.0], [0.0]], 0.0])

  def test_custom_metrics_aggregator(self):

    def sum_then_finalize_then_times_two(metric_finalizers,
                                         local_unfinalized_metrics_type):

      @federated_computation.federated_computation(
          computation_types.at_clients(local_unfinalized_metrics_type))
      def aggregation_computation(client_local_unfinalized_metrics):
        unfinalized_metrics_sum = intrinsics.federated_sum(
            client_local_unfinalized_metrics)

        @tensorflow_computation.tf_computation(local_unfinalized_metrics_type)
        def finalizer_computation(unfinalized_metrics):
          finalized_metrics = collections.OrderedDict()
          for metric_name, metric_finalizer in metric_finalizers.items():
            finalized_metrics[metric_name] = metric_finalizer(
                unfinalized_metrics[metric_name]) * 2
          return finalized_metrics

        return intrinsics.federated_map(finalizer_computation,
                                        unfinalized_metrics_sum)

      return aggregation_computation

    process = model_delta_client_work.build_model_delta_client_work(
        model_fn=create_model,
        optimizer=sgdm.build_sgdm(1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        metrics_aggregator=sum_then_finalize_then_times_two)
    client_model_weights = [create_test_initial_weights()]
    client_data = [create_test_dataset()]
    output = process.next(process.initialize(), client_model_weights,
                          client_data)
    # Train metrics should be multiplied by two by the custom aggregator.
    self.assertEqual(output.measurements['train']['num_examples'], 16)

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    client_tf = model_delta_client_work.build_model_delta_update_with_keras_optimizer(
        model_fn=create_model,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        use_experimental_simulation_loop=simulation)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = create_test_dataset()
    client_tf(optimizer, create_test_initial_weights(), dataset)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  @parameterized.named_parameters(
      ('tff_simple', sgdm.build_sgdm(1.0)),
      ('tff_momentum', sgdm.build_sgdm(1.0, momentum=0.9)),
      ('keras_simple', lambda: tf.keras.optimizers.SGD(1.0)),
      ('keras_momentum', lambda: tf.keras.optimizers.SGD(1.0, momentum=0.9)))
  def test_execution_with_optimizer(self, optimizer):
    client_work_process = model_delta_client_work.build_model_delta_client_work(
        create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)
    client_data = [create_test_dataset()]
    client_model_weights = [create_test_initial_weights()]

    state = client_work_process.initialize()
    output = client_work_process.next(state, client_model_weights, client_data)

    self.assertCountEqual(output.measurements.keys(), ['train'])


class FunctionalModelDeltaClientWorkExecutionTest(tf.test.TestCase,
                                                  parameterized.TestCase):

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES))
  def test_functional_model_matches_model_fn(self, weighting):
    dataset = create_test_dataset()

    # Build a FunctionalModel based client_model_update procedure. This will
    # be compared to a model_fn based implementation built below.
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=2)
    loss_fn = tf.keras.losses.MeanSquaredError()
    input_spec = dataset.element_spec
    functional_model = functional.functional_model_from_keras(
        keras_model, loss_fn=loss_fn, input_spec=input_spec)

    # Note: we must wrap in a `tf_computation` for the correct graph-context
    # processing of Keras models wrapped as FunctionalModel.
    @tensorflow_computation.tf_computation
    def client_update_functional_model(model_weights, dataset):
      model_delta_fn = model_delta_client_work.build_functional_model_delta_update(
          model=functional_model, weighting=weighting)
      return model_delta_fn(
          sgdm.build_sgdm(learning_rate=0.1), model_weights, dataset)

    # Build a model_fn based client_model_update procedure. This will be
    # comapred to the FunctionalModel variant built above to ensure they
    # procduce the same results.
    def model_fn():
      keras_model = model_examples.build_linear_regression_keras_functional_model(
          feature_dims=2)
      loss_fn = tf.keras.losses.MeanSquaredError()
      input_spec = dataset.element_spec
      return keras_utils.from_keras_model(
          keras_model, loss=loss_fn, input_spec=input_spec)

    client_update_model_fn = model_delta_client_work.build_model_delta_update_with_tff_optimizer(
        model_fn=model_fn, weighting=weighting)
    model_fn_optimizer = sgdm.build_sgdm(learning_rate=0.1)
    model_fn_weights = model_utils.ModelWeights.from_model(model_fn())

    functional_model_weights = functional_model.initial_weights
    for _ in range(10):
      # pylint: disable=cell-var-from-loop
      model_fn_output, _ = client_update_model_fn(model_fn_optimizer,
                                                  model_fn_weights, dataset)
      functional_model_output, _ = client_update_functional_model(
          functional_model_weights, dataset)
      self.assertAllClose(model_fn_output.update,
                          functional_model_output.update)
      self.assertAllClose(model_fn_output.update_weight,
                          functional_model_output.update_weight)
      model_fn_weights = attr.evolve(
          model_fn_weights,
          trainable=tf.nest.map_structure(
              lambda u, v: u + v * model_fn_output.update_weight,
              model_fn_weights.trainable, model_fn_output.update))
      functional_model_weights = (tf.nest.map_structure(
          lambda u, v: u + v * functional_model_output.update_weight,
          functional_model_weights[0],
          functional_model_output.update), functional_model_weights[1])
      # pylint: enable=cell-var-from-loop
    self.assertAllClose(
        attr.astuple(model_fn_weights), functional_model_weights)

  def test_metrics_output(self):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=2)
    loss_fn = tf.keras.losses.MeanSquaredError()
    dataset = create_test_dataset()
    input_spec = dataset.element_spec

    def build_metrics_fn():
      return collections.OrderedDict(num_examples=counters.NumExamplesCounter())

    functional_model = functional.functional_model_from_keras(
        keras_model,
        loss_fn=loss_fn,
        input_spec=input_spec,
        metrics_constructor=build_metrics_fn)

    process = model_delta_client_work.build_functional_model_delta_client_work(
        model=functional_model,
        optimizer=sgdm.build_sgdm(1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)
    num_clients = 3
    client_model_weights = [functional_model.initial_weights] * num_clients
    client_datasets = [dataset] * num_clients
    output = process.next(process.initialize(), client_model_weights,
                          client_datasets)
    self.assertEqual(output.measurements['train']['num_examples'],
                     8 * num_clients)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
