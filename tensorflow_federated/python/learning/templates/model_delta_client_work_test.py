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
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import model_delta_client_work


class ModelDeltaClientWorkComputationTest(test_case.TestCase,
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
                loss=tf.float32, num_examples=tf.int32),
            stat=collections.OrderedDict(num_examples=tf.int64)))

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


class ModelDeltaClientWorkExecutionTest(test_case.TestCase,
                                        parameterized.TestCase):
  """Tests of the client work of FedAvg using a common model and data."""

  def create_dataset(self):
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

  def create_model(self):
    return model_examples.LinearRegression(feature_dim=2)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable=[tf.zeros((2, 1)), tf.constant(0.0)],
        non_trainable=[0.0],
    )

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES))
  def test_keras_tff_client_work_equal(self, weighting):
    dataset = self.create_dataset()
    client_update_keras = model_delta_client_work.build_model_delta_update_with_keras_optimizer(
        model_fn=self.create_model, weighting=weighting)
    client_update_tff = model_delta_client_work.build_model_delta_update_with_tff_optimizer(
        model_fn=self.create_model, weighting=weighting)
    keras_result = client_update_keras(
        tf.keras.optimizers.SGD(learning_rate=0.1), self.initial_weights(),
        dataset)
    tff_result = client_update_tff(
        sgdm.build_sgdm(learning_rate=0.1), self.initial_weights(), dataset)
    self.assertAllClose(keras_result[0].update, tff_result[0].update)
    self.assertEqual(keras_result[0].update_weight, tff_result[0].update_weight)
    self.assertAllClose(keras_result[1], tff_result[1])
    self.assertDictEqual(keras_result[2], tff_result[2])

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
        model_fn=self.create_model,
        weighting=weighting,
        use_experimental_simulation_loop=simulation)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs)
    dataset = self.create_dataset()
    client_result, model_output, stat_output = self.evaluate(
        client_tf(optimizer, self.initial_weights(), dataset))
    # Both trainable parameters should have been updated, and we don't return
    # the non-trainable variable.
    self.assertAllGreater(
        np.linalg.norm(client_result.update, axis=-1), expected_norm)
    if weighting == client_weight_lib.ClientWeighting.UNIFORM:
      self.assertEqual(client_result.update_weight, 1.0)
    else:
      self.assertEqual(client_result.update_weight, 8.0)
    self.assertEqual(stat_output['num_examples'], 8)
    self.assertDictContainsSubset(
        {
            'num_examples': 8,
            'num_examples_float': 8.0,
            'num_batches': 3,
        }, model_output)
    self.assertBetween(model_output['loss'], np.finfo(np.float32).eps, 10.0)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    client_tf = model_delta_client_work.build_model_delta_update_with_keras_optimizer(
        model_fn=self.create_model,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = self.create_dataset()
    init_weights = self.initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(optimizer, init_weights, dataset)
    self.assertEqual(self.evaluate(client_outputs[0].update_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs[0].update), [[[0.0], [0.0]], 0.0])

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    client_tf = model_delta_client_work.build_model_delta_update_with_keras_optimizer(
        model_fn=self.create_model,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        use_experimental_simulation_loop=simulation)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = self.create_dataset()
    client_tf(optimizer, self.initial_weights(), dataset)
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
        self.create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)
    client_data = [self.create_dataset()]
    client_model_weights = [self.initial_weights()]

    state = client_work_process.initialize()
    output = client_work_process.next(state, client_model_weights, client_data)

    self.assertEqual(8, output.measurements['stat']['num_examples'])


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
