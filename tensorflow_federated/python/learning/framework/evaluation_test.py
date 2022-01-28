# Copyright 2021, The TensorFlow Federated Authors.
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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import evaluation

# Convenience aliases.
StructType = computation_types.StructType
TensorType = computation_types.TensorType


def keras_model_builder():
  # Create a simple linear regression model, single output.
  # We initialize all weights to one.
  return tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='ones',
          bias_initializer='ones',
          input_shape=(1,))
  ])


def create_dataset():
  # Create data satisfying y = 2*x + 1
  x = [[1.0], [2.0], [3.0]]
  y = [[3.0], [5.0], [7.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def get_input_spec():
  return create_dataset().element_spec


def tff_model_builder():

  return keras_utils.from_keras_model(
      keras_model=keras_model_builder(),
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanSquaredError()])


class BuildEvalWorkTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('default_simulation_loop', False),
      ('experimental_simulation_loop', True),
  )
  def test_evaluation_types(self, use_experimental_simulation_loop):
    model = tff_model_builder()
    model_weights_type = model_utils.weights_type_from_model(model)
    client_eval_work = evaluation.build_eval_work(
        tff_model_builder, model_weights_type, get_input_spec(),
        use_experimental_simulation_loop)

    self.assertIsInstance(client_eval_work, computation_base.Computation)
    type_signature = client_eval_work.type_signature
    self.assertLen(type_signature.parameter, 2)
    type_signature.parameter[0].check_assignable_from(model_weights_type)
    type_signature.parameter[1].check_assignable_from(
        computation_types.SequenceType(get_input_spec()))

  @parameterized.named_parameters(
      ('default_simulation_loop', False),
      ('experimental_simulation_loop', True),
  )
  def test_evaluation_on_default_weights(self,
                                         use_experimental_simulation_loop):
    model = tff_model_builder()
    model_weights_type = model_utils.weights_type_from_model(model)
    model_weights = model_utils.ModelWeights.from_model(model)

    client_eval_work = evaluation.build_eval_work(
        tff_model_builder, model_weights_type, get_input_spec(),
        use_experimental_simulation_loop)

    # All weights are set to 1, so the model outputs f(x) = x + 1.
    eval_metrics = client_eval_work(model_weights, create_dataset())
    self.assertCountEqual(eval_metrics.keys(),
                          ['local_outputs', 'num_examples'])
    self.assertEqual(eval_metrics['num_examples'], 3)

    local_outputs = eval_metrics['local_outputs']
    self.assertCountEqual(
        local_outputs.keys(),
        ['loss', 'mean_squared_error', 'num_examples', 'num_batches'])
    self.assertEqual(local_outputs['loss'], local_outputs['mean_squared_error'])
    expected_loss_sum = (3.0 - 2.0)**2 + (5.0 - 3.0)**2 + (7.0 - 4.0)**2
    self.assertAllClose(
        local_outputs['loss'], [expected_loss_sum, 3.0], atol=1e-6)

  def test_evaluation_on_input_weights(self):
    model = tff_model_builder()
    model_weights_type = model_utils.weights_type_from_model(model)
    model_weights = model_utils.ModelWeights.from_model(model)
    zero_weights = tf.nest.map_structure(tf.zeros_like, model_weights)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights, zero_weights)

    client_eval_work = evaluation.build_eval_work(tff_model_builder,
                                                  model_weights_type,
                                                  get_input_spec())

    # We compute metrics where all weights are set to 0, so the model should
    # output f(x) = 0.
    eval_metrics = client_eval_work(model_weights, create_dataset())
    self.assertCountEqual(eval_metrics.keys(),
                          ['local_outputs', 'num_examples'])
    self.assertEqual(eval_metrics['num_examples'], 3)

    local_outputs = eval_metrics['local_outputs']
    self.assertCountEqual(
        local_outputs.keys(),
        ['loss', 'mean_squared_error', 'num_examples', 'num_batches'])
    self.assertEqual(local_outputs['loss'], local_outputs['mean_squared_error'])
    expected_loss_sum = 9.0 + 25.0 + 49.0
    self.assertAllClose(
        local_outputs['loss'], [expected_loss_sum, 3.0], atol=1e-6)


class BuildModelMetricsAggregatorTest(tf.test.TestCase):

  def _get_metrics_type(self):
    return StructType([
        ('local_outputs',
         StructType([
             ('mean_squared_error', (TensorType(tf.float32),
                                     TensorType(tf.float32))),
             ('loss', (TensorType(tf.float32), TensorType(tf.float32))),
             ('num_examples', (TensorType(tf.int64),)),
             ('num_batches', (TensorType(tf.int64),)),
         ])),
        ('num_examples', TensorType(tf.float32)),
    ])

  def _get_aggregated_metrics_type(self):
    return StructType([
        ('eval',
         StructType([
             ('mean_squared_error', TensorType(tf.float32)),
             ('loss', TensorType(tf.float32)),
             ('num_examples', TensorType(tf.int64)),
             ('num_batches', TensorType(tf.int64)),
         ])),
        ('stat', StructType([
            ('num_examples', TensorType(tf.float32)),
        ])),
    ])

  def test_metrics_aggregator_types(self):
    model = tff_model_builder()
    metrics_type = self._get_metrics_type()
    model_metrics_aggregator = evaluation.build_model_metrics_aggregator(
        model, metrics_type)
    self.assertIsInstance(model_metrics_aggregator,
                          computation_base.Computation)
    aggregator_parameter = model_metrics_aggregator.type_signature.parameter
    aggregator_parameter.check_assignable_from(
        computation_types.at_clients(metrics_type))

    aggregator_result = model_metrics_aggregator.type_signature.result
    aggregator_result.check_assignable_from(
        computation_types.at_server(self._get_aggregated_metrics_type()))

  def test_metrics_aggregator_correctness_with_one_client(self):
    client_metrics = collections.OrderedDict(
        local_outputs=collections.OrderedDict(
            mean_squared_error=(4.0, 2.0),
            loss=(5.0, 1.0),
            num_examples=(10,),
            num_batches=(2,)),
        num_examples=10.0)

    model = tff_model_builder()
    metrics_type = self._get_metrics_type()
    model_metrics_aggregator = evaluation.build_model_metrics_aggregator(
        model, metrics_type)

    aggregate_metrics = model_metrics_aggregator([client_metrics])
    expected_metrics = collections.OrderedDict(
        eval=collections.OrderedDict(
            mean_squared_error=2.0, loss=5.0, num_examples=10, num_batches=2),
        stat=collections.OrderedDict(num_examples=10.0))
    self.assertAllClose(aggregate_metrics, expected_metrics, atol=1e-6)

  def test_metrics_aggregator_correctness_with_three_client(self):
    client_metrics1 = collections.OrderedDict(
        local_outputs=collections.OrderedDict(
            mean_squared_error=(4.0, 2.0),
            loss=(5.0, 1.0),
            num_examples=(3,),
            num_batches=(3,)),
        num_examples=10.0)
    client_metrics2 = collections.OrderedDict(
        local_outputs=collections.OrderedDict(
            mean_squared_error=(4.0, 4.0),
            loss=(1.0, 5.0),
            num_examples=(2,),
            num_batches=(1,)),
        num_examples=7.0)
    client_metrics3 = collections.OrderedDict(
        local_outputs=collections.OrderedDict(
            mean_squared_error=(6.0, 2.0),
            loss=(5.0, 5.0),
            num_examples=(5,),
            num_batches=(2,)),
        num_examples=3.0)

    model = tff_model_builder()
    metrics_type = self._get_metrics_type()
    model_metrics_aggregator = evaluation.build_model_metrics_aggregator(
        model, metrics_type)

    federated_metrics = [client_metrics1, client_metrics2, client_metrics3]

    aggregate_metrics = model_metrics_aggregator(federated_metrics)
    expected_metrics = collections.OrderedDict(
        eval=collections.OrderedDict(
            mean_squared_error=1.75, loss=1.0, num_examples=10, num_batches=6),
        stat=collections.OrderedDict(num_examples=20.0))
    self.assertAllClose(aggregate_metrics, expected_metrics, atol=1e-6)


class EvalComposerTest(tf.test.TestCase):

  def create_test_distributor(self):

    @computations.federated_computation(computation_types.at_server(tf.float32))
    def basic_distribute(x):
      return intrinsics.federated_broadcast(x)

    return basic_distribute

  def create_test_client_work(self):

    @tf.function
    def multiply_and_add(x, dataset):
      total_sum = 0.0
      for a in dataset:
        total_sum = total_sum + x * a
      return total_sum

    @computations.tf_computation(tf.float32,
                                 computation_types.SequenceType(tf.float32))
    def basic_client_work(x, dataset):
      return multiply_and_add(x, dataset)

    return basic_client_work

  def create_test_aggregator(self):

    @computations.federated_computation(
        computation_types.at_clients(tf.float32))
    def basic_aggregate(x):
      return intrinsics.federated_sum(x)

    return basic_aggregate

  def test_basic_composition_has_expected_types(self):
    eval_computation = evaluation.compose_eval_computation(
        self.create_test_distributor(), self.create_test_client_work(),
        self.create_test_aggregator())
    expected_parameter = computation_types.StructType([
        computation_types.at_server(tf.float32),
        computation_types.at_clients(
            computation_types.SequenceType(tf.float32))
    ])
    eval_computation.type_signature.parameter.check_assignable_from(
        expected_parameter)
    expected_result = computation_types.at_server(tf.float32)
    eval_computation.type_signature.result.check_assignable_from(
        expected_result)

  def test_basic_composition_computes_expected_value(self):
    eval_computation = evaluation.compose_eval_computation(
        self.create_test_distributor(), self.create_test_client_work(),
        self.create_test_aggregator())
    client_data = [[1.0, 2.0, 3.0], [-1.0, -2.0, -5.0]]
    actual_result = eval_computation(1.0, client_data)
    self.assertEqual(actual_result, -2.0)

  def test_basic_composition_with_struct_type(self):
    distributor_struct = computation_types.at_server(StructType([tf.float32]))

    @computations.federated_computation(distributor_struct)
    def distributor_with_struct_parameter(x):
      return intrinsics.federated_broadcast(x[0])

    eval_computation = evaluation.compose_eval_computation(
        distributor_with_struct_parameter, self.create_test_client_work(),
        self.create_test_aggregator())
    expected_parameter = computation_types.StructType([
        distributor_struct,
        computation_types.at_clients(
            computation_types.SequenceType(tf.float32))
    ])
    eval_computation.type_signature.parameter.check_assignable_from(
        expected_parameter)
    expected_result = computation_types.at_server(tf.float32)
    eval_computation.type_signature.result.check_assignable_from(
        expected_result)

  def test_raises_on_python_callable_distributor(self):

    def python_distributor(x):
      return x

    with self.assertRaises(TypeError):
      evaluation.compose_eval_computation(python_distributor,
                                          self.create_test_client_work(),
                                          self.create_test_aggregator())

  def test_raises_on_python_callable_client_work(self):

    def python_client_work(x, y):
      del y
      return x

    with self.assertRaises(TypeError):
      evaluation.compose_eval_computation(self.create_test_distributor(),
                                          python_client_work,
                                          self.create_test_aggregator())

  def test_raises_on_python_callable_aggregator(self):

    def python_aggregator(x):
      return x

    with self.assertRaises(TypeError):
      evaluation.compose_eval_computation(self.create_test_distributor(),
                                          self.create_test_client_work(),
                                          python_aggregator)

  def test_no_arg_distributor_raises(self):

    @computations.federated_computation
    def no_arg_distribute():
      return intrinsics.federated_value(1.0, placements.CLIENTS)

    with self.assertRaises(evaluation.FederatedEvalTypeError):
      evaluation.compose_eval_computation(no_arg_distribute,
                                          self.create_test_client_work(),
                                          self.create_test_aggregator())

  def test_two_arg_distributor_raises(self):

    @computations.federated_computation(
        computation_types.at_server(tf.float32),
        computation_types.at_server(tf.float32))
    def two_arg_distribute(x, y):
      del y
      return intrinsics.federated_broadcast(x)

    with self.assertRaises(evaluation.FederatedEvalTypeError):
      evaluation.compose_eval_computation(two_arg_distribute,
                                          self.create_test_client_work(),
                                          self.create_test_aggregator())

  def test_distributor_with_client_parameter_raises(self):

    @computations.federated_computation(
        computation_types.at_clients(tf.float32))
    def distributor_with_client_parameter(x):
      return x

    with self.assertRaises(evaluation.FederatedEvalTypeError):
      evaluation.compose_eval_computation(distributor_with_client_parameter,
                                          self.create_test_client_work(),
                                          self.create_test_aggregator())

  def test_distributor_with_server_result_raises(self):

    @computations.federated_computation(computation_types.at_server(tf.float32))
    def distributor_with_server_result(x):
      return x

    with self.assertRaises(evaluation.FederatedEvalTypeError):
      evaluation.compose_eval_computation(distributor_with_server_result,
                                          self.create_test_client_work(),
                                          self.create_test_aggregator())

  def test_federated_client_work_raises(self):

    @computations.federated_computation(
        computation_types.at_clients(tf.float32),
        computation_types.at_clients(
            computation_types.SequenceType(tf.float32)))
    def federated_client_work(model, dataset):
      return intrinsics.federated_map(self.create_test_client_work(),
                                      (model, dataset))

    with self.assertRaises(evaluation.FederatedEvalTypeError):
      evaluation.compose_eval_computation(self.create_test_distributor(),
                                          federated_client_work,
                                          self.create_test_aggregator())

  def test_no_arg_aggregator_raises(self):

    @computations.federated_computation
    def no_arg_aggregate():
      return intrinsics.federated_value(1.0, placements.SERVER)

    with self.assertRaises(evaluation.FederatedEvalTypeError):
      evaluation.compose_eval_computation(self.create_test_distributor(),
                                          self.create_test_client_work(),
                                          no_arg_aggregate)

  def test_two_arg_aggregator_raises(self):

    @computations.federated_computation(
        computation_types.at_clients(tf.float32),
        computation_types.at_clients(tf.float32))
    def two_arg_aggregate(x, y):
      del y
      return intrinsics.federated_sum(x)

    with self.assertRaises(evaluation.FederatedEvalTypeError):
      evaluation.compose_eval_computation(self.create_test_distributor(),
                                          self.create_test_client_work(),
                                          two_arg_aggregate)

  def test_aggregator_with_server_parameter_raises(self):

    @computations.federated_computation(computation_types.at_server(tf.float32))
    def aggregator_with_server_parameter(x):
      return x

    with self.assertRaises(evaluation.FederatedEvalTypeError):
      evaluation.compose_eval_computation(self.create_test_distributor(),
                                          self.create_test_client_work(),
                                          aggregator_with_server_parameter)

  def test_aggregator_with_client_result_raises(self):

    @computations.federated_computation(
        computation_types.at_clients(tf.float32))
    def aggregator_with_client_result(x):
      return x

    with self.assertRaises(evaluation.FederatedEvalTypeError):
      evaluation.compose_eval_computation(self.create_test_distributor(),
                                          self.create_test_client_work(),
                                          aggregator_with_client_result)

  def test_distributor_client_work_type_mismatch_raises(self):

    @computations.tf_computation(tf.int32, tf.float32)
    def client_work_with_int_parameter(x, y):
      del x
      return y

    with self.assertRaises(evaluation.FederatedEvalInputOutputError):
      evaluation.compose_eval_computation(self.create_test_distributor(),
                                          client_work_with_int_parameter,
                                          self.create_test_aggregator())

  def test_client_work_aggregator_type_mismatch_raises(self):

    @computations.tf_computation(tf.float32, tf.int32)
    def client_work_with_int_result(x, y):
      del x
      return y

    with self.assertRaises(evaluation.FederatedEvalInputOutputError):
      evaluation.compose_eval_computation(self.create_test_distributor(),
                                          client_work_with_int_result,
                                          self.create_test_aggregator())


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
