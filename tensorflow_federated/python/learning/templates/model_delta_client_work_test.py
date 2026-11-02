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
import itertools
from unittest import mock

from absl.testing import parameterized
import federated_language
import keras
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import cpp_execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_backend import serialization_utils
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import model_delta_client_work
from tensorflow_federated.python.learning.templates import type_checks


class ModelDeltaClientWorkComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_initialize_has_expected_type_signature_with_tff_optimizer(
      self, weighting
  ):
    optimizer = sgdm.build_sgdm(learning_rate=1.0)
    model_fn = model_examples.LinearRegression

    client_work_process = model_delta_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting
    )

    expected_state_type = federated_language.FederatedType(
        collections.OrderedDict(learning_rate=np.float32),
        federated_language.SERVER,
    )
    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        client_work_process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_next_has_expected_type_signature_with_tff_optimizer(self, weighting):
    optimizer = sgdm.build_sgdm(learning_rate=1.0)
    model_fn = model_examples.LinearRegression

    client_work_process = model_delta_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting
    )

    mw_type = model_weights_lib.ModelWeights(
        trainable=federated_language.to_type(
            [(np.float32, (2, 1)), np.float32]
        ),
        non_trainable=federated_language.to_type([np.float32]),
    )
    expected_param_model_weights_type = federated_language.FederatedType(
        mw_type, federated_language.CLIENTS
    )
    element_type = tensorflow_types.to_type(model_fn().input_spec)
    expected_param_data_type = federated_language.FederatedType(
        federated_language.SequenceType(element_type),
        federated_language.CLIENTS,
    )
    expected_result_type = federated_language.FederatedType(
        client_works.ClientResult(
            update=mw_type.trainable,
            update_weight=federated_language.TensorType(np.float32),
        ),
        federated_language.CLIENTS,
    )
    expected_state_type = federated_language.FederatedType(
        collections.OrderedDict(learning_rate=np.float32),
        federated_language.SERVER,
    )
    expected_measurements_type = federated_language.FederatedType(
        collections.OrderedDict(
            train=collections.OrderedDict(
                loss=np.float32,
                num_examples=np.int32,
                num_non_finite_clients=np.int32,
            )
        ),
        federated_language.SERVER,
    )
    expected_next_type = federated_language.FunctionType(
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
    self.assertTrue(
        client_work_process.next.type_signature.is_equivalent_to(
            expected_next_type
        )
    )

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_get_hparams_has_expected_type_signature_with_tff_optimizer(
      self, weighting
  ):
    optimizer = sgdm.build_sgdm(learning_rate=1.0)
    model_fn = model_examples.LinearRegression

    client_work_process = model_delta_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting
    )

    expected_state_type = collections.OrderedDict(learning_rate=np.float32)
    expected_hparams_type = expected_state_type
    expected_get_hparams_type = federated_language.FunctionType(
        parameter=expected_state_type, result=expected_hparams_type
    )
    self.assertTrue(
        client_work_process.get_hparams.type_signature.is_equivalent_to(
            expected_get_hparams_type
        )
    )

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_set_hparams_has_expected_type_signature_with_tff_optimizer(
      self, weighting
  ):
    optimizer = sgdm.build_sgdm(learning_rate=1.0)
    model_fn = model_examples.LinearRegression

    client_work_process = model_delta_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting
    )

    expected_state_type = collections.OrderedDict(learning_rate=np.float32)
    expected_hparams_type = expected_state_type
    expected_parameter_type = federated_language.StructType(
        [('state', expected_state_type), ('hparams', expected_hparams_type)]
    )
    expected_set_hparams_type = federated_language.FunctionType(
        parameter=expected_parameter_type, result=expected_state_type
    )
    self.assertTrue(
        client_work_process.set_hparams.type_signature.is_equivalent_to(
            expected_set_hparams_type
        )
    )

  def test_raises_with_created_model(self):
    with self.assertRaises(TypeError):
      model_delta_client_work.build_model_delta_client_work(
          model_examples.LinearRegression(),
          sgdm.build_sgdm(learning_rate=1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      )


def create_test_dataset() -> tf.data.Dataset:
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
          y=[[0.0], [0.0], [1.0], [1.0]],
      )
  )
  # Repeat the dataset 3 times with batches of 3 examples, producing 3
  # minibatches. Note that `batch` is required for this dataset to be useable,
  # as it adds the batch dimension which is expected by the model.
  return dataset.repeat(3).batch(3)


def create_test_initial_weights() -> model_weights_lib.ModelWeights:
  return model_weights_lib.ModelWeights(
      trainable=[np.zeros([2, 1], np.float32), 0.0],
      non_trainable=[0.0],
  )


def create_model():
  return model_examples.LinearRegression(feature_dim=2)


class ModelDeltaClientWorkExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):
  """Tests of the client work of FedAvg using a common model and data."""

  @parameterized.named_parameters(
      (
          'dataset_reduce_noclip_uniform',
          loop_builder.LoopImplementation.DATASET_REDUCE,
          0.1,
          client_weight_lib.ClientWeighting.UNIFORM,
      ),
      (
          'dataset_reduce_noclip_num_examples',
          loop_builder.LoopImplementation.DATASET_REDUCE,
          0.1,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
      (
          'dataset_iterator_noclip_uniform',
          loop_builder.LoopImplementation.DATASET_ITERATOR,
          0.1,
          client_weight_lib.ClientWeighting.UNIFORM,
      ),
      (
          'dataset_iterator_noclip_num_examples',
          loop_builder.LoopImplementation.DATASET_ITERATOR,
          0.1,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
  )
  def test_client_tf(self, loop_implementation, expected_norm, weighting):
    client_tf = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=weighting,
            loop_implementation=loop_implementation,
        )
    )
    optimizer = sgdm.build_sgdm(learning_rate=0.1)
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
      self.assertEqual(client_result.update_weight, 12.0)
    expected = {'num_examples': 12}
    self.assertEqual(model_output, {**model_output, **expected})
    self.assertBetween(model_output['loss'][0], np.finfo(np.float32).eps, 10.0)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    client_tf = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            loop_implementation=loop_builder.LoopImplementation.DATASET_REDUCE,
        )
    )
    optimizer = sgdm.build_sgdm(learning_rate=0.1)
    dataset = create_test_dataset()
    init_weights = create_test_initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(optimizer, init_weights, dataset)
    # Since this client has non-finite model weights update, we will zero out
    # its `update` and set `update_weight` to zero.
    self.assertEqual(client_outputs[0].update_weight, 0.0)
    self.assertAllClose(client_outputs[0].update, [[[0.0], [0.0]], 0.0])
    # Since this client has non-finite model weights update, we will reset its
    # local metrics, which essentially excludes it from server aggregation.
    tf.nest.map_structure(
        self.assertAllEqual,
        client_outputs[1],
        collections.OrderedDict(
            loss=[0.0, 0.0], num_examples=0, num_non_finite_clients=1
        ),
    )

  def test_correct_update_weight_with_traced_function(self):
    client_tf = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            loop_implementation=loop_builder.LoopImplementation.DATASET_REDUCE,
        )
    )
    optimizer = sgdm.build_sgdm(learning_rate=0.1)
    init_weights = create_test_initial_weights()
    dataset_with_nan = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, np.nan]],
            y=[[0.0], [0.0], [1.0], [1.0]],
        )
    ).batch(1)
    dataset_wo_nan = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]],
        )
    ).batch(1)
    # Obtain a concrete function after federated_language.framework.
    client_concrete_fn = client_tf.get_concrete_function(
        optimizer, init_weights, dataset_wo_nan, optimizer_hparams=None
    )
    # Execute the traced function with different data inputs. If the data input
    # has nan, the corresponding model delta will be non-finite, and hence,
    # the `update_weight` should be 0.0 (i.e., exclude this client from model
    # aggregation). Otherwise, the `update_weight` should be larger than 0.0.
    client_outputs = client_concrete_fn(
        optimizer, init_weights, dataset_wo_nan, optimizer_hparams=None
    )
    self.assertGreater(client_outputs[0].update_weight, 0.0)
    client_outputs = client_concrete_fn(
        optimizer, init_weights, dataset_with_nan, optimizer_hparams=None
    )
    self.assertEqual(client_outputs[0].update_weight, 0.0)

  def test_custom_metrics_aggregator(self):

    def sum_then_finalize_then_times_two(metric_finalizers):

      @federated_language.federated_computation
      def aggregation_computation(client_local_unfinalized_metrics):
        unfinalized_metrics_sum = federated_language.federated_sum(
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

        return federated_language.federated_map(
            finalizer_computation, unfinalized_metrics_sum
        )

      return aggregation_computation

    process = model_delta_client_work.build_model_delta_client_work(
        model_fn=create_model,
        optimizer=sgdm.build_sgdm(learning_rate=1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        metrics_aggregator=sum_then_finalize_then_times_two,
    )
    client_model_weights = [create_test_initial_weights()]
    client_data = [create_test_dataset()]
    output = process.next(
        process.initialize(), client_model_weights, client_data
    )
    # Train metrics should be multiplied by two by the custom aggregator.
    self.assertEqual(output.measurements['train']['num_examples'], 24)

  @parameterized.named_parameters(
      ('dataset_reduce', loop_builder.LoopImplementation.DATASET_REDUCE),
      ('dataset_iterator', loop_builder.LoopImplementation.DATASET_ITERATOR),
  )
  @mock.patch.object(loop_builder, 'build_training_loop')
  def test_client_tf_dataset_loops(self, loop_implementation, mock_method):
    model_delta_client_work.build_model_delta_update_with_tff_optimizer(
        model_fn=create_model,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        loop_implementation=loop_implementation,
    )
    mock_method.assert_called_once_with(loop_implementation=loop_implementation)

  @parameterized.named_parameters(
      ('tff_simple', sgdm.build_sgdm(learning_rate=1.0)),
      ('tff_momentum', sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)),
  )
  def test_execution_with_optimizer(self, optimizer):
    client_work_process = model_delta_client_work.build_model_delta_client_work(
        create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    client_data = [create_test_dataset()]
    client_model_weights = [create_test_initial_weights()]

    state = client_work_process.initialize()
    output = client_work_process.next(state, client_model_weights, client_data)

    self.assertCountEqual(output.measurements.keys(), ['train'])

  def test_get_hparams_returns_expected_result_with_tff_optimizer(self):
    optimizer = sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)
    client_work_process = model_delta_client_work.build_model_delta_client_work(
        create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    state = client_work_process.initialize()

    hparams = client_work_process.get_hparams(state)

    expected_hparams = collections.OrderedDict(learning_rate=1.0, momentum=0.9)
    self.assertDictEqual(hparams, expected_hparams)

  def test_set_hparams_returns_expected_result_with_tff_optimizer(self):
    optimizer = sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)
    client_work_process = model_delta_client_work.build_model_delta_client_work(
        create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    state = client_work_process.initialize()
    hparams = collections.OrderedDict(learning_rate=0.5, momentum=0.3)

    state = client_work_process.set_hparams(state, hparams)

    self.assertDictEqual(state, hparams)

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_tff_client_work_uses_optimizer_hparams(self, weighting):
    dataset = create_test_dataset()
    optimizer1 = sgdm.build_sgdm(learning_rate=0.0)
    optimizer2 = sgdm.build_sgdm(learning_rate=0.1)
    optimizer_hparams = collections.OrderedDict(learning_rate=1.0)

    client_update_tff = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=weighting,
            loop_implementation=loop_builder.LoopImplementation.DATASET_REDUCE,
        )
    )
    result1 = client_update_tff(
        optimizer1, create_test_initial_weights(), dataset, optimizer_hparams
    )
    client_update_tff = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=weighting,
            loop_implementation=loop_builder.LoopImplementation.DATASET_REDUCE,
        )
    )
    result2 = client_update_tff(
        optimizer2, create_test_initial_weights(), dataset, optimizer_hparams
    )

    self.assertAllClose(result1[0].update, result2[0].update)
    self.assertEqual(result1[0].update_weight, result2[0].update_weight)
    self.assertAllClose(result1[1], result2[1])


class FunctionalModelDeltaClientWorkExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      (
          f'{weighting[0]}_{is_simulation_loop[0]}',
          weighting[1],
          is_simulation_loop[1],
      )
      for weighting, is_simulation_loop in itertools.product(
          [
              ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
              ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
          ],
          [
              (
                  'dataset_iterator',
                  loop_builder.LoopImplementation.DATASET_ITERATOR,
              ),
              (
                  'dataset_reduce',
                  loop_builder.LoopImplementation.DATASET_REDUCE,
              ),
          ],
      )
  )
  def test_functional_model_matches_model_fn(
      self, weighting, loop_implementation
  ):
    dataset = create_test_dataset()

    # Build a FunctionalModel based client_model_update procedure. This will
    # be compared to a model_fn based implementation built below.
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=2
    )
    loss_fn = keras.losses.MeanSquaredError()
    input_spec = dataset.element_spec
    functional_model = functional.functional_model_from_keras(
        keras_model, loss_fn=loss_fn, input_spec=input_spec
    )

    # Note: we must wrap in a `tf_computation` for the correct graph-context
    # processing of Keras models wrapped as FunctionalModel.
    @tensorflow_computation.tf_computation
    def client_update_functional_model(model_weights, dataset):
      model_delta_fn = (
          model_delta_client_work.build_functional_model_delta_update(
              model=functional_model,
              weighting=weighting,
              loop_implementation=loop_implementation,
          )
      )
      return model_delta_fn(
          sgdm.build_sgdm(learning_rate=0.1), model_weights, dataset
      )

    # Build a model_fn based client_model_update procedure. This will be
    # comapred to the FunctionalModel variant built above to ensure they
    # procduce the same results.
    def model_fn():
      keras_model = (
          model_examples.build_linear_regression_keras_functional_model(
              feature_dims=2
          )
      )
      loss_fn = tf.keras.losses.MeanSquaredError()
      input_spec = dataset.element_spec
      return keras_utils.from_keras_model(
          keras_model, loss=loss_fn, input_spec=input_spec
      )

    client_update_model_fn = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=model_fn,
            weighting=weighting,
            loop_implementation=loop_implementation,
        )
    )
    model_fn_optimizer = sgdm.build_sgdm(learning_rate=0.1)
    model_fn_weights = model_weights_lib.ModelWeights.from_model(model_fn())

    functional_model_weights = functional_model.initial_weights
    for _ in range(10):
      # pylint: disable=cell-var-from-loop
      model_fn_output, _ = client_update_model_fn(
          model_fn_optimizer, model_fn_weights, dataset
      )
      functional_model_output, _ = client_update_functional_model(
          functional_model_weights, dataset
      )
      self.assertAllClose(
          model_fn_output.update, functional_model_output.update
      )
      self.assertAllClose(
          model_fn_output.update_weight, functional_model_output.update_weight
      )
      model_fn_weights = model_weights_lib.ModelWeights(
          trainable=tf.nest.map_structure(
              lambda u, v: u + v * model_fn_output.update_weight,
              model_fn_weights.trainable,
              model_fn_output.update,
          ),
          non_trainable=model_fn_weights.non_trainable,
      )
      functional_model_weights = (
          tf.nest.map_structure(
              lambda u, v: u + v * functional_model_output.update_weight,
              functional_model_weights[0],
              functional_model_output.update,
          ),
          functional_model_weights[1],
      )
      # pylint: enable=cell-var-from-loop
    self.assertAllClose(tuple(model_fn_weights), functional_model_weights)

  def _create_dataset_and_functional_model(self):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=2
    )
    loss_fn = keras.losses.MeanSquaredError()
    dataset = create_test_dataset()
    input_spec = dataset.element_spec

    def build_metrics_fn():
      return collections.OrderedDict(num_examples=counters.NumExamplesCounter())

    functional_model = functional.functional_model_from_keras(
        keras_model,
        loss_fn=loss_fn,
        input_spec=input_spec,
        metrics_constructor=build_metrics_fn,
    )
    return dataset, functional_model

  def test_metrics_output(self):
    dataset, functional_model = self._create_dataset_and_functional_model()
    process = model_delta_client_work.build_functional_model_delta_client_work(
        model=functional_model,
        optimizer=sgdm.build_sgdm(learning_rate=1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    num_clients = 3
    client_model_weights = [functional_model.initial_weights] * num_clients
    client_datasets = [dataset] * num_clients
    output = process.next(
        process.initialize(), client_model_weights, client_datasets
    )
    self.assertEqual(
        output.measurements['train']['num_examples'], 12 * num_clients
    )

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_metrics_output_with_non_finite_updates(self, bad_value):
    dataset, functional_model = self._create_dataset_and_functional_model()
    process = model_delta_client_work.build_functional_model_delta_client_work(
        model=functional_model,
        optimizer=sgdm.build_sgdm(learning_rate=1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    initial_weights = (
        (
            np.array([[0.0], [0.0]], dtype=np.float32),
            np.array([bad_value], dtype=np.float32),
        ),
        (),
    )
    num_clients = 3
    client_model_weights = [initial_weights] * num_clients
    client_datasets = [dataset] * num_clients
    output = process.next(
        process.initialize(), client_model_weights, client_datasets
    )
    # Since all clients have non-finite model weights update, we reset their
    # local metrics. The resulting aggregated metric at server is zero.
    self.assertEqual(output.measurements['train']['num_examples'], 0)
    self.assertEqual(
        output.measurements['train']['num_non_finite_clients'], num_clients
    )

  @parameterized.named_parameters(
      (
          'dataset_iterator',
          loop_builder.LoopImplementation.DATASET_ITERATOR,
      ),
      (
          'dataset_reduce',
          loop_builder.LoopImplementation.DATASET_REDUCE,
      ),
  )
  def test_functional_client_update_advances_dropout_seed_across_batches(
      self, loop_implementation
  ):
    num_batches = 3
    feature_dim = 4
    dropout_rate = 0.5
    dropout_seed = 42
    learning_rate = 0.1
    dataset = tf.data.Dataset.from_tensor_slices((
        np.ones([num_batches, feature_dim], dtype=np.float32),
        np.zeros([num_batches, 1], dtype=np.float32),
    )).batch(1)

    def build_keras_model():
      return keras.Sequential([
          keras.layers.InputLayer(shape=(feature_dim,)),
          keras.layers.Dropout(dropout_rate, seed=dropout_seed),
          keras.layers.Dense(
              1, kernel_initializer='ones', bias_initializer='zeros'
          ),
      ])

    functional_model = functional.functional_model_from_keras(
        build_keras_model(),
        loss_fn=keras.losses.MeanSquaredError(),
        input_spec=dataset.element_spec,
    )

    @tensorflow_computation.tf_computation
    def client_update_functional(model_weights, data):
      model_delta_fn = (
          model_delta_client_work.build_functional_model_delta_update(
              model=functional_model,
              weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
              loop_implementation=loop_implementation,
          )
      )
      return model_delta_fn(
          sgdm.build_sgdm(learning_rate=learning_rate), model_weights, data
      )

    client_update_variable = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=lambda: functional.model_from_functional(functional_model),
            weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            loop_implementation=loop_implementation,
        )
    )
    optimizer = sgdm.build_sgdm(learning_rate=learning_rate)
    variable_weights = model_weights_lib.ModelWeights.from_model(
        functional.model_from_functional(functional_model)
    )
    # Pass `ModelWeights` directly to verify that
    # `build_functional_model_delta_update` unpacks `ModelWeights` containers
    # under both `DATASET_ITERATOR` and `DATASET_REDUCE`.
    functional_output, _ = client_update_functional(
        model_weights_lib.ModelWeights(*functional_model.initial_weights),
        dataset,
    )
    variable_output, _ = client_update_variable(
        optimizer, variable_weights, dataset
    )
    self.assertAllClose(functional_output.update, variable_output.update)

    # Verify the 3-batch delta differs from a model whose forward pass leaves
    # the non-trainable seed frozen at `initial_weights[1]` on every batch.
    frozen_model = functional.FunctionalModel(
        initial_weights=functional_model.initial_weights,
        predict_on_batch_fn=functional_model.predict_on_batch,
        loss_fn=functional_model.loss,
        input_spec=functional_model.input_spec,
    )

    @tensorflow_computation.tf_computation
    def client_update_frozen(model_weights, data):
      model_delta_fn = (
          model_delta_client_work.build_functional_model_delta_update(
              model=frozen_model,
              weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
              loop_implementation=loop_implementation,
          )
      )
      return model_delta_fn(
          sgdm.build_sgdm(learning_rate=learning_rate), model_weights, data
      )

    frozen_output, _ = client_update_frozen(
        model_weights_lib.ModelWeights(*frozen_model.initial_weights),
        dataset,
    )
    self.assertNotAllClose(functional_output.update, frozen_output.update)


_TENSOR_DATASET_BATCH_SIZE = 3
_TENSOR_DATASET_FEATURE_DIM = 2
_TENSOR_DATASET_LEARNING_RATE = 0.1
# Substrings (lower case) of op types that must not be in the graphs of a
# computation that uses tensor client datasets.
_FORBIDDEN_OP_TYPE_SUBSTRINGS = ('dataset', 'iterator', 'variant')
_WHILE_OP_TYPES = frozenset(['While', 'StatelessWhile'])


def _create_examples(num_examples: int) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed=num_examples)
  x = rng.uniform(size=[num_examples, _TENSOR_DATASET_FEATURE_DIM])
  y = rng.uniform(size=[num_examples, 1])
  return x.astype(np.float32), y.astype(np.float32)


def _create_batched_dataset(x: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(x=x, y=y)
  ).batch(_TENSOR_DATASET_BATCH_SIZE, drop_remainder=False)


def _create_tensor_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_batches: int | None = None,
    pad_value: float = 0.0,
) -> collections.OrderedDict[str, object]:
  """Returns the tensor client dataset of `_create_batched_dataset(x, y)`.

  Args:
    x: The features, with the shape `[num_examples, feature_dim]`.
    y: The labels, with the shape `[num_examples, 1]`.
    num_batches: The size of the leading (batch index) axis. If `None`, this is
      the number of valid batches. Must not be less than the number of valid
      batches.
    pad_value: The value of the padding examples.
  """
  num_examples = x.shape[0]
  num_valid_batches = -(-num_examples // _TENSOR_DATASET_BATCH_SIZE)
  if num_batches is None:
    num_batches = num_valid_batches
  if num_batches < num_valid_batches:
    raise ValueError(f'{num_batches=} is less than {num_valid_batches=}.')
  num_padded_examples = num_batches * _TENSOR_DATASET_BATCH_SIZE

  def pad_and_reshape(array):
    padded = np.full(
        [num_padded_examples, *array.shape[1:]], pad_value, dtype=array.dtype
    )
    padded[:num_examples] = array
    return padded.reshape(
        [num_batches, _TENSOR_DATASET_BATCH_SIZE, *array.shape[1:]]
    )

  valid_example_mask = np.arange(num_padded_examples) < num_examples
  return collections.OrderedDict([
      (
          model_delta_client_work.TENSOR_DATASET_DATA_KEY,
          collections.OrderedDict(x=pad_and_reshape(x), y=pad_and_reshape(y)),
      ),
      (
          model_delta_client_work.TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY,
          valid_example_mask.reshape([num_batches, _TENSOR_DATASET_BATCH_SIZE]),
      ),
      (
          model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY,
          np.int32(num_valid_batches),
      ),
  ])


def _tensor_dataset_input_spec() -> collections.OrderedDict[str, tf.TensorSpec]:
  return collections.OrderedDict(
      x=tf.TensorSpec([None, _TENSOR_DATASET_FEATURE_DIM], tf.float32),
      y=tf.TensorSpec([None, 1], tf.float32),
  )


def _create_tensor_dataset_functional_model() -> functional.FunctionalModel:

  def build_metrics_fn():
    return collections.OrderedDict(
        loss=keras.metrics.MeanSquaredError(),
        num_examples=counters.NumExamplesCounter(),
        num_batches=counters.NumBatchesCounter(),
    )

  return functional.functional_model_from_keras(
      model_examples.build_linear_regression_keras_functional_model(
          feature_dims=_TENSOR_DATASET_FEATURE_DIM
      ),
      loss_fn=keras.losses.MeanSquaredError(),
      input_spec=_tensor_dataset_input_spec(),
      metrics_constructor=build_metrics_fn,
  )


def _get_tensorflow_op_types(
    comp: federated_language.framework.ConcreteComputation,
) -> set[str]:
  """Returns the op types of all TensorFlow graphs in `comp`.

  The op types include the ops in the function libraries of the graphs.

  Args:
    comp: The computation to inspect.
  """
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


# Parameters `(name, num_examples, num_batches)` for the tensor client dataset
# tests. If `num_batches` is `None`, the tensor client dataset has no padding
# batches.
_TENSOR_DATASET_CASES = (
    ('full_batches', 6, None),
    ('partial_last_batch', 7, None),
    ('padding_batches', 7, 5),
    ('single_partial_batch', 2, None),
    ('zero_examples_no_batches', 0, None),
    ('zero_examples_padding_batches', 0, 2),
)
# The subset of `_TENSOR_DATASET_CASES` that the TFF runtime can pass as a
# `tf.data.Dataset` argument. The runtime does not accept a dataset with a
# partial batch or with no elements.
_RUNTIME_DATASET_CASES = (
    ('full_batches', 6, None),
    ('full_batches_and_padding_batches', 6, 4),
    ('single_full_batch', 3, None),
)
_WEIGHTING_CASES = (
    ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
    ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
)


class TensorDatasetTypeTest(tf.test.TestCase):

  def test_type_from_dict_input_spec_with_unknown_batch_size(self):
    actual_type = model_delta_client_work.tensor_dataset_type_from_input_spec(
        _tensor_dataset_input_spec()
    )

    expected_type = federated_language.StructType([
        (
            model_delta_client_work.TENSOR_DATASET_DATA_KEY,
            federated_language.StructType([
                (
                    'x',
                    federated_language.TensorType(
                        np.float32, [None, None, _TENSOR_DATASET_FEATURE_DIM]
                    ),
                ),
                (
                    'y',
                    federated_language.TensorType(np.float32, [None, None, 1]),
                ),
            ]),
        ),
        (
            model_delta_client_work.TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY,
            federated_language.TensorType(np.bool_, [None, None]),
        ),
        (
            model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY,
            federated_language.TensorType(np.int32),
        ),
    ])
    self.assertTrue(
        actual_type.is_equivalent_to(expected_type),
        msg=f'{actual_type} is not equivalent to {expected_type}',
    )

  def test_type_from_tuple_input_spec_with_fixed_batch_size(self):
    input_spec = (
        tf.TensorSpec([_TENSOR_DATASET_BATCH_SIZE, 4], tf.float32),
        tf.TensorSpec([_TENSOR_DATASET_BATCH_SIZE], tf.int64),
    )

    actual_type = model_delta_client_work.tensor_dataset_type_from_input_spec(
        input_spec
    )

    expected_type = federated_language.StructType([
        (
            model_delta_client_work.TENSOR_DATASET_DATA_KEY,
            federated_language.StructType([
                (
                    None,
                    federated_language.TensorType(
                        np.float32, [None, _TENSOR_DATASET_BATCH_SIZE, 4]
                    ),
                ),
                (
                    None,
                    federated_language.TensorType(
                        np.int64, [None, _TENSOR_DATASET_BATCH_SIZE]
                    ),
                ),
            ]),
        ),
        (
            model_delta_client_work.TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY,
            federated_language.TensorType(np.bool_, [None, None]),
        ),
        (
            model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY,
            federated_language.TensorType(np.int32),
        ),
    ])
    self.assertTrue(
        actual_type.is_equivalent_to(expected_type),
        msg=f'{actual_type} is not equivalent to {expected_type}',
    )

  def test_type_is_not_equivalent_to_sequence_type(self):
    input_spec = _tensor_dataset_input_spec()

    actual_type = model_delta_client_work.tensor_dataset_type_from_input_spec(
        input_spec
    )

    sequence_type = federated_language.SequenceType(
        tensorflow_types.to_type(input_spec)
    )
    self.assertFalse(actual_type.is_equivalent_to(sequence_type))

  def test_type_is_client_placed_tensor_dataset(self):
    actual_type = model_delta_client_work.tensor_dataset_type_from_input_spec(
        _tensor_dataset_input_spec()
    )

    self.assertTrue(
        type_checks.is_client_placed_tensor_dataset(
            federated_language.FederatedType(
                actual_type, federated_language.CLIENTS
            )
        )
    )

  def test_keys_match_type_checks_keys(self):
    self.assertEqual(
        model_delta_client_work.TENSOR_DATASET_DATA_KEY,
        type_checks.TENSOR_DATASET_DATA_KEY,
    )
    self.assertEqual(
        model_delta_client_work.TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY,
        type_checks.TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY,
    )
    self.assertEqual(
        model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY,
        type_checks.TENSOR_DATASET_NUM_VALID_BATCHES_KEY,
    )


class ReduceTensorDatasetTest(tf.test.TestCase, parameterized.TestCase):

  def _reduce_and_count(self, tensor_dataset):
    """Returns `(num_batches, num_examples, x_sum)` of the reduced batches."""

    def reduce_fn(state, batch):
      num_batches, num_examples, x_sum = state
      return (
          num_batches + 1,
          num_examples + tf.shape(batch['x'])[0],
          x_sum + tf.reduce_sum(batch['x'], axis=0),
      )

    initial_state = (
        tf.constant(0, tf.int32),
        tf.constant(0, tf.int32),
        tf.zeros([_TENSOR_DATASET_FEATURE_DIM], tf.float32),
    )

    @tf.function
    def reduce(tensor_dataset):
      return model_delta_client_work.reduce_tensor_dataset(
          reduce_fn, tensor_dataset, initial_state
      )

    return self.evaluate(reduce(tensor_dataset))

  @parameterized.named_parameters(
      (name, num_examples, num_batches)
      for name, num_examples, num_batches in _TENSOR_DATASET_CASES
  )
  def test_reduce_skips_padding(self, num_examples, num_batches):
    x, y = _create_examples(num_examples)
    # The padding is NaN, so the sum is NaN if the reduction reads padding.
    tensor_dataset = _create_tensor_dataset(
        x, y, num_batches=num_batches, pad_value=np.nan
    )

    num_reduced_batches, num_reduced_examples, x_sum = self._reduce_and_count(
        tensor_dataset
    )

    expected_num_batches = len(list(_create_batched_dataset(x, y)))
    self.assertEqual(num_reduced_batches, expected_num_batches)
    self.assertEqual(num_reduced_examples, num_examples)
    self.assertAllClose(x_sum, np.sum(x, axis=0))

  def test_reduce_gives_same_batches_as_dataset(self):
    x, y = _create_examples(7)
    tensor_dataset = _create_tensor_dataset(x, y, num_batches=4)
    batches = []

    def reduce_fn(state, batch):
      batches.append(batch)
      return state

    # In eager mode, `tf.while_loop` calls the body once for each iteration.
    model_delta_client_work.reduce_tensor_dataset(
        reduce_fn, tensor_dataset, tf.constant(0)
    )

    expected_batches = list(_create_batched_dataset(x, y))
    self.assertLen(batches, len(expected_batches))
    for batch, expected_batch in zip(batches, expected_batches):
      self.assertAllClose(batch, expected_batch)

  def test_reduce_stops_at_num_valid_batches(self):
    x, y = _create_examples(6)
    tensor_dataset = _create_tensor_dataset(x, y)
    tensor_dataset[
        model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY
    ] = np.int32(1)

    num_reduced_batches, num_reduced_examples, x_sum = self._reduce_and_count(
        tensor_dataset
    )

    self.assertEqual(num_reduced_batches, 1)
    self.assertEqual(num_reduced_examples, _TENSOR_DATASET_BATCH_SIZE)
    self.assertAllClose(x_sum, np.sum(x[:_TENSOR_DATASET_BATCH_SIZE], axis=0))

  def test_reduce_does_not_read_past_last_batch(self):
    x, y = _create_examples(6)
    tensor_dataset = _create_tensor_dataset(x, y)
    tensor_dataset[
        model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY
    ] = np.int32(10)

    num_reduced_batches, num_reduced_examples, x_sum = self._reduce_and_count(
        tensor_dataset
    )

    self.assertEqual(num_reduced_batches, 2)
    self.assertEqual(num_reduced_examples, 6)
    self.assertAllClose(x_sum, np.sum(x, axis=0))

  def test_reduce_skips_all_masked_batches(self):
    x, y = _create_examples(6)
    # Two valid batches, then two padding batches. `num_valid_batches` includes
    # the padding batches, so the loop must skip them with the mask.
    tensor_dataset = _create_tensor_dataset(
        x, y, num_batches=4, pad_value=np.nan
    )
    tensor_dataset[
        model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY
    ] = np.int32(4)

    num_reduced_batches, num_reduced_examples, x_sum = self._reduce_and_count(
        tensor_dataset
    )

    self.assertEqual(num_reduced_batches, 2)
    self.assertEqual(num_reduced_examples, 6)
    self.assertAllClose(x_sum, np.sum(x, axis=0))


class TensorFunctionalModelDeltaClientWorkComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(*_WEIGHTING_CASES)
  def test_next_has_expected_client_data_type(self, weighting):
    model = _create_tensor_dataset_functional_model()

    process = (
        model_delta_client_work.build_tensor_functional_model_delta_client_work(
            model=model,
            optimizer=sgdm.build_sgdm(learning_rate=1.0),
            client_weighting=weighting,
        )
    )

    client_data_type = process.next.type_signature.parameter['client_data']
    self.assertIsInstance(client_data_type, federated_language.FederatedType)
    self.assertEqual(client_data_type.placement, federated_language.CLIENTS)
    member_type = client_data_type.member
    data_type = member_type[model_delta_client_work.TENSOR_DATASET_DATA_KEY]
    self.assertEqual(data_type['x'].dtype, np.float32)
    self.assertEqual(
        data_type['x'].shape, (None, None, _TENSOR_DATASET_FEATURE_DIM)
    )
    self.assertEqual(data_type['y'].dtype, np.float32)
    self.assertEqual(data_type['y'].shape, (None, None, 1))
    mask_type = member_type[
        model_delta_client_work.TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY
    ]
    self.assertEqual(mask_type.dtype, np.bool_)
    self.assertEqual(mask_type.shape, (None, None))
    num_valid_batches_type = member_type[
        model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY
    ]
    self.assertEqual(num_valid_batches_type.dtype, np.int32)
    self.assertEqual(num_valid_batches_type.shape, ())
    self.assertTrue(
        member_type.is_equivalent_to(
            model_delta_client_work.tensor_dataset_type_from_input_spec(
                model.input_spec
            )
        )
    )

  @parameterized.named_parameters(*_WEIGHTING_CASES)
  def test_type_signatures_match_dataset_client_work(self, weighting):
    model = _create_tensor_dataset_functional_model()
    optimizer = sgdm.build_sgdm(learning_rate=1.0)

    tensor_process = (
        model_delta_client_work.build_tensor_functional_model_delta_client_work(
            model=model, optimizer=optimizer, client_weighting=weighting
        )
    )
    dataset_process = (
        model_delta_client_work.build_functional_model_delta_client_work(
            model=model, optimizer=optimizer, client_weighting=weighting
        )
    )

    # Only the type of the client data is different.
    self.assertTrue(
        tensor_process.initialize.type_signature.is_equivalent_to(
            dataset_process.initialize.type_signature
        )
    )
    self.assertTrue(
        tensor_process.next.type_signature.result.is_equivalent_to(
            dataset_process.next.type_signature.result
        )
    )
    tensor_parameter = tensor_process.next.type_signature.parameter
    dataset_parameter = dataset_process.next.type_signature.parameter
    for name in ('state', 'weights'):
      self.assertTrue(
          tensor_parameter[name].is_equivalent_to(dataset_parameter[name])
      )
    self.assertFalse(
        tensor_parameter['client_data'].is_equivalent_to(
            dataset_parameter['client_data']
        )
    )

  def test_raises_with_non_functional_model(self):
    with self.assertRaises(TypeError):
      model_delta_client_work.build_tensor_functional_model_delta_client_work(
          model=create_model,
          optimizer=sgdm.build_sgdm(learning_rate=1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      )

  def test_graph_has_while_loop_and_no_dataset_ops(self):
    process = (
        model_delta_client_work.build_tensor_functional_model_delta_client_work(
            model=_create_tensor_dataset_functional_model(),
            optimizer=sgdm.build_sgdm(learning_rate=1.0),
            client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        )
    )

    op_types = _get_tensorflow_op_types(process.next)

    self.assertEmpty(_get_forbidden_op_types(op_types))
    self.assertNotEmpty(op_types & _WHILE_OP_TYPES)

  def test_dataset_client_work_graph_has_dataset_ops(self):
    # Make sure that `_get_tensorflow_op_types` finds the ops of the client
    # computation. Thus the test above can detect dataset ops.
    process = model_delta_client_work.build_functional_model_delta_client_work(
        model=_create_tensor_dataset_functional_model(),
        optimizer=sgdm.build_sgdm(learning_rate=1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )

    op_types = _get_tensorflow_op_types(process.next)

    self.assertNotEmpty(_get_forbidden_op_types(op_types))


class TensorFunctionalModelDeltaClientWorkExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):

  def _build_in_graph_dataset_client_update(self, model, weighting):
    """Returns the reference client update for the examples `(x, y)`.

    The TFF runtime cannot accept a `tf.data.Dataset` argument that has a
    partial batch or no elements. Thus the reference computation accepts the
    examples and makes the batched `tf.data.Dataset` in the graph.

    Args:
      model: A `tff.learning.models.FunctionalModel`.
      weighting: A `tff.learning.ClientWeighting` value.
    """

    # Note: we must wrap in a `tf_computation` for the correct graph-context
    # processing of Keras models wrapped as FunctionalModel.
    @tensorflow_computation.tf_computation
    def dataset_client_update(model_weights, x, y):
      dataset = tf.data.Dataset.from_tensor_slices(
          collections.OrderedDict(x=x, y=y)
      ).batch(_TENSOR_DATASET_BATCH_SIZE, drop_remainder=False)
      client_update = model_delta_client_work.build_functional_model_delta_update(
          model=model,
          weighting=weighting,
          loop_implementation=loop_builder.LoopImplementation.DATASET_REDUCE,
      )
      return client_update(
          sgdm.build_sgdm(learning_rate=_TENSOR_DATASET_LEARNING_RATE),
          model_weights,
          dataset,
      )

    return dataset_client_update

  def _build_tensor_client_update(self, model, weighting):

    @tensorflow_computation.tf_computation
    def tensor_client_update(model_weights, tensor_dataset):
      client_update = (
          model_delta_client_work.build_tensor_functional_model_delta_update(
              model=model, weighting=weighting
          )
      )
      return client_update(
          sgdm.build_sgdm(learning_rate=_TENSOR_DATASET_LEARNING_RATE),
          model_weights,
          tensor_dataset,
      )

    return tensor_client_update

  def _assert_update_is_not_trivial(self, client_result, num_examples):
    update_norm = np.linalg.norm(
        np.concatenate([np.ravel(u) for u in client_result.update])
    )
    if num_examples > 0:
      self.assertGreater(update_norm, 0.0)
    else:
      self.assertEqual(update_norm, 0.0)

  def _assert_expected_update_weight(
      self, client_result, weighting, num_examples
  ):
    if weighting == client_weight_lib.ClientWeighting.NUM_EXAMPLES:
      self.assertAllClose(client_result.update_weight, float(num_examples))
    else:
      self.assertAllClose(client_result.update_weight, 1.0)

  @parameterized.named_parameters(
      (f'{weighting_name}_{case_name}', weighting, num_examples, num_batches)
      for (weighting_name, weighting), (
          case_name,
          num_examples,
          num_batches,
      ) in itertools.product(_WEIGHTING_CASES, _RUNTIME_DATASET_CASES)
  )
  def test_client_work_matches_dataset_client_work(
      self, weighting, num_examples, num_batches
  ):
    model = _create_tensor_dataset_functional_model()
    x, y = _create_examples(num_examples)
    tensor_process = (
        model_delta_client_work.build_tensor_functional_model_delta_client_work(
            model=model,
            optimizer=sgdm.build_sgdm(
                learning_rate=_TENSOR_DATASET_LEARNING_RATE
            ),
            client_weighting=weighting,
        )
    )
    dataset_process = (
        model_delta_client_work.build_functional_model_delta_client_work(
            model=model,
            optimizer=sgdm.build_sgdm(
                learning_rate=_TENSOR_DATASET_LEARNING_RATE
            ),
            client_weighting=weighting,
        )
    )
    client_model_weights = [model.initial_weights]

    tensor_output = tensor_process.next(
        tensor_process.initialize(),
        client_model_weights,
        [_create_tensor_dataset(x, y, num_batches=num_batches)],
    )
    dataset_output = dataset_process.next(
        dataset_process.initialize(),
        client_model_weights,
        [_create_batched_dataset(x, y)],
    )

    tensor_result = tensor_output.result[0]
    dataset_result = dataset_output.result[0]
    self.assertAllClose(tensor_result.update, dataset_result.update)
    self.assertAllClose(
        tensor_result.update_weight, dataset_result.update_weight
    )
    self.assertAllClose(tensor_output.measurements, dataset_output.measurements)
    self._assert_expected_update_weight(tensor_result, weighting, num_examples)
    self._assert_update_is_not_trivial(tensor_result, num_examples)
    train_metrics = tensor_output.measurements['train']
    self.assertEqual(train_metrics['num_examples'], num_examples)
    self.assertEqual(
        train_metrics['num_batches'],
        -(-num_examples // _TENSOR_DATASET_BATCH_SIZE),
    )
    self.assertEqual(train_metrics['num_non_finite_clients'], 0)

  @parameterized.named_parameters(
      (
          f'{weighting_name}_{case_name}_{pad_name}',
          weighting,
          num_examples,
          num_batches,
          pad_value,
      )
      for (weighting_name, weighting), (
          case_name,
          num_examples,
          num_batches,
      ), (pad_name, pad_value) in itertools.product(
          _WEIGHTING_CASES,
          _TENSOR_DATASET_CASES,
          (('zero_padding', 0.0), ('nan_padding', np.nan)),
      )
  )
  def test_client_update_matches_dataset_client_update(
      self, weighting, num_examples, num_batches, pad_value
  ):
    model = _create_tensor_dataset_functional_model()
    x, y = _create_examples(num_examples)
    tensor_client_update = self._build_tensor_client_update(model, weighting)
    dataset_client_update = self._build_in_graph_dataset_client_update(
        model, weighting
    )

    # If the padding is NaN and the client update reads the padding, the update
    # is not finite, and the update weight is zero.
    tensor_result, tensor_metrics = tensor_client_update(
        model.initial_weights,
        _create_tensor_dataset(
            x, y, num_batches=num_batches, pad_value=pad_value
        ),
    )
    dataset_result, dataset_metrics = dataset_client_update(
        model.initial_weights, x, y
    )

    self.assertAllClose(tensor_result.update, dataset_result.update)
    self.assertAllClose(
        tensor_result.update_weight, dataset_result.update_weight
    )
    self.assertAllClose(tensor_metrics, dataset_metrics)
    self._assert_expected_update_weight(tensor_result, weighting, num_examples)
    self._assert_update_is_not_trivial(tensor_result, num_examples)
    self.assertEqual(
        tensor_metrics[model_delta_client_work.NUM_NON_FINITE_CLIENTS_KEY], 0
    )
    self.assertEqual(np.sum(tensor_metrics['num_examples']), num_examples)
    self.assertEqual(
        np.sum(tensor_metrics['num_batches']),
        -(-num_examples // _TENSOR_DATASET_BATCH_SIZE),
    )

  @parameterized.named_parameters(
      (f'{weighting_name}_{case_name}', weighting, num_examples, num_batches)
      for (weighting_name, weighting), (
          case_name,
          num_examples,
          num_batches,
      ) in itertools.product(_WEIGHTING_CASES, _TENSOR_DATASET_CASES)
  )
  def test_client_work_matches_dataset_client_update(
      self, weighting, num_examples, num_batches
  ):
    model = _create_tensor_dataset_functional_model()
    x, y = _create_examples(num_examples)
    tensor_process = (
        model_delta_client_work.build_tensor_functional_model_delta_client_work(
            model=model,
            optimizer=sgdm.build_sgdm(
                learning_rate=_TENSOR_DATASET_LEARNING_RATE
            ),
            client_weighting=weighting,
        )
    )
    dataset_client_update = self._build_in_graph_dataset_client_update(
        model, weighting
    )

    tensor_output = tensor_process.next(
        tensor_process.initialize(),
        [model.initial_weights],
        [_create_tensor_dataset(x, y, num_batches=num_batches)],
    )
    dataset_result, dataset_metrics = dataset_client_update(
        model.initial_weights, x, y
    )

    tensor_result = tensor_output.result[0]
    self.assertAllClose(tensor_result.update, dataset_result.update)
    self.assertAllClose(
        tensor_result.update_weight, dataset_result.update_weight
    )
    # With one client, the summed metrics are the metrics of the client. The
    # loss is a mean: the first element of the state is the sum of the losses,
    # and the second element is the number of examples.
    loss_sum, loss_count = dataset_metrics['loss']
    expected_loss = loss_sum / loss_count if loss_count > 0 else 0.0
    train_metrics = tensor_output.measurements['train']
    self.assertAllClose(train_metrics['loss'], expected_loss)
    self.assertEqual(
        train_metrics['num_examples'], np.sum(dataset_metrics['num_examples'])
    )
    self.assertEqual(
        train_metrics['num_batches'], np.sum(dataset_metrics['num_batches'])
    )
    self.assertEqual(train_metrics['num_non_finite_clients'], 0)

  def test_all_masked_and_non_prefix_masked_batches_match_dataset(self):
    num_examples = 7
    model = _create_tensor_dataset_functional_model()
    x, y = _create_examples(num_examples)
    # The rows of the tensor client dataset are:
    #   0: examples 0, 1, 2.
    #   1: only padding. The loop must skip this batch.
    #   2: examples 3, 4, 5.
    #   3: padding, example 6, padding. The mask is not a prefix.
    # Thus the valid batches are the same as `_create_batched_dataset(x, y)`.
    # The padding is NaN, so the update is not finite if the loop reads it.
    num_batches = 4
    data_x = np.full(
        [num_batches, _TENSOR_DATASET_BATCH_SIZE, _TENSOR_DATASET_FEATURE_DIM],
        np.nan,
        np.float32,
    )
    data_y = np.full(
        [num_batches, _TENSOR_DATASET_BATCH_SIZE, 1], np.nan, np.float32
    )
    mask = np.zeros([num_batches, _TENSOR_DATASET_BATCH_SIZE], np.bool_)
    data_x[0], data_y[0], mask[0] = x[0:3], y[0:3], True
    data_x[2], data_y[2], mask[2] = x[3:6], y[3:6], True
    data_x[3, 1], data_y[3, 1], mask[3, 1] = x[6], y[6], True
    tensor_dataset = collections.OrderedDict([
        (
            model_delta_client_work.TENSOR_DATASET_DATA_KEY,
            collections.OrderedDict(x=data_x, y=data_y),
        ),
        (model_delta_client_work.TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY, mask),
        (
            model_delta_client_work.TENSOR_DATASET_NUM_VALID_BATCHES_KEY,
            np.int32(num_batches),
        ),
    ])
    weighting = client_weight_lib.ClientWeighting.NUM_EXAMPLES
    tensor_client_update = self._build_tensor_client_update(model, weighting)
    dataset_client_update = self._build_in_graph_dataset_client_update(
        model, weighting
    )

    tensor_result, tensor_metrics = tensor_client_update(
        model.initial_weights, tensor_dataset
    )
    dataset_result, dataset_metrics = dataset_client_update(
        model.initial_weights, x, y
    )

    self.assertAllClose(tensor_result.update, dataset_result.update)
    self.assertAllClose(
        tensor_result.update_weight, dataset_result.update_weight
    )
    self.assertAllClose(tensor_metrics, dataset_metrics)
    self.assertAllClose(tensor_result.update_weight, float(num_examples))
    self.assertEqual(
        tensor_metrics[model_delta_client_work.NUM_NON_FINITE_CLIENTS_KEY], 0
    )
    # The skipped batch does not count.
    self.assertEqual(np.sum(tensor_metrics['num_batches']), 3)

  def test_metrics_output_with_non_finite_updates(self):
    model = _create_tensor_dataset_functional_model()
    process = (
        model_delta_client_work.build_tensor_functional_model_delta_client_work(
            model=model,
            optimizer=sgdm.build_sgdm(learning_rate=1.0),
            client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        )
    )
    trainable_weights, non_trainable_weights = model.initial_weights
    bad_trainable_weights = tuple(
        np.full_like(w, np.nan) for w in trainable_weights
    )
    x, y = _create_examples(7)

    output = process.next(
        process.initialize(),
        [(bad_trainable_weights, non_trainable_weights)],
        [_create_tensor_dataset(x, y)],
    )

    self.assertEqual(output.result[0].update_weight, 0.0)
    self.assertEqual(output.measurements['train']['num_examples'], 0)
    self.assertEqual(output.measurements['train']['num_non_finite_clients'], 1)


if __name__ == '__main__':
  cpp_execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
