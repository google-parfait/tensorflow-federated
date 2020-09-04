# Copyright 2019, The TensorFlow Federated Authors.
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
"""End-to-end example testing Federated Averaging against the MNIST model."""

import collections

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shared import fed_avg_schedule

_Batch = collections.namedtuple('Batch', ['x', 'y'])


def _batch_fn(has_nan=False):
  batch = _Batch(
      x=np.ones([1, 784], dtype=np.float32), y=np.ones([1, 1], dtype=np.int64))
  if has_nan:
    batch[0][0, 0] = np.nan
  return batch


def _create_input_spec():
  return _Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int64, shape=[None, 1]))


def _uncompiled_model_builder():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=_create_input_spec(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


class ModelDeltaProcessTest(tf.test.TestCase):

  def _run_rounds(self, iterproc, federated_data, num_rounds):
    train_outputs = []
    initial_state = iterproc.initialize()
    state = initial_state
    for round_num in range(num_rounds):
      state, metrics = iterproc.next(state, federated_data)
      train_outputs.append(metrics)
      logging.info('Round %d: %s', round_num, metrics)
    return state, train_outputs, initial_state

  def test_fed_avg_without_schedule_decreases_loss(self):
    federated_data = [[_batch_fn()]]

    iterproc = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 5)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_fed_avg_with_custom_client_weight_fn(self):
    federated_data = [[_batch_fn()]]

    def client_weight_fn(local_outputs):
      return 1.0/(1.0 + local_outputs['loss'][-1])

    iterproc = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        client_weight_fn=client_weight_fn)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 5)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_client_update_with_finite_delta(self):
    federated_data = [_batch_fn()]
    model = _uncompiled_model_builder()
    client_optimizer = tf.keras.optimizers.SGD(0.1)
    client_update = fed_avg_schedule.create_client_update_fn()
    outputs = client_update(model, federated_data,
                            fed_avg_schedule._get_weights(model),
                            client_optimizer)
    self.assertAllEqual(self.evaluate(outputs.client_weight), 1)
    self.assertAllEqual(
        self.evaluate(outputs.optimizer_output['num_examples']), 1)

  def test_client_update_with_non_finite_delta(self):
    federated_data = [_batch_fn(has_nan=True)]
    model = _uncompiled_model_builder()
    client_optimizer = tf.keras.optimizers.SGD(0.1)
    client_update = fed_avg_schedule.create_client_update_fn()
    outputs = client_update(model, federated_data,
                            fed_avg_schedule._get_weights(model),
                            client_optimizer)
    self.assertAllEqual(self.evaluate(outputs.client_weight), 0)

  def test_server_update_with_nan_data_is_noop(self):
    federated_data = [[_batch_fn(has_nan=True)]]

    iterproc = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    state, _, initial_state = self._run_rounds(iterproc, federated_data, 1)
    self.assertAllClose(state.model.trainable, initial_state.model.trainable,
                        1e-8)
    self.assertAllClose(state.model.non_trainable,
                        initial_state.model.non_trainable, 1e-8)

  def test_server_update_with_inf_weight_is_noop(self):
    federated_data = [[_batch_fn()]]
    client_weight_fn = lambda x: np.inf

    iterproc = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        client_weight_fn=client_weight_fn)

    state, _, initial_state = self._run_rounds(iterproc, federated_data, 1)
    self.assertAllClose(state.model.trainable, initial_state.model.trainable,
                        1e-8)
    self.assertAllClose(state.model.non_trainable,
                        initial_state.model.non_trainable, 1e-8)

  def test_fed_avg_with_client_schedule(self):
    federated_data = [[_batch_fn()]]

    @tf.function
    def lr_schedule(x):
      return 0.1 if x < 1.5 else 0.0

    iterproc = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        client_lr=lr_schedule,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[1]['loss'], train_outputs[0]['loss'])
    self.assertNear(
        train_outputs[2]['loss'], train_outputs[3]['loss'], err=1e-4)

  def test_fed_avg_with_server_schedule(self):
    federated_data = [[_batch_fn()]]

    @tf.function
    def lr_schedule(x):
      return 1.0 if x < 1.5 else 0.0

    iterproc = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        server_lr=lr_schedule)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[1]['loss'], train_outputs[0]['loss'])
    self.assertNear(
        train_outputs[2]['loss'], train_outputs[3]['loss'], err=1e-4)

  def test_fed_avg_with_client_and_server_schedules(self):
    federated_data = [[_batch_fn()]]

    iterproc = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        client_lr=lambda x: 0.1 / (x + 1)**2,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        server_lr=lambda x: 1.0 / (x + 1)**2)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 6)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])
    train_gap_first_half = train_outputs[0]['loss'] - train_outputs[2]['loss']
    train_gap_second_half = train_outputs[3]['loss'] - train_outputs[5]['loss']
    self.assertLess(train_gap_second_half, train_gap_first_half)

  def test_build_with_preprocess_function(self):
    test_dataset = tf.data.Dataset.range(5)
    client_datasets_type = tff.FederatedType(
        tff.SequenceType(test_dataset.element_spec), tff.CLIENTS)

    @tff.tf_computation(tff.SequenceType(test_dataset.element_spec))
    def preprocess_dataset(ds):

      def to_batch(x):
        return _Batch(
            tf.fill(dims=(784,), value=float(x) * 2.0),
            tf.expand_dims(tf.cast(x + 1, dtype=tf.int64), axis=0))

      return ds.map(to_batch).batch(2)

    iterproc = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    iterproc = tff.simulation.compose_dataset_computation_with_iterative_process(
        preprocess_dataset, iterproc)

    with tf.Graph().as_default():
      test_model_for_types = _uncompiled_model_builder()

    server_state_type = tff.FederatedType(
        fed_avg_schedule.ServerState(
            model=tff.framework.type_from_tensors(
                tff.learning.ModelWeights(
                    test_model_for_types.trainable_variables,
                    test_model_for_types.non_trainable_variables)),
            optimizer_state=(tf.int64,),
            round_num=tf.float32), tff.SERVER)
    metrics_type = test_model_for_types.federated_output_computation.type_signature.result

    expected_parameter_type = collections.OrderedDict(
        server_state=server_state_type,
        federated_dataset=client_datasets_type,
    )
    expected_result_type = (server_state_type, metrics_type)

    expected_type = tff.FunctionType(
        parameter=expected_parameter_type, result=expected_result_type)
    self.assertTrue(
        iterproc.next.type_signature.is_equivalent_to(expected_type),
        msg='{s}\n!={t}'.format(
            s=iterproc.next.type_signature, t=expected_type))

  def test_execute_with_preprocess_function(self):
    test_dataset = tf.data.Dataset.range(1)

    @tff.tf_computation(tff.SequenceType(test_dataset.element_spec))
    def preprocess_dataset(ds):

      def to_example(x):
        del x  # Unused.
        return _Batch(
            x=np.ones([784], dtype=np.float32), y=np.ones([1], dtype=np.int64))

      return ds.map(to_example).batch(1)

    iterproc = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    iterproc = tff.simulation.compose_dataset_computation_with_iterative_process(
        preprocess_dataset, iterproc)

    _, train_outputs, _ = self._run_rounds(iterproc, [test_dataset], 6)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])
    train_gap_first_half = train_outputs[0]['loss'] - train_outputs[2]['loss']
    train_gap_second_half = train_outputs[3]['loss'] - train_outputs[5]['loss']
    self.assertLess(train_gap_second_half, train_gap_first_half)


if __name__ == '__main__':
  tf.test.main()
