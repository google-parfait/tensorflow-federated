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

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.flars import flars_fedavg
from tensorflow_federated.python.research.flars import flars_optimizer


def _create_input_spec():
  return collections.OrderedDict(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int64, shape=[None, 1]))


def _keras_model_fn():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  input_spec = _create_input_spec()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


def create_client_data():
  emnist_batch = collections.OrderedDict(
      label=[5],
      pixels=tf.random.stateless_uniform(shape=(28, 28), seed=(7, 42)))

  output_types = collections.OrderedDict(label=tf.int64, pixels=tf.float32)
  output_shapes = collections.OrderedDict(
      label=tf.TensorShape([1]), pixels=tf.TensorShape([28, 28]))

  dataset = tf.data.Dataset.from_generator(lambda: (yield emnist_batch),
                                           output_types, output_shapes)

  def client_data():
    return tff.simulation.models.mnist.keras_dataset_from_emnist(
        dataset).repeat(2).batch(2)

  return client_data


class FlarsFedAvgTest(tf.test.TestCase):

  def test_construction(self):
    it_process = flars_fedavg.build_federated_averaging_process(
        _keras_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type), '{<x=float32[?,784],y=int64[?,1]>*}@CLIENTS')

  def test_simple_training(self):
    it_process = flars_fedavg.build_federated_averaging_process(
        _keras_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
    server_state = it_process.initialize()

    # Test out manually setting weights:
    keras_model = tff.simulation.models.mnist.create_keras_model(
        compile_model=True)

    @tf.function
    def deterministic_batch():
      return collections.OrderedDict(
          x=tf.ones([1, 784], dtype=tf.float32),
          y=tf.ones([1, 1], dtype=tf.int64))

    batch = deterministic_batch()
    federated_data = [[batch]]

    def keras_evaluate(state):
      state.model.assign_weights_to(keras_model)
      # N.B. The loss computed here won't match the loss computed by TFF because
      # of the Dropout layer.
      keras_model.test_on_batch(**batch)

    loss_list = []
    for _ in range(3):
      keras_evaluate(server_state)
      server_state, output = it_process.next(server_state, federated_data)
      loss_list.append(output['loss'])
    keras_evaluate(server_state)

    self.assertLess(tf.reduce_mean(loss_list[1:]), loss_list[0])

  def test_self_contained_example_keras_model(self):
    client_data = create_client_data()
    train_data = [client_data()]
    trainer = flars_fedavg.build_federated_averaging_process(
        _keras_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
    state = trainer.initialize()
    losses = []
    for _ in range(2):
      state, outputs = trainer.next(state, train_data)
      # Track the loss.
      losses.append(outputs['loss'])
    self.assertLess(losses[1], losses[0])


def server_init(model, optimizer):
  """Returns initial `tff.learning.framework.ServerState`.

  Args:
    model: A `tff.learning.Model`.
    optimizer: A `tf.keras.optimizer.Optimizer`.

  Returns:
    A `tff.learning.framework.ServerState` namedtuple.
  """
  optimizer_vars = flars_fedavg._create_optimizer_vars(model, optimizer)
  return (flars_fedavg.ServerState(
      model=tff.learning.framework.ModelWeights.from_model(model),
      optimizer_state=optimizer_vars), optimizer_vars)


class ServerTest(tf.test.TestCase):

  def test_nan_examples_ignored(self):
    server_optimizer_fn = lambda: flars_optimizer.FLARSOptimizer(1.0)
    model = _keras_model_fn()
    server_optimizer = server_optimizer_fn()
    state, optimizer_vars = server_init(model, server_optimizer)

    grad_norm = [1.0, 1.0]
    weights_delta = tf.nest.map_structure(
        lambda t: tf.ones_like(t) * float('inf'),
        flars_fedavg.tff.learning.framework.ModelWeights.from_model(
            model).trainable)

    old_model_vars = state.model
    for _ in range(2):
      state = flars_fedavg.server_update(model, server_optimizer,
                                         optimizer_vars, state, weights_delta,
                                         grad_norm)
    model_vars = state.model
    # Assert the model hasn't changed.
    self.assertAllClose(old_model_vars.trainable, model_vars.trainable)
    self.assertAllClose(old_model_vars.non_trainable, model_vars.non_trainable)


class ClientTest(tf.test.TestCase):

  def test_self_contained_example(self):
    client_data = create_client_data()
    model = _keras_model_fn()
    outputs = flars_fedavg.client_update(
        model, tf.keras.optimizers.SGD(0.1), client_data(),
        flars_fedavg.tff.learning.framework.ModelWeights.from_model(model))

    self.assertAllEqual(outputs.weights_delta_weight, 2)
    # Expect a grad for each layer:
    #   [Conv, Pool, Conv, Pool, Dense + Bias, Dense + Bias] = 8
    self.assertLen(outputs.optimizer_output['flat_grads_norm_sum'], 8)
    self.assertEqual(outputs.optimizer_output['num_examples'], 2)


if __name__ == '__main__':
  tf.test.main()
