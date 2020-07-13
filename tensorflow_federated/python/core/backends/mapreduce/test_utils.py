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
"""Utils that support unit tests in this component."""

import collections

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.mapreduce import canonical_form
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.templates import iterative_process


def get_temperature_sensor_example():
  """Constructs `canonical_form.CanonicalForm` for temperature sensors example.

  The temperature sensor example computes the fraction of sensors that report
  temperatures over the threshold.

  Returns:
    An instance of `canonical_form.CanonicalForm`.
  """

  @computations.tf_computation
  def initialize():
    return {'num_rounds': tf.constant(0)}

  # The state of the server is a singleton tuple containing just the integer
  # counter `num_rounds`.
  server_state_type = computation_types.NamedTupleType([
      ('num_rounds', tf.int32),
  ])

  @computations.tf_computation(server_state_type)
  def prepare(state):
    return {'max_temperature': 32.0 + tf.cast(state.num_rounds, tf.float32)}

  # The initial state of the client is a singleton tuple containing a single
  # float `max_temperature`, which is the threshold received from the server.
  client_state_type = computation_types.NamedTupleType([
      ('max_temperature', tf.float32),
  ])

  # The client data is a sequence of floats.
  client_data_type = computation_types.SequenceType(tf.float32)

  @computations.tf_computation(client_data_type, client_state_type)
  def work(data, state):
    """See the `canonical_form.CanonicalForm` definition of `work`."""

    def fn(s, x):
      return {
          'num': s['num'] + 1,
          'max': tf.maximum(s['max'], x),
      }

    reduce_result = data.reduce({
        'num': np.int32(0),
        'max': np.float32(-459.67)
    }, fn)
    client_updates = collections.OrderedDict(
        is_over=reduce_result['max'] > state.max_temperature)
    return client_updates, []

  # The client update is a singleton tuple with a Boolean-typed `is_over`.
  client_update_type = computation_types.NamedTupleType([('is_over', tf.bool)])

  # The accumulator for client updates is a pair of counters, one for the
  # number of clients over threshold, and the other for the total number of
  # client updates processed so far.
  accumulator_type = computation_types.NamedTupleType([('num_total', tf.int32),
                                                       ('num_over', tf.int32)])

  @computations.tf_computation
  def zero():
    return collections.OrderedDict(
        num_total=tf.constant(0), num_over=tf.constant(0))

  @computations.tf_computation(accumulator_type, client_update_type)
  def accumulate(accumulator, update):
    return collections.OrderedDict(
        num_total=accumulator.num_total + 1,
        num_over=accumulator.num_over + tf.cast(update.is_over, tf.int32))

  @computations.tf_computation(accumulator_type, accumulator_type)
  def merge(accumulator1, accumulator2):
    return collections.OrderedDict(
        num_total=accumulator1.num_total + accumulator2.num_total,
        num_over=accumulator1.num_over + accumulator2.num_over)

  @computations.tf_computation(merge.type_signature.result)
  def report(accumulator):
    return collections.OrderedDict(
        ratio_over_threshold=(tf.cast(accumulator['num_over'], tf.float32) /
                              tf.cast(accumulator['num_total'], tf.float32)))

  @computations.tf_computation
  def bitwidth():
    return []

  update_type = computation_types.NamedTupleType([
      computation_types.NamedTupleType([('ratio_over_threshold', tf.float32)]),
      computation_types.NamedTupleType([]),
  ])

  @computations.tf_computation(server_state_type, update_type)
  def update(state, update):
    return ({'num_rounds': state.num_rounds + 1}, update[0])

  return canonical_form.CanonicalForm(initialize, prepare, work, zero,
                                      accumulate, merge, report, bitwidth,
                                      update)


def get_mnist_training_example():
  """Constructs `canonical_form.CanonicalForm` for mnist training.

  Returns:
    An instance of `canonical_form.CanonicalForm`.
  """
  model_nt = collections.namedtuple('Model', 'weights bias')
  server_state_nt = (collections.namedtuple('ServerState', 'model num_rounds'))

  # Start with a model filled with zeros, and the round counter set to zero.
  @computations.tf_computation
  def initialize():
    return server_state_nt(
        model=model_nt(weights=tf.zeros([784, 10]), bias=tf.zeros([10])),
        num_rounds=tf.constant(0))

  server_state_tff_type = server_state_nt(
      model=model_nt(weights=(tf.float32, [784, 10]), bias=(tf.float32, [10])),
      num_rounds=tf.int32)
  client_state_nt = (
      collections.namedtuple('ClientState', 'model learning_rate'))

  # Pass the model to the client, along with a dynamically adjusted learning
  # rate that starts at 0.1 and decays exponentially by a factor of 0.9.
  @computations.tf_computation(server_state_tff_type)
  def prepare(state):
    learning_rate = 0.1 * tf.pow(0.9, tf.cast(state.num_rounds, tf.float32))
    return client_state_nt(model=state.model, learning_rate=learning_rate)

  batch_nt = collections.namedtuple('Batch', 'x y')
  batch_tff_type = batch_nt(x=(tf.float32, [None, 784]), y=(tf.int32, [None]))
  dataset_tff_type = computation_types.SequenceType(batch_tff_type)
  model_tff_type = model_nt(
      weights=(tf.float32, [784, 10]), bias=(tf.float32, [10]))
  client_state_tff_type = client_state_nt(
      model=model_tff_type, learning_rate=tf.float32)
  loop_state_nt = collections.namedtuple('LoopState', 'num_examples total_loss')
  update_nt = collections.namedtuple('Update', 'model num_examples loss')

  # Train the model locally, emit the loclaly-trained model and the number of
  # examples as an update, and the average loss and the number of examples as
  # local client stats.
  @computations.tf_computation(dataset_tff_type, client_state_tff_type)
  def work(data, state):  # pylint: disable=missing-docstring
    model_vars = model_nt(
        weights=tf.Variable(initial_value=state.model.weights, name='weights'),
        bias=tf.Variable(initial_value=state.model.bias, name='bias'))
    init_model = tf.compat.v1.global_variables_initializer()

    optimizer = tf.keras.optimizers.SGD(state.learning_rate)

    @tf.function
    def reduce_fn(loop_state, batch):
      """Compute a single gradient step on an given batch of examples."""
      with tf.GradientTape() as tape:
        pred_y = tf.nn.softmax(
            tf.matmul(batch.x, model_vars.weights) + model_vars.bias)
        loss = -tf.reduce_mean(
            tf.reduce_sum(
                tf.one_hot(batch.y, 10) * tf.math.log(pred_y), axis=[1]))
      grads = tape.gradient(loss, model_vars)
      optimizer.apply_gradients(
          zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars)))
      return loop_state_nt(
          num_examples=loop_state.num_examples + 1,
          total_loss=loop_state.total_loss + loss)

    with tf.control_dependencies([init_model]):
      loop_state = data.reduce(
          loop_state_nt(num_examples=0, total_loss=np.float32(0.0)), reduce_fn)
      num_examples = loop_state.num_examples
      total_loss = loop_state.total_loss
      with tf.control_dependencies([num_examples, total_loss]):
        loss = total_loss / tf.cast(num_examples, tf.float32)

    return update_nt(model=model_vars, num_examples=num_examples, loss=loss), []

  accumulator_nt = update_nt

  # Initialize accumulators for aggregation with zero model and zero examples.
  @computations.tf_computation
  def zero():
    return accumulator_nt(
        model=model_nt(weights=tf.zeros([784, 10]), bias=tf.zeros([10])),
        num_examples=tf.constant(0),
        loss=tf.constant(0.0, dtype=tf.float32))

  update_tff_type = update_nt(
      model=model_tff_type, num_examples=tf.int32, loss=tf.float32)
  accumulator_tff_type = update_tff_type

  # We add an update to an accumulator with the update's model multipled by the
  # number of examples, so we can compute a weighted average in the end.
  @computations.tf_computation(accumulator_tff_type, update_tff_type)
  def accumulate(accumulator, update):
    scaling_factor = tf.cast(update.num_examples, tf.float32)
    scaled_model = tf.nest.map_structure(lambda x: x * scaling_factor,
                                         update.model)
    return accumulator_nt(
        model=tf.nest.map_structure(tf.add, accumulator.model, scaled_model),
        num_examples=accumulator.num_examples + update.num_examples,
        loss=accumulator.loss + update.loss * scaling_factor)

  # Merging accumulators does not involve scaling.
  @computations.tf_computation(accumulator_tff_type, accumulator_tff_type)
  def merge(accumulator1, accumulator2):
    return accumulator_nt(
        model=tf.nest.map_structure(tf.add, accumulator1.model,
                                    accumulator2.model),
        num_examples=accumulator1.num_examples + accumulator2.num_examples,
        loss=accumulator1.loss + accumulator2.loss)

  report_nt = accumulator_nt

  # The result of aggregation is produced by dividing the accumulated model by
  # the total number of examples. Same for loss.
  @computations.tf_computation(accumulator_tff_type)
  def report(accumulator):
    scaling_factor = 1.0 / tf.cast(accumulator.num_examples, tf.float32)
    scaled_model = model_nt(
        weights=accumulator.model.weights * scaling_factor,
        bias=accumulator.model.bias * scaling_factor)
    return report_nt(
        model=scaled_model,
        num_examples=accumulator.num_examples,
        loss=accumulator.loss * scaling_factor)

  @computations.tf_computation
  def bitwidth():
    return []

  update_type = computation_types.NamedTupleType([accumulator_tff_type, []])
  metrics_nt = collections.namedtuple('Metrics', 'num_rounds num_examples loss')

  # Pass the newly averaged model along with an incremented round counter over
  # to the next round, and output the counters and loss as server metrics.
  @computations.tf_computation(server_state_tff_type, update_type)
  def update(state, update):
    report = update[0]
    num_rounds = state.num_rounds + 1
    return (server_state_nt(model=report.model, num_rounds=num_rounds),
            metrics_nt(
                num_rounds=num_rounds,
                num_examples=report.num_examples,
                loss=report.loss))

  return canonical_form.CanonicalForm(initialize, prepare, work, zero,
                                      accumulate, merge, report, bitwidth,
                                      update)


def computation_to_building_block(comp):
  return building_blocks.ComputationBuildingBlock.from_proto(
      comp._computation_proto)  # pylint: disable=protected-access


def get_iterative_process_for_example_with_unused_lambda_arg():
  """Returns an iterative process having a Lambda not referencing its arg."""
  server_state_type = computation_types.NamedTupleType([('num_clients',
                                                         tf.int32)])

  def _bind_federated_value(unused_input, input_type, federated_output_value):
    federated_input_type = computation_types.FederatedType(
        input_type, placements.CLIENTS)
    wrapper = computations.federated_computation(
        lambda _: federated_output_value, federated_input_type)
    return wrapper(unused_input)

  def count_clients_federated(client_data):
    client_ones = intrinsics.federated_value(1, placements.CLIENTS)

    client_ones = _bind_federated_value(
        client_data, computation_types.SequenceType(tf.string), client_ones)
    return intrinsics.federated_sum(client_ones)

  @computations.federated_computation
  def init_fn():
    return intrinsics.federated_value(
        collections.OrderedDict(num_clients=0), placements.SERVER)

  @computations.federated_computation([
      computation_types.FederatedType(server_state_type, placements.SERVER),
      computation_types.FederatedType(
          computation_types.SequenceType(tf.string), placements.CLIENTS)
  ])
  def next_fn(server_state, client_val):
    """`next` function for `tff.templates.IterativeProcess`."""
    server_update = intrinsics.federated_zip(
        collections.OrderedDict(
            num_clients=count_clients_federated(client_val)))

    server_output = intrinsics.federated_value((), placements.SERVER)
    server_output = _bind_federated_value(
        intrinsics.federated_broadcast(server_state), server_state_type,
        server_output)

    return server_update, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_example_with_unused_tf_computation_arg():
  """Returns an iterative process with a @tf.function with an unused arg."""
  server_state_type = computation_types.NamedTupleType([('num_clients',
                                                         tf.int32)])

  def _bind_tf_function(unused_input, tf_func):
    tf_wrapper = tf.function(lambda _: tf_func())
    input_federated_type = unused_input.type_signature
    wrapper = computations.tf_computation(tf_wrapper,
                                          input_federated_type.member)
    return intrinsics.federated_map(wrapper, unused_input)

  def count_clients_federated(client_data):

    @tf.function
    def client_ones_fn():
      return tf.ones(shape=[], dtype=tf.int32)

    client_ones = _bind_tf_function(client_data, client_ones_fn)
    return intrinsics.federated_sum(client_ones)

  @computations.federated_computation
  def init_fn():
    return intrinsics.federated_value(
        collections.OrderedDict(num_clients=0), placements.SERVER)

  @computations.federated_computation([
      computation_types.FederatedType(server_state_type, placements.SERVER),
      computation_types.FederatedType(
          computation_types.SequenceType(tf.string), placements.CLIENTS)
  ])
  def next_fn(server_state, client_val):
    """`next` function for `tff.templates.IterativeProcess`."""
    server_update = intrinsics.federated_zip(
        collections.OrderedDict(
            num_clients=count_clients_federated(client_val)))

    server_output = intrinsics.federated_value((), placements.SERVER)
    server_output = intrinsics.federated_sum(
        _bind_tf_function(
            intrinsics.federated_broadcast(server_state), tf.timestamp))

    return server_update, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_example_with_lambda_returning_aggregation():
  """Gets iterative process with indirection to the called intrinsic."""
  server_state_type = computation_types.NamedTupleType([('num_clients',
                                                         tf.int32)])
  client_val_type = computation_types.FederatedType(server_state_type,
                                                    placements.CLIENTS)

  @computations.federated_computation()
  def computation_returning_lambda():

    @computations.federated_computation(tf.int32)
    def computation_returning_sum(x):
      tuple_containing_intrinsic = [
          building_blocks.Intrinsic(
              'federated_sum',
              computation_types.FunctionType(
                  client_val_type,
                  computation_types.FederatedType(client_val_type.member,
                                                  placements.SERVER))), x
      ]

      return tuple_containing_intrinsic[0]

    return computation_returning_sum

  @computations.federated_computation
  def init_fn():
    return intrinsics.federated_value(
        collections.OrderedDict(num_clients=0), placements.SERVER)

  @computations.federated_computation([
      computation_types.FederatedType(server_state_type, placements.SERVER),
      client_val_type,
  ])
  def next_fn(server_state, client_val):
    """`next` function for `tff.templates.IterativeProcess`."""
    server_update = intrinsics.federated_sum(client_val)
    server_output = intrinsics.federated_value((), placements.SERVER)
    state_at_clients = intrinsics.federated_broadcast(server_state)
    lambda_returning_sum = computation_returning_lambda()
    sum_fn = lambda_returning_sum(1)
    server_output = sum_fn(state_at_clients)

    return server_update, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)
