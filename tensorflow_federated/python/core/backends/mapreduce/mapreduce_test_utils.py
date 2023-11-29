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
"""Utilities for testing the mapreduce backend."""

import collections

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import iterative_process

MapReduceFormExample = collections.namedtuple(
    'MapReduceFormExample', ['mrf', 'initialize']
)


def generate_unnamed_type_signature(update, work):
  """Generates a type signature for the MapReduceForm based on components."""
  parameter = computation_types.StructType([
      (
          None,
          computation_types.FederatedType(
              update.type_signature.parameter[0], placements.SERVER
          ),
      ),
      (
          None,
          computation_types.FederatedType(
              work.type_signature.parameter[0], placements.CLIENTS
          ),
      ),
  ])
  result = computation_types.StructType([
      (
          None,
          computation_types.FederatedType(
              update.type_signature.parameter[0], placements.SERVER
          ),
      ),
      (
          None,
          computation_types.FederatedType(
              update.type_signature.result[1], placements.SERVER
          ),
      ),
  ])
  return computation_types.FunctionType(parameter, result)


def _make_map_reduce_form_example(
    initialize: computation_impl.ConcreteComputation,
    type_signature: computation_types.FunctionType,
    prepare: computation_impl.ConcreteComputation,
    work: computation_impl.ConcreteComputation,
    zero: computation_impl.ConcreteComputation,
    accumulate: computation_impl.ConcreteComputation,
    merge: computation_impl.ConcreteComputation,
    report: computation_impl.ConcreteComputation,
    secure_sum_bitwidth: computation_impl.ConcreteComputation,
    secure_sum_max_input: computation_impl.ConcreteComputation,
    secure_sum_modulus: computation_impl.ConcreteComputation,
    update: computation_impl.ConcreteComputation,
) -> MapReduceFormExample:
  """Constructs a MapReduceFormExample given the component comps."""

  def _uniquify_reference_names(comp: computation_impl.ConcreteComputation):
    return computation_impl.ConcreteComputation.from_building_block(
        tree_transformations.uniquify_reference_names(comp.to_building_block())[
            0
        ]
    )

  return MapReduceFormExample(
      mrf=forms.MapReduceForm(
          type_signature,
          _uniquify_reference_names(prepare),
          _uniquify_reference_names(work),
          _uniquify_reference_names(zero),
          _uniquify_reference_names(accumulate),
          _uniquify_reference_names(merge),
          _uniquify_reference_names(report),
          _uniquify_reference_names(secure_sum_bitwidth),
          _uniquify_reference_names(secure_sum_max_input),
          _uniquify_reference_names(secure_sum_modulus),
          _uniquify_reference_names(update),
      ),
      initialize=_uniquify_reference_names(initialize),
  )


def get_temperature_sensor_example():
  """Constructs `forms.MapReduceForm` for temperature sensors example.

  The temperature sensor example computes the fraction of sensors that report
  temperatures over the threshold.

  Returns:
    A tuple of: (1) an instance of `forms.MapReduceForm` and (2) an associated
    `computation_base.Computation` that generates an initial state compatible
    with the server state expected by the `forms.MapReduceForm`.
  """

  @federated_computation.federated_computation()
  def initialize():
    @tensorflow_computation.tf_computation
    def initialize_tf():
      return collections.OrderedDict(num_rounds=0)

    return intrinsics.federated_value(initialize_tf(), placements.SERVER)

  # The state of the server is a singleton tuple containing just the integer
  # counter `num_rounds`.
  server_state_type = collections.OrderedDict(num_rounds=np.int32)

  @tensorflow_computation.tf_computation(server_state_type)
  def prepare(state):
    return collections.OrderedDict(
        max_temperature=32.0 + tf.cast(state['num_rounds'], tf.float32)
    )

  # The initial state of the client is a singleton tuple containing a single
  # float `max_temperature`, which is the threshold received from the server.
  client_state_type = collections.OrderedDict(max_temperature=np.float32)

  # The client data is a sequence of floats.
  client_data_type = computation_types.SequenceType(np.float32)

  @tensorflow_computation.tf_computation(client_data_type, client_state_type)
  def work(data, state):
    """See the `forms.MapReduceForm` definition of `work`."""

    def fn(s, x):
      return {
          'num': s['num'] + 1,
          'max': tf.maximum(s['max'], x),
      }

    reduce_result = data.reduce(
        {'num': np.int32(0), 'max': np.float32(-459.67)}, fn
    )
    client_updates = collections.OrderedDict(
        is_over=reduce_result['max'] > state['max_temperature']
    )
    return client_updates, [], [], []

  # The client update is a singleton tuple with a Boolean-typed `is_over`.
  client_update_type = collections.OrderedDict(is_over=np.bool_)

  # The accumulator for client updates is a pair of counters, one for the
  # number of clients over threshold, and the other for the total number of
  # client updates processed so far.
  accumulator_type = collections.OrderedDict(
      num_total=np.int32, num_over=np.int32
  )

  @tensorflow_computation.tf_computation
  def zero():
    return collections.OrderedDict(num_total=0, num_over=0)

  @tensorflow_computation.tf_computation(accumulator_type, client_update_type)
  def accumulate(accumulator, update):
    return collections.OrderedDict(
        num_total=accumulator['num_total'] + 1,
        num_over=accumulator['num_over'] + tf.cast(update['is_over'], tf.int32),
    )

  @tensorflow_computation.tf_computation(accumulator_type, accumulator_type)
  def merge(accumulator1, accumulator2):
    return collections.OrderedDict(
        num_total=accumulator1['num_total'] + accumulator2['num_total'],
        num_over=accumulator1['num_over'] + accumulator2['num_over'],
    )

  @tensorflow_computation.tf_computation(merge.type_signature.result)
  def report(accumulator):
    return collections.OrderedDict(
        ratio_over_threshold=(
            tf.cast(accumulator['num_over'], tf.float32)
            / tf.cast(accumulator['num_total'], tf.float32)
        )
    )

  unit_comp = tensorflow_computation.tf_computation(lambda: [])
  bitwidth = unit_comp
  max_input = unit_comp
  modulus = unit_comp

  update_type = (
      collections.OrderedDict(ratio_over_threshold=np.float32),
      (),
      (),
      (),
  )

  @tensorflow_computation.tf_computation(server_state_type, update_type)
  def update(state, update):
    return (
        collections.OrderedDict(num_rounds=state['num_rounds'] + 1),
        update[0],
    )

  type_signature = generate_unnamed_type_signature(update, work)
  return _make_map_reduce_form_example(
      initialize,
      type_signature,
      prepare,
      work,
      zero,
      accumulate,
      merge,
      report,
      bitwidth,
      max_input,
      modulus,
      update,
  )


def get_federated_sum_example(
    *, secure_sum: bool = False
) -> tuple[forms.MapReduceForm, computation_base.Computation]:
  """Constructs `forms.MapReduceForm` which performs a sum aggregation.

  Args:
    secure_sum: Whether to use `federated_secure_sum_bitwidth`. Defaults to
      `federated_sum`.

  Returns:
    An instance of `forms.MapReduceForm`.
  """

  @tensorflow_computation.tf_computation
  def initialize_tf():
    return ()

  @federated_computation.federated_computation()
  def initialize():
    return intrinsics.federated_value(initialize_tf(), placements.SERVER)

  server_state_type = initialize_tf.type_signature.result

  @tensorflow_computation.tf_computation(server_state_type)
  def prepare(state):
    return state

  @tensorflow_computation.tf_computation(
      computation_types.SequenceType(np.int32), prepare.type_signature.result
  )
  def work(data, _):
    client_sum = data.reduce(initial_state=0, reduce_func=tf.add)
    if secure_sum:
      return [], client_sum, [], []
    else:
      return client_sum, [], [], []

  @tensorflow_computation.tf_computation
  def zero():
    if secure_sum:
      return ()
    else:
      return 0

  client_update_type = work.type_signature.result[0]
  accumulator_type = zero.type_signature.result

  @tensorflow_computation.tf_computation(accumulator_type, client_update_type)
  def accumulate(accumulator, update):
    if secure_sum:
      return ()
    else:
      return accumulator + update

  @tensorflow_computation.tf_computation(accumulator_type, accumulator_type)
  def merge(accumulator1, accumulator2):
    if secure_sum:
      return ()
    else:
      return accumulator1 + accumulator2

  @tensorflow_computation.tf_computation(merge.type_signature.result)
  def report(accumulator):
    return accumulator

  bitwidth = tensorflow_computation.tf_computation(lambda: 32)
  max_input = tensorflow_computation.tf_computation(lambda: 0)
  modulus = tensorflow_computation.tf_computation(lambda: 0)

  update_type = (
      merge.type_signature.result,
      work.type_signature.result[1],
      work.type_signature.result[2],
      work.type_signature.result[3],
  )

  @tensorflow_computation.tf_computation(server_state_type, update_type)
  def update(state, update):
    if secure_sum:
      return state, update[1]
    else:
      return state, update[0]

  type_signature = generate_unnamed_type_signature(update, work)
  return _make_map_reduce_form_example(
      initialize,
      type_signature,
      prepare,
      work,
      zero,
      accumulate,
      merge,
      report,
      bitwidth,
      max_input,
      modulus,
      update,
  )


def get_mnist_training_example():
  """Constructs `forms.MapReduceForm` for mnist training.

  Returns:
    An instance of `forms.MapReduceForm`.
  """
  model_nt = collections.namedtuple('Model', 'weights bias')
  server_state_nt = collections.namedtuple('ServerState', 'model num_rounds')

  # Start with a model filled with zeros, and the round counter set to zero.
  @federated_computation.federated_computation()
  def _initialize():
    @tensorflow_computation.tf_computation
    def initialize_tf():
      return server_state_nt(
          model=model_nt(
              weights=np.zeros([784, 10], np.float32),
              bias=np.zeros([10], np.float32),
          ),
          num_rounds=0,
      )

    return intrinsics.federated_value(initialize_tf(), placements.SERVER)

  server_state_tff_type = server_state_nt(
      model=model_nt(weights=(np.float32, [784, 10]), bias=(np.float32, [10])),
      num_rounds=np.int32,
  )
  client_state_nt = collections.namedtuple('ClientState', 'model learning_rate')

  # Pass the model to the client, along with a dynamically adjusted learning
  # rate that starts at 0.1 and decays exponentially by a factor of 0.9.
  @tensorflow_computation.tf_computation(server_state_tff_type)
  def _prepare(state):
    learning_rate = 0.1 * tf.pow(0.9, tf.cast(state.num_rounds, tf.float32))
    return client_state_nt(model=state.model, learning_rate=learning_rate)

  batch_nt = collections.namedtuple('Batch', 'x y')
  batch_tff_type = batch_nt(x=(np.float32, [None, 784]), y=(np.int32, [None]))
  dataset_tff_type = computation_types.SequenceType(batch_tff_type)
  model_tff_type = model_nt(
      weights=(np.float32, [784, 10]), bias=(np.float32, [10])
  )
  client_state_tff_type = client_state_nt(
      model=model_tff_type, learning_rate=np.float32
  )
  loop_state_nt = collections.namedtuple('LoopState', 'num_examples total_loss')
  update_nt = collections.namedtuple('Update', 'model num_examples loss')

  # Train the model locally, emit the loclaly-trained model and the number of
  # examples as an update, and the average loss and the number of examples as
  # local client stats.
  @tensorflow_computation.tf_computation(
      dataset_tff_type, client_state_tff_type
  )
  def _work(data, state):
    model_vars = model_nt(
        weights=tf.Variable(initial_value=state.model.weights, name='weights'),
        bias=tf.Variable(initial_value=state.model.bias, name='bias'),
    )
    init_model = tf.compat.v1.global_variables_initializer()

    optimizer = tf.keras.optimizers.SGD(state.learning_rate)

    @tf.function
    def reduce_fn(loop_state, batch):
      """Compute a single gradient step on an given batch of examples."""
      with tf.GradientTape() as tape:
        pred_y = tf.nn.softmax(
            tf.matmul(batch.x, model_vars.weights) + model_vars.bias
        )
        loss = -tf.reduce_mean(
            tf.reduce_sum(
                tf.one_hot(batch.y, 10) * tf.math.log(pred_y), axis=[1]
            )
        )
      grads = tape.gradient(loss, model_vars)
      optimizer.apply_gradients(
          zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars))
      )
      return loop_state_nt(
          num_examples=loop_state.num_examples + 1,
          total_loss=loop_state.total_loss + loss,
      )

    with tf.control_dependencies([init_model]):
      loop_state = data.reduce(
          loop_state_nt(num_examples=0, total_loss=np.float32(0.0)), reduce_fn
      )
      num_examples = loop_state.num_examples
      total_loss = loop_state.total_loss
      with tf.control_dependencies([num_examples, total_loss]):
        loss = total_loss / tf.cast(num_examples, tf.float32)

    return (
        update_nt(model=model_vars, num_examples=num_examples, loss=loss),
        [],
        [],
        [],
    )

  accumulator_nt = update_nt

  # Initialize accumulators for aggregation with zero model and zero examples.
  @tensorflow_computation.tf_computation
  def _zero():
    return accumulator_nt(
        model=model_nt(
            weights=np.zeros([784, 10], np.float32),
            bias=np.zeros([10], np.float32),
        ),
        num_examples=0,
        loss=0.0,
    )

  update_tff_type = update_nt(
      model=model_tff_type, num_examples=np.int32, loss=np.float32
  )
  accumulator_tff_type = update_tff_type

  # We add an update to an accumulator with the update's model multipled by the
  # number of examples, so we can compute a weighted average in the end.
  @tensorflow_computation.tf_computation(accumulator_tff_type, update_tff_type)
  def _accumulate(accumulator, update):
    scaling_factor = tf.cast(update.num_examples, tf.float32)
    scaled_model = tf.nest.map_structure(
        lambda x: x * scaling_factor, update.model
    )
    return accumulator_nt(
        model=tf.nest.map_structure(tf.add, accumulator.model, scaled_model),
        num_examples=accumulator.num_examples + update.num_examples,
        loss=accumulator.loss + update.loss * scaling_factor,
    )

  # Merging accumulators does not involve scaling.
  @tensorflow_computation.tf_computation(
      accumulator_tff_type, accumulator_tff_type
  )
  def _merge(accumulator1, accumulator2):
    return accumulator_nt(
        model=tf.nest.map_structure(
            tf.add, accumulator1.model, accumulator2.model
        ),
        num_examples=accumulator1.num_examples + accumulator2.num_examples,
        loss=accumulator1.loss + accumulator2.loss,
    )

  report_nt = accumulator_nt

  # The result of aggregation is produced by dividing the accumulated model by
  # the total number of examples. Same for loss.
  @tensorflow_computation.tf_computation(accumulator_tff_type)
  def _report(accumulator):
    scaling_factor = 1.0 / tf.cast(accumulator.num_examples, tf.float32)
    scaled_model = model_nt(
        weights=accumulator.model.weights * scaling_factor,
        bias=accumulator.model.bias * scaling_factor,
    )
    return report_nt(
        model=scaled_model,
        num_examples=accumulator.num_examples,
        loss=accumulator.loss * scaling_factor,
    )

  unit_computation = tensorflow_computation.tf_computation(lambda: [])
  secure_sum_bitwidth = unit_computation
  secure_sum_max_input = unit_computation
  secure_sum_modulus = unit_computation

  update_type = (accumulator_tff_type, (), (), ())
  metrics_nt = collections.namedtuple('Metrics', 'num_rounds num_examples loss')

  # Pass the newly averaged model along with an incremented round counter over
  # to the next round, and output the counters and loss as server metrics.
  @tensorflow_computation.tf_computation(server_state_tff_type, update_type)
  def _update(state, update):
    report = update[0]
    num_rounds = state.num_rounds + 1
    return (
        server_state_nt(model=report.model, num_rounds=num_rounds),
        metrics_nt(
            num_rounds=num_rounds,
            num_examples=report.num_examples,
            loss=report.loss,
        ),
    )

  type_signature = generate_unnamed_type_signature(_update, _work)
  return _make_map_reduce_form_example(
      _initialize,
      type_signature,
      _prepare,
      _work,
      _zero,
      _accumulate,
      _merge,
      _report,
      secure_sum_bitwidth,
      secure_sum_max_input,
      secure_sum_modulus,
      _update,
  )


def get_iterative_process_for_example_with_unused_lambda_arg():
  """Returns an iterative process having a Lambda not referencing its arg."""
  server_state_type = collections.OrderedDict(num_clients=np.int32)

  def _bind_federated_value(unused_input, input_type, federated_output_value):
    federated_input_type = computation_types.FederatedType(
        input_type, placements.CLIENTS
    )
    wrapper = federated_computation.federated_computation(
        lambda _: federated_output_value, federated_input_type
    )
    return wrapper(unused_input)

  def count_clients_federated(client_data):
    client_ones = intrinsics.federated_value(1, placements.CLIENTS)

    client_ones = _bind_federated_value(
        client_data, computation_types.SequenceType(np.str_), client_ones
    )
    return intrinsics.federated_sum(client_ones)

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_value(
        collections.OrderedDict(num_clients=0), placements.SERVER
    )

  @federated_computation.federated_computation([
      computation_types.FederatedType(server_state_type, placements.SERVER),
      computation_types.FederatedType(
          computation_types.SequenceType(np.str_), placements.CLIENTS
      ),
  ])
  def next_fn(server_state, client_val):
    """`next` function for `tff.templates.IterativeProcess`."""
    server_update = intrinsics.federated_zip(
        collections.OrderedDict(num_clients=count_clients_federated(client_val))
    )

    server_output = intrinsics.federated_value((), placements.SERVER)
    server_output = _bind_federated_value(
        intrinsics.federated_broadcast(server_state),
        server_state_type,
        server_output,
    )

    return server_update, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_example_with_unused_tf_computation_arg():
  """Returns an iterative process with a @tf.function with an unused arg."""
  server_state_type = collections.OrderedDict(num_clients=np.int32)

  def _bind_tf_function(unused_input, tf_func):
    tf_wrapper = tf.function(lambda _: tf_func())
    input_federated_type = unused_input.type_signature
    wrapper = tensorflow_computation.tf_computation(
        tf_wrapper,
        input_federated_type.member,  # pytype: disable=attribute-error
    )
    return intrinsics.federated_map(wrapper, unused_input)

  def count_clients_federated(client_data):
    @tf.function
    def client_ones_fn():
      return np.ones(shape=[], dtype=np.int32)

    client_ones = _bind_tf_function(client_data, client_ones_fn)
    return intrinsics.federated_sum(client_ones)

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_value(
        collections.OrderedDict(num_clients=0), placements.SERVER
    )

  @federated_computation.federated_computation([
      computation_types.FederatedType(server_state_type, placements.SERVER),
      computation_types.FederatedType(
          computation_types.SequenceType(np.str_), placements.CLIENTS
      ),
  ])
  def next_fn(server_state, client_val):
    """`next` function for `tff.templates.IterativeProcess`."""
    server_update = intrinsics.federated_zip(
        collections.OrderedDict(num_clients=count_clients_federated(client_val))
    )

    server_output = intrinsics.federated_sum(
        _bind_tf_function(
            intrinsics.federated_broadcast(server_state), tf.timestamp
        )
    )

    return server_update, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_example_with_lambda_returning_aggregation():
  """Gets iterative process with indirection to the called intrinsic."""
  server_state_type = collections.OrderedDict(num_clients=np.int32)
  client_val_type = computation_types.FederatedType(
      server_state_type, placements.CLIENTS
  )

  @federated_computation.federated_computation
  def computation_returning_lambda():

    @federated_computation.federated_computation(np.int32)
    def computation_returning_sum(x):
      tuple_containing_intrinsic = [
          building_blocks.Intrinsic(
              'federated_sum',
              computation_types.FunctionType(
                  client_val_type,
                  computation_types.FederatedType(
                      client_val_type.member, placements.SERVER
                  ),
              ),
          ),
          x,
      ]

      return tuple_containing_intrinsic[0]

    return computation_returning_sum

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_value(
        collections.OrderedDict(num_clients=0), placements.SERVER
    )

  @federated_computation.federated_computation([
      computation_types.FederatedType(server_state_type, placements.SERVER),
      client_val_type,
  ])
  def next_fn(server_state, client_val):
    """`next` function for `tff.templates.IterativeProcess`."""
    server_update = intrinsics.federated_sum(client_val)
    state_at_clients = intrinsics.federated_broadcast(server_state)
    lambda_returning_sum = computation_returning_lambda()
    sum_fn = lambda_returning_sum(1)
    server_output = sum_fn(state_at_clients)

    return server_update, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)
