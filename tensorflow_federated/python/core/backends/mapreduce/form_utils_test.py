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

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.mapreduce import distribute_aggregate_test_utils
from tensorflow_federated.python.core.backends.mapreduce import form_utils
from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.backends.mapreduce import mapreduce_test_utils
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.templates import iterative_process


def get_iterative_process_for_sum_example():
  """Returns an iterative process for a sum example.

  This iterative process contains all the components required to compile to
  `forms.MapReduceForm`.
  """

  @federated_computation.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @tensorflow_computation.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @tensorflow_computation.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, 1

  @tensorflow_computation.tf_computation(
      [tf.int32, tf.int32], [tf.int32, tf.int32]
  )
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @federated_computation.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum_bitwidth(
        client_updates[1], 8
    )
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]]
    )
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_computation_with_nested_broadcasts():
  """Returns a computation with nested federated broadcasts.

  This computation contains all the components required to compile to
  `forms.MapReduceForm`.
  """

  @tensorflow_computation.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @tensorflow_computation.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, 1

  @tensorflow_computation.tf_computation(
      [tf.int32, tf.int32], [tf.int32, tf.int32]
  )
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @federated_computation.federated_computation(
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER)
  )
  def broadcast_and_return_arg_and_result(x):
    broadcasted = intrinsics.federated_broadcast(x)
    return [broadcasted, x]

  @federated_computation.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def comp_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    unused_client_input, to_broadcast = broadcast_and_return_arg_and_result(s2)
    client_input = intrinsics.federated_broadcast(to_broadcast)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum_bitwidth(
        client_updates[1], 8
    )
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]]
    )
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return comp_fn


def get_iterative_process_for_sum_example_with_no_prepare():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_map` with a prepare
  function before the `federated_broadcast`.
  """

  @federated_computation.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @tensorflow_computation.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, 1

  @tensorflow_computation.tf_computation(
      [tf.int32, tf.int32], [tf.int32, tf.int32]
  )
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @federated_computation.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    # No call to `federated_map` with a `prepare` function.
    client_input = intrinsics.federated_broadcast(server_state)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum_bitwidth(
        client_updates[1], 8
    )
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]]
    )
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_broadcast():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_broadcast`. As a
  result, this iterative process does not have a call to `federated_map` with a
  prepare function before the `federated_broadcast`.
  """

  @federated_computation.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @tensorflow_computation.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return 1, 1

  @tensorflow_computation.tf_computation(
      [tf.int32, tf.int32], [tf.int32, tf.int32]
  )
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @federated_computation.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    # No call to `federated_map` with prepare.
    # No call to `federated_broadcast`.
    client_updates = intrinsics.federated_map(work, client_data)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum_bitwidth(
        client_updates[1], 8
    )
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]]
    )
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_federated_aggregate():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_aggregate`.
  """

  @federated_computation.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value(0, placements.SERVER)

  @tensorflow_computation.tf_computation(tf.int32)
  def prepare(server_state):
    return server_state

  @tensorflow_computation.tf_computation(tf.int32, tf.int32)
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1

  @tensorflow_computation.tf_computation([tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @federated_computation.federated_computation([
      computation_types.FederatedType(tf.int32, placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates = intrinsics.federated_map(work, c3)
    # No call to `federated_aggregate`.
    secure_update = intrinsics.federated_secure_sum_bitwidth(client_updates, 8)
    s6 = intrinsics.federated_zip([server_state, secure_update])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_federated_secure_sum_bitwidth():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to
  `federated_secure_sum_bitwidth`.
  """

  @federated_computation.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value(0, placements.SERVER)

  @tensorflow_computation.tf_computation(tf.int32)
  def prepare(server_state):
    return server_state

  @tensorflow_computation.tf_computation(tf.int32, tf.int32)
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1

  @tensorflow_computation.tf_computation([tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @federated_computation.federated_computation([
      computation_types.FederatedType(tf.int32, placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates)
    # No call to `federated_secure_sum_bitwidth`.
    s6 = intrinsics.federated_zip([server_state, unsecure_update])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_update():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_map` with a prepare
  function before the `federated_broadcast`.
  """

  @federated_computation.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @tensorflow_computation.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @tensorflow_computation.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, 1

  @federated_computation.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum_bitwidth(
        client_updates[1], 8
    )
    new_server_state = intrinsics.federated_zip(
        [unsecure_update, secure_update]
    )
    # No call to `federated_map` with an `update` function.
    server_output = intrinsics.federated_value([], placements.SERVER)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_server_state():
  """Returns an iterative process for a sum example.

  This iterative process does not use the server state passed into the next
  function and returns an empty server state from the next function. As a
  result, this iterative process does not have a call to `federated_broadcast`
  and it does not have a call to `federated_map` with a prepare function before
  the `federated_broadcast`.
  """

  @federated_computation.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([], placements.SERVER)

  @tensorflow_computation.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return 1, 1

  @tensorflow_computation.tf_computation([tf.int32, tf.int32])
  def update(global_update):
    return global_update

  @federated_computation.federated_computation([
      computation_types.FederatedType([], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    del server_state  # Unused
    # No call to `federated_map` with prepare.
    # No call to `federated_broadcast`.
    client_updates = intrinsics.federated_map(work, client_data)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum_bitwidth(
        client_updates[1], 8
    )
    s5 = intrinsics.federated_zip([unsecure_update, secure_update])
    # Empty server state.
    new_server_state = intrinsics.federated_value([], placements.SERVER)
    server_output = intrinsics.federated_map(update, s5)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_aggregation():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_aggregate` or
  `federated_secure_sum_bitwidth` and as a result it should fail to compile to
  `forms.MapReduceForm`.
  """

  @federated_computation.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @tensorflow_computation.tf_computation(
      [tf.int32, tf.int32], [tf.int32, tf.int32]
  )
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @federated_computation.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    del client_data
    # No call to `federated_aggregate`.
    unsecure_update = intrinsics.federated_value(1, placements.SERVER)
    # No call to `federated_secure_sum_bitwidth`.
    secure_update = intrinsics.federated_value(1, placements.SERVER)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]]
    )
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_minimal_sum_example():
  """Returns an iterative process for a sum example.

  This iterative process contains the fewest components required to compile to
  `forms.MapReduceForm`.
  """

  @federated_computation.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    zero = tensorflow_computation.tf_computation(lambda: [0, 0, 0, 0])
    return intrinsics.federated_eval(zero, placements.SERVER)

  @tensorflow_computation.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return 1, 1, 1, 1

  @federated_computation.federated_computation([
      computation_types.at_server([tf.int32, tf.int32, tf.int32, tf.int32]),
      computation_types.at_clients(tf.int32),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    del server_state  # Unused
    # No call to `federated_map` with prepare.
    # No call to `federated_broadcast`.
    client_updates = intrinsics.federated_map(work, client_data)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_sum_bitwidth_update = intrinsics.federated_secure_sum_bitwidth(
        client_updates[1], bitwidth=8
    )
    secure_sum_update = intrinsics.federated_secure_sum(
        client_updates[2], max_input=1
    )
    secure_modular_sum_update = intrinsics.federated_secure_modular_sum(
        client_updates[3], modulus=8
    )
    new_server_state = intrinsics.federated_zip([
        unsecure_update,
        secure_sum_bitwidth_update,
        secure_sum_update,
        secure_modular_sum_update,
    ])
    # No call to `federated_map` with an `update` function.
    server_output = intrinsics.federated_value([], placements.SERVER)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_example_cf_compatible_iterative_processes():
  # pyformat: disable
  return [
      ('sum_example',
       get_iterative_process_for_sum_example()),
      ('sum_example_with_no_prepare',
       get_iterative_process_for_sum_example_with_no_prepare()),
      ('sum_example_with_no_broadcast',
       get_iterative_process_for_sum_example_with_no_broadcast()),
      ('sum_example_with_no_federated_aggregate',
       get_iterative_process_for_sum_example_with_no_federated_aggregate()),
      ('sum_example_with_no_federated_secure_sum_bitwidth',
       get_iterative_process_for_sum_example_with_no_federated_secure_sum_bitwidth()),
      ('sum_example_with_no_update',
       get_iterative_process_for_sum_example_with_no_update()),
      ('sum_example_with_no_server_state',
       get_iterative_process_for_sum_example_with_no_server_state()),
      ('minimal_sum_example',
       get_iterative_process_for_minimal_sum_example()),
      ('example_with_unused_lambda_arg',
       mapreduce_test_utils.get_iterative_process_for_example_with_unused_lambda_arg()),
      ('example_with_unused_tf_computation_arg',
       mapreduce_test_utils.get_iterative_process_for_example_with_unused_tf_computation_arg()),
  ]
  # pyformat: enable


class FederatedFormTestCase(tf.test.TestCase):
  """A base class that overrides evaluate to handle various executors."""

  def evaluate(self, value):
    if tf.is_tensor(value):
      return super().evaluate(value)
    elif isinstance(value, (np.ndarray, np.number)):
      return value
    else:
      raise TypeError(
          'Cannot evaluate value of type `{!s}`.'.format(type(value))
      )


class GetComputationForDistributeAggregateFormTest(
    FederatedFormTestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      (
          'temperature',
          distribute_aggregate_test_utils.get_temperature_sensor_example(),
      ),
      ('mnist', distribute_aggregate_test_utils.get_mnist_training_example()),
  )
  def test_type_signature_matches_generated_computation(self, example):
    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    self.assertTrue(
        comp.type_signature.is_equivalent_to(example.daf.type_signature)
    )

  def test_with_temperature_sensor_example(self):
    example = distribute_aggregate_test_utils.get_temperature_sensor_example()

    state = example.initialize()

    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    state, metrics = comp(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertAllEqual(state, collections.OrderedDict(num_rounds=1))
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.5)
    )

    state, metrics = comp(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.75)
    )


class GetDistributeAggregateFormTest(
    FederatedFormTestCase, parameterized.TestCase
):

  def test_next_computation_returning_tensor_fails_well(self):
    initialize = (
        distribute_aggregate_test_utils.get_temperature_sensor_example().initialize
    )
    init_result = initialize.type_signature.result
    lam = building_blocks.Lambda(
        'x', init_result, building_blocks.Reference('x', init_result)
    )
    bad_comp = computation_impl.ConcreteComputation.from_building_block(lam)
    with self.assertRaises(TypeError):
      form_utils.get_distribute_aggregate_form_for_computation(bad_comp)

  def test_broadcast_dependent_on_aggregate_fails_well(self):
    example = distribute_aggregate_test_utils.get_mnist_training_example()
    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    comp_bb = comp.to_building_block()
    top_level_param = building_blocks.Reference(
        comp_bb.parameter_name, comp_bb.parameter_type
    )
    first_result = building_blocks.Call(comp_bb, top_level_param)
    middle_param = building_blocks.Struct([
        building_blocks.Selection(first_result, index=0),
        building_blocks.Selection(top_level_param, index=1),
    ])
    second_result = building_blocks.Call(comp_bb, middle_param)
    not_reducible = building_blocks.Lambda(
        comp_bb.parameter_name, comp_bb.parameter_type, second_result
    )
    bad_comp = computation_impl.ConcreteComputation.from_building_block(
        not_reducible
    )
    with self.assertRaisesRegex(ValueError, 'broadcast dependent on aggregate'):
      form_utils.get_distribute_aggregate_form_for_computation(bad_comp)

  def test_gets_distribute_aggregate_form_for_nested_broadcast(self):
    comp = get_computation_with_nested_broadcasts()
    daf = form_utils.get_distribute_aggregate_form_for_computation(comp)
    self.assertIsInstance(daf, forms.DistributeAggregateForm)

  def test_constructs_distribute_aggregate_form_from_mnist_training_example(
      self,
  ):
    comp = form_utils.get_computation_for_distribute_aggregate_form(
        distribute_aggregate_test_utils.get_mnist_training_example().daf
    )
    daf = form_utils.get_distribute_aggregate_form_for_computation(comp)
    self.assertIsInstance(daf, forms.DistributeAggregateForm)

  def test_temperature_example_round_trip(self):
    example = distribute_aggregate_test_utils.get_temperature_sensor_example()
    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    new_initialize = form_utils.get_state_initialization_computation(
        example.initialize
    )
    new_daf = form_utils.get_distribute_aggregate_form_for_computation(comp)
    new_comp = form_utils.get_computation_for_distribute_aggregate_form(new_daf)

    state = new_initialize()
    self.assertEqual(state['num_rounds'], 0)

    state, metrics = new_comp(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertEqual(state['num_rounds'], 1)
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.5)
    )

    state, metrics = new_comp(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.75)
    )
    # Check that no TF work has been unintentionally duplicated.
    self.assertEqual(
        tree_analysis.count_tensorflow_variables_under(
            comp.to_building_block()
        ),
        tree_analysis.count_tensorflow_variables_under(
            new_comp.to_building_block()
        ),
    )

  def test_mnist_training_round_trip(self):
    example = distribute_aggregate_test_utils.get_mnist_training_example()
    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    new_initialize = form_utils.get_state_initialization_computation(
        example.initialize
    )
    new_daf = form_utils.get_distribute_aggregate_form_for_computation(comp)
    new_comp = form_utils.get_computation_for_distribute_aggregate_form(new_daf)

    state1 = example.initialize()
    state2 = new_initialize()
    self.assertAllClose(state1, state2)
    whimsy_x = np.array([[0.5] * 784], dtype=np.float32)
    whimsy_y = np.array([1], dtype=np.int32)
    client_data = [collections.OrderedDict(x=whimsy_x, y=whimsy_y)]
    round_1 = new_comp(state1, [client_data])
    state = round_1[0]
    metrics = round_1[1]
    alt_round_1 = new_comp(state2, [client_data])
    alt_state = alt_round_1[0]
    self.assertAllClose(state, alt_state)
    alt_metrics = alt_round_1[1]
    self.assertAllClose(metrics, alt_metrics)
    # Check that no TF work has been unintentionally duplicated.
    self.assertEqual(
        tree_analysis.count_tensorflow_variables_under(
            comp.to_building_block()
        ),
        tree_analysis.count_tensorflow_variables_under(
            new_comp.to_building_block()
        ),
    )

  @parameterized.named_parameters(
      *get_example_cf_compatible_iterative_processes()
  )
  def test_returns_distribute_aggregate_form(self, ip):
    daf = form_utils.get_distribute_aggregate_form_for_computation(ip.next)
    self.assertIsInstance(daf, forms.DistributeAggregateForm)

  def test_returns_distribute_aggregate_form_with_indirection_to_intrinsic(
      self,
  ):
    ip = (
        mapreduce_test_utils.get_iterative_process_for_example_with_lambda_returning_aggregation()
    )
    daf = form_utils.get_distribute_aggregate_form_for_computation(ip.next)
    self.assertIsInstance(daf, forms.DistributeAggregateForm)


class GetComputationForMapReduceFormTest(
    FederatedFormTestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('temperature', mapreduce_test_utils.get_temperature_sensor_example()),
      ('mnist', mapreduce_test_utils.get_mnist_training_example()),
  )
  def test_type_signature_matches_generated_computation(self, example):
    comp = form_utils.get_computation_for_map_reduce_form(example.mrf)
    self.assertTrue(
        comp.type_signature.is_equivalent_to(example.mrf.type_signature)
    )

  def test_with_temperature_sensor_example(self):
    example = mapreduce_test_utils.get_temperature_sensor_example()

    state = example.initialize()

    comp = form_utils.get_computation_for_map_reduce_form(example.mrf)
    state, metrics = comp(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertAllEqual(state, collections.OrderedDict(num_rounds=1))
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.5)
    )

    state, metrics = comp(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.75)
    )


class CheckMapReduceFormCompatibleWithComputationTest(
    FederatedFormTestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      *get_example_cf_compatible_iterative_processes()
  )
  def test_allows_valid_computation(self, ip):
    form_utils.check_computation_compatible_with_map_reduce_form(ip.next)

  def test_disallows_broadcast_dependent_on_aggregate(self):

    @federated_computation.federated_computation(
        computation_types.at_server(tf.int32), computation_types.at_clients(())
    )
    def comp(server_state, client_data):
      del server_state, client_data
      client_val = intrinsics.federated_value(0, placements.CLIENTS)
      server_agg = intrinsics.federated_sum(client_val)
      # This broadcast is dependent on the result of the above aggregation,
      # which is not supported by MapReduce form.
      broadcasted = intrinsics.federated_broadcast(server_agg)
      server_agg_again = intrinsics.federated_sum(broadcasted)
      # `next` must return two values.
      return server_agg_again, intrinsics.federated_value((), placements.SERVER)

    with self.assertRaises(ValueError):
      form_utils.check_computation_compatible_with_map_reduce_form(comp)


class GetMapReduceFormTest(FederatedFormTestCase, parameterized.TestCase):

  def test_next_computation_returning_tensor_fails_well(self):
    initialize = (
        mapreduce_test_utils.get_temperature_sensor_example().initialize
    )
    init_result = initialize.type_signature.result
    lam = building_blocks.Lambda(
        'x', init_result, building_blocks.Reference('x', init_result)
    )
    bad_comp = computation_impl.ConcreteComputation.from_building_block(lam)
    with self.assertRaises(TypeError):
      form_utils.get_map_reduce_form_for_computation(bad_comp)

  def test_broadcast_dependent_on_aggregate_fails_well(self):
    example = mapreduce_test_utils.get_temperature_sensor_example()
    comp = form_utils.get_computation_for_map_reduce_form(example.mrf)
    comp_bb = comp.to_building_block()
    top_level_param = building_blocks.Reference(
        comp_bb.parameter_name, comp_bb.parameter_type
    )
    first_result = building_blocks.Call(comp_bb, top_level_param)
    middle_param = building_blocks.Struct([
        building_blocks.Selection(first_result, index=0),
        building_blocks.Selection(top_level_param, index=1),
    ])
    second_result = building_blocks.Call(comp_bb, middle_param)
    not_reducible = building_blocks.Lambda(
        comp_bb.parameter_name, comp_bb.parameter_type, second_result
    )
    bad_comp = computation_impl.ConcreteComputation.from_building_block(
        not_reducible
    )

    with self.assertRaisesRegex(ValueError, 'broadcast dependent on aggregate'):
      form_utils.get_map_reduce_form_for_computation(bad_comp)

  def test_gets_map_reduce_form_for_nested_broadcast(self):
    comp = get_computation_with_nested_broadcasts()
    mrf = form_utils.get_map_reduce_form_for_computation(comp)
    self.assertIsInstance(mrf, forms.MapReduceForm)

  def test_constructs_map_reduce_form_from_mnist_training_example(self):
    comp = form_utils.get_computation_for_map_reduce_form(
        mapreduce_test_utils.get_mnist_training_example().mrf
    )
    mrf = form_utils.get_map_reduce_form_for_computation(comp)
    self.assertIsInstance(mrf, forms.MapReduceForm)

  def test_temperature_example_round_trip(self):
    # NOTE: the roundtrip through MapReduceForm->Comp->MapReduceForm seems
    # to lose the python container annotations on the StructType.
    example = mapreduce_test_utils.get_temperature_sensor_example()
    comp = form_utils.get_computation_for_map_reduce_form(example.mrf)
    new_initialize = form_utils.get_state_initialization_computation(
        example.initialize
    )
    new_mrf = form_utils.get_map_reduce_form_for_computation(comp)
    new_comp = form_utils.get_computation_for_map_reduce_form(new_mrf)
    state = new_initialize()
    self.assertEqual(state['num_rounds'], 0)

    state, metrics = new_comp(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertEqual(state['num_rounds'], 1)
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.5)
    )

    state, metrics = new_comp(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.75)
    )
    self.assertEqual(
        tree_analysis.count_tensorflow_variables_under(
            comp.to_building_block()
        ),
        tree_analysis.count_tensorflow_variables_under(
            new_comp.to_building_block()
        ),
    )

  def test_mnist_training_round_trip(self):
    example = mapreduce_test_utils.get_mnist_training_example()
    comp = form_utils.get_computation_for_map_reduce_form(example.mrf)

    # TODO(b/208887729): We disable grappler to work around attempting to hoist
    # transformed functions of the same name into the eager context. When this
    # execution is C++-backed, this can go away.
    grappler_config = tf.compat.v1.ConfigProto()
    grappler_config.graph_options.rewrite_options.disable_meta_optimizer = True
    new_initialize = form_utils.get_state_initialization_computation(
        example.initialize
    )
    new_mrf = form_utils.get_map_reduce_form_for_computation(
        comp, grappler_config
    )
    new_comp = form_utils.get_computation_for_map_reduce_form(new_mrf)

    state1 = example.initialize()
    state2 = new_initialize()
    self.assertAllClose(state1, state2)
    whimsy_x = np.array([[0.5] * 784], dtype=np.float32)
    whimsy_y = np.array([1], dtype=np.int32)
    client_data = [collections.OrderedDict(x=whimsy_x, y=whimsy_y)]
    round_1 = new_comp(state1, [client_data])
    state = round_1[0]
    metrics = round_1[1]
    alt_round_1 = new_comp(state2, [client_data])
    alt_state = alt_round_1[0]
    self.assertAllClose(state, alt_state)
    alt_metrics = alt_round_1[1]
    self.assertAllClose(metrics, alt_metrics)
    self.assertEqual(
        tree_analysis.count_tensorflow_variables_under(
            comp.to_building_block()
        ),
        tree_analysis.count_tensorflow_variables_under(
            new_comp.to_building_block()
        ),
    )

  @parameterized.named_parameters(
      *get_example_cf_compatible_iterative_processes()
  )
  def test_returns_map_reduce_form(self, ip):
    mrf = form_utils.get_map_reduce_form_for_computation(ip.next)

    self.assertIsInstance(mrf, forms.MapReduceForm)

  @parameterized.named_parameters(
      *get_example_cf_compatible_iterative_processes()
  )
  def test_returns_canonical_form_with_grappler_disabled(self, ip):
    grappler_config = tf.compat.v1.ConfigProto()
    grappler_config.graph_options.rewrite_options.disable_meta_optimizer = True
    mrf = form_utils.get_map_reduce_form_for_computation(
        ip.next, grappler_config
    )

    self.assertIsInstance(mrf, forms.MapReduceForm)

  def test_returns_map_reduce_form_for_sum_example_with_no_aggregation(self):
    ip = get_iterative_process_for_sum_example_with_no_aggregation()
    mrf = form_utils.get_map_reduce_form_for_computation(ip.next)
    self.assertIsInstance(mrf, forms.MapReduceForm)

  def test_returns_map_reduce_form_with_indirection_to_intrinsic(self):
    ip = (
        mapreduce_test_utils.get_iterative_process_for_example_with_lambda_returning_aggregation()
    )

    mrf = form_utils.get_map_reduce_form_for_computation(ip.next)

    self.assertIsInstance(mrf, forms.MapReduceForm)

  def get_map_reduce_form_for_client_to_server_fn(
      self, client_to_server_fn
  ) -> forms.MapReduceForm:
    """Produces a `MapReduceForm` for the provided `client_to_server_fn`.

    Creates an `iterative_process.IterativeProcess` which uses
    `client_to_server_fn` to map from `client_data` to `server_output`, then
    passes this value through `get_map_reduce_form_for_computation`.

    Args:
      client_to_server_fn: A function from client-placed data to server-placed
        output.

    Returns:
      A `forms.MapReduceForm` which uses the embedded `client_to_server_fn`.
    """

    @federated_computation.federated_computation([
        computation_types.at_server(()),
        computation_types.at_clients(tf.int32),
    ])
    def comp_fn(server_state, client_data):
      server_output = client_to_server_fn(client_data)
      return server_state, server_output

    return form_utils.get_map_reduce_form_for_computation(comp_fn)

  def test_returns_map_reduce_form_with_secure_sum_bitwidth(self):
    mrf = self.get_map_reduce_form_for_client_to_server_fn(
        lambda data: intrinsics.federated_secure_sum_bitwidth(data, 7)
    )
    self.assertEqual(mrf.secure_sum_bitwidth(), (7,))

  def test_returns_map_reduce_form_with_secure_sum_max_input(self):
    mrf = self.get_map_reduce_form_for_client_to_server_fn(
        lambda data: intrinsics.federated_secure_sum(data, 12)
    )
    self.assertEqual(mrf.secure_sum_max_input(), (12,))

  def test_returns_map_reduce_form_with_secure_modular_sum_modulus(self):
    mrf = self.get_map_reduce_form_for_client_to_server_fn(
        lambda data: intrinsics.federated_secure_modular_sum(data, 22)
    )
    self.assertEqual(mrf.secure_modular_sum_modulus(), (22,))


class BroadcastFormTest(tf.test.TestCase):

  def test_roundtrip(self):
    add = tensorflow_computation.tf_computation(lambda x, y: x + y)
    server_data_type = computation_types.at_server(tf.int32)
    client_data_type = computation_types.at_clients(tf.int32)

    @federated_computation.federated_computation(
        server_data_type, client_data_type
    )
    def add_server_number_plus_one(server_number, client_numbers):
      one = intrinsics.federated_value(1, placements.SERVER)
      server_context = intrinsics.federated_map(add, (one, server_number))
      client_context = intrinsics.federated_broadcast(server_context)
      return intrinsics.federated_map(add, (client_context, client_numbers))

    bf = form_utils.get_broadcast_form_for_computation(
        add_server_number_plus_one
    )
    self.assertEqual(bf.server_data_label, 'server_number')
    self.assertEqual(bf.client_data_label, 'client_numbers')
    type_test_utils.assert_types_equivalent(
        bf.compute_server_context.type_signature,
        computation_types.FunctionType(tf.int32, (tf.int32,)),
    )
    self.assertEqual(2, bf.compute_server_context(1)[0])
    type_test_utils.assert_types_equivalent(
        bf.client_processing.type_signature,
        computation_types.FunctionType(((tf.int32,), tf.int32), tf.int32),
    )
    self.assertEqual(3, bf.client_processing((1,), 2))

    round_trip_comp = form_utils.get_computation_for_broadcast_form(bf)
    type_test_utils.assert_types_equivalent(
        round_trip_comp.type_signature,
        add_server_number_plus_one.type_signature,
    )
    # 2 (server data) + 1 (constant in comp) + 2 (client data) = 5 (output)
    self.assertEqual([5, 6, 7], round_trip_comp(2, [2, 3, 4]))

  def test_roundtrip_no_broadcast(self):
    add_five = tensorflow_computation.tf_computation(lambda x: x + 5)
    server_data_type = computation_types.at_server(())
    client_data_type = computation_types.at_clients(tf.int32)

    @federated_computation.federated_computation(
        server_data_type, client_data_type
    )
    def add_five_at_clients(naught_at_server, client_numbers):
      del naught_at_server
      return intrinsics.federated_map(add_five, client_numbers)

    bf = form_utils.get_broadcast_form_for_computation(add_five_at_clients)
    self.assertEqual(bf.server_data_label, 'naught_at_server')
    self.assertEqual(bf.client_data_label, 'client_numbers')
    type_test_utils.assert_types_equivalent(
        bf.compute_server_context.type_signature,
        computation_types.FunctionType((), ()),
    )
    type_test_utils.assert_types_equivalent(
        bf.client_processing.type_signature,
        computation_types.FunctionType(((), tf.int32), tf.int32),
    )
    self.assertEqual(6, bf.client_processing((), 1))

    round_trip_comp = form_utils.get_computation_for_broadcast_form(bf)
    type_test_utils.assert_types_equivalent(
        round_trip_comp.type_signature, add_five_at_clients.type_signature
    )
    self.assertEqual([10, 11, 12], round_trip_comp((), [5, 6, 7]))


class AsFunctionOfSingleSubparameterTest(tf.test.TestCase):

  def assert_selected_param_to_result_type(self, old_lam, new_lam, index):
    old_type = old_lam.type_signature
    new_type = new_lam.type_signature
    old_type.check_function()
    new_type.check_function()
    type_test_utils.assert_types_equivalent(
        new_type,
        computation_types.FunctionType(
            old_type.parameter[index], old_type.result
        ),
    )

  def test_selection(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server]
    )
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0
        ),
    )
    new_lam = form_utils._as_function_of_single_subparameter(lam, 0)
    self.assert_selected_param_to_result_type(lam, new_lam, 0)

  def test_named_element_selection(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType([
        (None, fed_at_server),
        ('a', fed_at_clients),
    ])
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), name='a'
        ),
    )
    new_lam = form_utils._as_function_of_single_subparameter(lam, 1)
    self.assert_selected_param_to_result_type(lam, new_lam, 1)


class AsFunctionOfSomeSubparametersTest(tf.test.TestCase):

  def test_raises_on_non_tuple_parameter(self):
    lam = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32)
    )
    with self.assertRaises(tree_transformations.ParameterSelectionError):
      form_utils._as_function_of_some_federated_subparameters(lam, [(0,)])

  def test_raises_on_selection_from_non_tuple(self):
    lam = building_blocks.Lambda(
        'x', [tf.int32], building_blocks.Reference('x', [tf.int32])
    )
    with self.assertRaises(tree_transformations.ParameterSelectionError):
      form_utils._as_function_of_some_federated_subparameters(lam, [(0, 0)])

  def test_raises_on_non_federated_selection(self):
    lam = building_blocks.Lambda(
        'x', [tf.int32], building_blocks.Reference('x', [tf.int32])
    )
    with self.assertRaises(form_utils._NonFederatedSelectionError):
      form_utils._as_function_of_some_federated_subparameters(lam, [(0,)])

  def test_raises_on_selections_at_different_placements(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server]
    )
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0
        ),
    )
    with self.assertRaises(form_utils._MismatchedSelectionPlacementError):
      form_utils._as_function_of_some_federated_subparameters(lam, [(0,), (1,)])

  def test_single_element_selection(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server]
    )
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0
        ),
    )

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0,)]
    )
    expected_parameter_type = computation_types.at_clients((tf.int32,))
    type_test_utils.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(
            expected_parameter_type, lam.result.type_signature
        ),
    )

  def test_single_named_element_selection(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [('a', fed_at_clients), ('b', fed_at_server)]
    )
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), name='a'
        ),
    )

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0,)]
    )
    expected_parameter_type = computation_types.at_clients((tf.int32,))
    type_test_utils.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(
            expected_parameter_type, lam.result.type_signature
        ),
    )

  def test_single_element_selection_leaves_no_unbound_references(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server]
    )
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0
        ),
    )
    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0,)]
    )
    unbound_references = transformation_utils.get_map_of_unbound_references(
        new_lam
    )[new_lam]
    self.assertEmpty(unbound_references)

  def test_single_nested_element_selection(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [[fed_at_clients], fed_at_server]
    )
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Selection(
                building_blocks.Reference('x', tuple_of_federated_types),
                index=0,
            ),
            index=0,
        ),
    )

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0, 0)]
    )
    expected_parameter_type = computation_types.at_clients((tf.int32,))
    type_test_utils.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(
            expected_parameter_type, lam.result.type_signature
        ),
    )

  def test_multiple_nested_element_selection(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [[fed_at_clients], fed_at_server, [fed_at_clients]]
    )
    first_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0
        ),
        index=0,
    )
    second_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=2
        ),
        index=0,
    )
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Struct([first_selection, second_selection]),
    )

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0, 0), (2, 0)]
    )

    expected_parameter_type = computation_types.at_clients((tf.int32, tf.int32))
    type_test_utils.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(
            expected_parameter_type, lam.result.type_signature
        ),
    )

  def test_multiple_nested_named_element_selection(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType([
        ('a', [('a', fed_at_clients)]),
        ('b', fed_at_server),
        ('c', [('c', fed_at_clients)]),
    ])
    first_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), name='a'
        ),
        name='a',
    )
    second_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), name='c'
        ),
        name='c',
    )
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Struct([first_selection, second_selection]),
    )

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0, 0), (2, 0)]
    )

    expected_parameter_type = computation_types.at_clients((tf.int32, tf.int32))
    type_test_utils.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(
            expected_parameter_type, lam.result.type_signature
        ),
    )

  def test_binding_multiple_args_results_in_unique_names(self):
    fed_at_clients = computation_types.FederatedType(
        tf.int32, placements.CLIENTS
    )
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [[fed_at_clients], fed_at_server, [fed_at_clients]]
    )
    first_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0
        ),
        index=0,
    )
    second_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=2
        ),
        index=0,
    )
    lam = building_blocks.Lambda(
        'x',
        tuple_of_federated_types,
        building_blocks.Struct([first_selection, second_selection]),
    )
    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0, 0), (2, 0)]
    )
    tree_analysis.check_has_unique_names(new_lam)


if __name__ == '__main__':
  # The test execution context replaces all secure intrinsics with insecure
  # reductions.
  execution_contexts.set_sync_test_cpp_execution_context()
  tf.test.main()
