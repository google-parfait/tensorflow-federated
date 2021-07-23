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

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.mapreduce import form_utils
from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.backends.mapreduce import test_utils as mapreduce_test_utils
from tensorflow_federated.python.core.backends.reference import reference_context
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances
from tensorflow_federated.python.core.templates import iterative_process


def get_iterative_process_for_sum_example():
  """Returns an iterative process for a sum example.

  This iterative process contains all the components required to compile to
  `forms.MapReduceForm`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, 1

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
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
        client_updates[1], 8)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_with_nested_broadcasts():
  """Returns an iterative process with nested federated broadcasts.

  This iterative process contains all the components required to compile to
  `forms.MapReduceForm`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, 1

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation(
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER))
  def broadcast_and_return_arg_and_result(x):
    broadcasted = intrinsics.federated_broadcast(x)
    return [broadcasted, x]

  @computations.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.templates.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    unused_client_input, to_broadcast = broadcast_and_return_arg_and_result(s2)
    client_input = intrinsics.federated_broadcast(to_broadcast)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum_bitwidth(
        client_updates[1], 8)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_prepare():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_map` with a prepare
  function before the `federated_broadcast`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, 1

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
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
        client_updates[1], 8)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_broadcast():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_broadcast`. As a
  result, this iterative process does not have a call to `federated_map` with a
  prepare function before the `federated_broadcast`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return 1, 1

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
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
        client_updates[1], 8)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_federated_aggregate():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_aggregate`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value(0, placements.SERVER)

  @computations.tf_computation(tf.int32)
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, tf.int32)
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1

  @computations.tf_computation([tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
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


def get_iterative_process_for_sum_example_with_no_federated_secure_sum_bitwidth(
):
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to
  `federated_secure_sum_bitwidth`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value(0, placements.SERVER)

  @computations.tf_computation(tf.int32)
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, tf.int32)
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1

  @computations.tf_computation([tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
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

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, 1

  @computations.federated_computation([
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
        client_updates[1], 8)
    new_server_state = intrinsics.federated_zip(
        [unsecure_update, secure_update])
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

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([], placements.SERVER)

  @computations.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return 1, 1

  @computations.tf_computation([tf.int32, tf.int32])
  def update(global_update):
    return global_update

  @computations.federated_computation([
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
        client_updates[1], 8)
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

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
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
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_minimal_sum_example():
  """Returns an iterative process for a sum example.

  This iterative process contains the fewest components required to compile to
  `forms.MapReduceForm`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.templates.IterativeProcess`."""
    zero = computations.tf_computation(lambda: [0, 0])
    return intrinsics.federated_eval(zero, placements.SERVER)

  @computations.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return 1, 1

  @computations.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
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
        client_updates[1], 8)
    new_server_state = intrinsics.federated_zip(
        [unsecure_update, secure_update])
    # No call to `federated_map` with an `update` function.
    server_output = intrinsics.federated_value([], placements.SERVER)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_example_cf_compatible_iterative_processes():
  # pyformat: disable
  return (
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
       mapreduce_test_utils.get_iterative_process_for_example_with_unused_tf_computation_arg()))
  # pyformat: enable


class MapReduceFormTestCase(test_case.TestCase):
  """A base class that overrides evaluate to handle various executors."""

  def evaluate(self, value):
    if tf.is_tensor(value):
      return super().evaluate(value)
    elif isinstance(value, (np.ndarray, np.number)):
      return value
    else:
      raise TypeError('Cannot evaluate value of type `{!s}`.'.format(
          type(value)))


class GetIterativeProcessForMapReduceFormTest(MapReduceFormTestCase):

  def test_with_temperature_sensor_example(self):
    mrf = mapreduce_test_utils.get_temperature_sensor_example()
    it = form_utils.get_iterative_process_for_map_reduce_form(mrf)

    state = it.initialize()
    self.assertAllEqual(state, collections.OrderedDict(num_rounds=0))

    state, metrics = it.next(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertAllEqual(state, collections.OrderedDict(num_rounds=1))
    self.assertAllClose(metrics,
                        collections.OrderedDict(ratio_over_threshold=0.5))

    state, metrics = it.next(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllClose(metrics,
                        collections.OrderedDict(ratio_over_threshold=0.75))


class CheckMapReduceFormCompatibleWithIterativeProcessTest(
    MapReduceFormTestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      *get_example_cf_compatible_iterative_processes())
  def test_allows_valid_computation(self, ip):
    form_utils.check_iterative_process_compatible_with_map_reduce_form(ip)

  def test_disallows_broadcast_dependent_on_aggregate(self):

    @computations.federated_computation
    def init_comp():
      return intrinsics.federated_value(0, placements.SERVER)

    @computations.federated_computation(
        computation_types.at_server(tf.int32), computation_types.at_clients(()))
    def next_comp(server_state, client_data):
      del server_state, client_data
      client_val = intrinsics.federated_value(0, placements.CLIENTS)
      server_agg = intrinsics.federated_sum(client_val)
      # This broadcast is dependent on the result of the above aggregation,
      # which is not supported by MapReduce form.
      broadcasted = intrinsics.federated_broadcast(server_agg)
      server_agg_again = intrinsics.federated_sum(broadcasted)
      # `next` must return two values.
      return server_agg_again, intrinsics.federated_value((), placements.SERVER)

    ip = iterative_process.IterativeProcess(init_comp, next_comp)

    with self.assertRaises(ValueError):
      form_utils.check_iterative_process_compatible_with_map_reduce_form(ip)


class GetMapReduceFormForIterativeProcessTest(MapReduceFormTestCase,
                                              parameterized.TestCase):

  def test_next_computation_returning_tensor_fails_well(self):
    mrf = mapreduce_test_utils.get_temperature_sensor_example()
    it = form_utils.get_iterative_process_for_map_reduce_form(mrf)
    init_result = it.initialize.type_signature.result
    lam = building_blocks.Lambda('x', init_result,
                                 building_blocks.Reference('x', init_result))
    bad_it = iterative_process.IterativeProcess(
        it.initialize,
        computation_wrapper_instances.building_block_to_computation(lam))
    with self.assertRaises(TypeError):
      form_utils.get_map_reduce_form_for_iterative_process(bad_it)

  def test_broadcast_dependent_on_aggregate_fails_well(self):
    mrf = mapreduce_test_utils.get_temperature_sensor_example()
    it = form_utils.get_iterative_process_for_map_reduce_form(mrf)
    next_comp = it.next.to_building_block()
    top_level_param = building_blocks.Reference(next_comp.parameter_name,
                                                next_comp.parameter_type)
    first_result = building_blocks.Call(next_comp, top_level_param)
    middle_param = building_blocks.Struct([
        building_blocks.Selection(first_result, index=0),
        building_blocks.Selection(top_level_param, index=1)
    ])
    second_result = building_blocks.Call(next_comp, middle_param)
    not_reducible = building_blocks.Lambda(next_comp.parameter_name,
                                           next_comp.parameter_type,
                                           second_result)
    not_reducible_it = iterative_process.IterativeProcess(
        it.initialize,
        computation_wrapper_instances.building_block_to_computation(
            not_reducible))

    with self.assertRaisesRegex(ValueError, 'broadcast dependent on aggregate'):
      form_utils.get_map_reduce_form_for_iterative_process(not_reducible_it)

  def test_gets_map_reduce_form_for_nested_broadcast(self):
    ip = get_iterative_process_with_nested_broadcasts()
    mrf = form_utils.get_map_reduce_form_for_iterative_process(ip)
    self.assertIsInstance(mrf, forms.MapReduceForm)

  def test_constructs_map_reduce_form_from_mnist_training_example(self):
    it = form_utils.get_iterative_process_for_map_reduce_form(
        mapreduce_test_utils.get_mnist_training_example())
    mrf = form_utils.get_map_reduce_form_for_iterative_process(it)
    self.assertIsInstance(mrf, forms.MapReduceForm)

  def test_temperature_example_round_trip(self):
    # NOTE: the roundtrip through MapReduceForm->IterProc->MapReduceForm seems
    # to lose the python container annotations on the StructType.
    it = form_utils.get_iterative_process_for_map_reduce_form(
        mapreduce_test_utils.get_temperature_sensor_example())
    mrf = form_utils.get_map_reduce_form_for_iterative_process(it)
    new_it = form_utils.get_iterative_process_for_map_reduce_form(mrf)
    state = new_it.initialize()
    self.assertEqual(state.num_rounds, 0)

    state, metrics = new_it.next(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertEqual(state.num_rounds, 1)
    self.assertAllClose(metrics,
                        collections.OrderedDict(ratio_over_threshold=0.5))

    state, metrics = new_it.next(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllClose(metrics,
                        collections.OrderedDict(ratio_over_threshold=0.75))
    self.assertEqual(
        tree_analysis.count_tensorflow_variables_under(
            it.next.to_building_block()),
        tree_analysis.count_tensorflow_variables_under(
            new_it.next.to_building_block()))

  def test_mnist_training_round_trip(self):
    it = form_utils.get_iterative_process_for_map_reduce_form(
        mapreduce_test_utils.get_mnist_training_example())
    mrf = form_utils.get_map_reduce_form_for_iterative_process(it)
    new_it = form_utils.get_iterative_process_for_map_reduce_form(mrf)
    state1 = it.initialize()
    state2 = new_it.initialize()
    self.assertAllClose(state1, state2)
    whimsy_x = np.array([[0.5] * 784], dtype=np.float32)
    whimsy_y = np.array([1], dtype=np.int32)
    client_data = [collections.OrderedDict(x=whimsy_x, y=whimsy_y)]
    round_1 = it.next(state1, [client_data])
    state = round_1[0]
    metrics = round_1[1]
    alt_round_1 = new_it.next(state2, [client_data])
    alt_state = alt_round_1[0]
    self.assertAllClose(state, alt_state)
    alt_metrics = alt_round_1[1]
    self.assertAllClose(metrics, alt_metrics)
    self.assertEqual(
        tree_analysis.count_tensorflow_variables_under(
            it.next.to_building_block()),
        tree_analysis.count_tensorflow_variables_under(
            new_it.next.to_building_block()))

  @parameterized.named_parameters(
      *get_example_cf_compatible_iterative_processes())
  def test_returns_map_reduce_form(self, ip):
    mrf = form_utils.get_map_reduce_form_for_iterative_process(ip)

    self.assertIsInstance(mrf, forms.MapReduceForm)

  @parameterized.named_parameters(
      *get_example_cf_compatible_iterative_processes())
  def test_returns_canonical_form_with_grappler_disabled(self, ip):
    grappler_config = tf.compat.v1.ConfigProto()
    grappler_config.graph_options.rewrite_options.disable_meta_optimizer = True
    mrf = form_utils.get_map_reduce_form_for_iterative_process(
        ip, grappler_config)

    self.assertIsInstance(mrf, forms.MapReduceForm)

  def test_returns_map_reduce_form_for_sum_example_with_no_aggregation(self):
    ip = get_iterative_process_for_sum_example_with_no_aggregation()
    mrf = form_utils.get_map_reduce_form_for_iterative_process(ip)
    self.assertIsInstance(mrf, forms.MapReduceForm)

  def test_returns_map_reduce_form_with_indirection_to_intrinsic(self):
    ip = mapreduce_test_utils.get_iterative_process_for_example_with_lambda_returning_aggregation(
    )

    mrf = form_utils.get_map_reduce_form_for_iterative_process(ip)

    self.assertIsInstance(mrf, forms.MapReduceForm)


class BroadcastFormTest(test_case.TestCase):

  def test_roundtrip(self):
    add = computations.tf_computation(lambda x, y: x + y)
    server_data_type = computation_types.at_server(tf.int32)
    client_data_type = computation_types.at_clients(tf.int32)

    @computations.federated_computation(server_data_type, client_data_type)
    def add_server_number_plus_one(server_number, client_numbers):
      one = intrinsics.federated_value(1, placements.SERVER)
      server_context = intrinsics.federated_map(add, (one, server_number))
      client_context = intrinsics.federated_broadcast(server_context)
      return intrinsics.federated_map(add, (client_context, client_numbers))

    bf = form_utils.get_broadcast_form_for_computation(
        add_server_number_plus_one)
    self.assertEqual(bf.server_data_label, 'server_number')
    self.assertEqual(bf.client_data_label, 'client_numbers')
    self.assert_types_equivalent(
        bf.compute_server_context.type_signature,
        computation_types.FunctionType(tf.int32, (tf.int32,)))
    self.assertEqual(2, bf.compute_server_context(1)[0])
    self.assert_types_equivalent(
        bf.client_processing.type_signature,
        computation_types.FunctionType(((tf.int32,), tf.int32), tf.int32))
    self.assertEqual(3, bf.client_processing((1,), 2))

    round_trip_comp = form_utils.get_computation_for_broadcast_form(bf)
    self.assert_types_equivalent(round_trip_comp.type_signature,
                                 add_server_number_plus_one.type_signature)
    # 2 (server data) + 1 (constant in comp) + 2 (client data) = 5 (output)
    self.assertEqual([5, 6, 7], round_trip_comp(2, [2, 3, 4]))

  def test_roundtrip_no_broadcast(self):
    add_five = computations.tf_computation(lambda x: x + 5)
    server_data_type = computation_types.at_server(())
    client_data_type = computation_types.at_clients(tf.int32)

    @computations.federated_computation(server_data_type, client_data_type)
    def add_five_at_clients(naught_at_server, client_numbers):
      del naught_at_server
      return intrinsics.federated_map(add_five, client_numbers)

    bf = form_utils.get_broadcast_form_for_computation(add_five_at_clients)
    self.assertEqual(bf.server_data_label, 'naught_at_server')
    self.assertEqual(bf.client_data_label, 'client_numbers')
    self.assert_types_equivalent(bf.compute_server_context.type_signature,
                                 computation_types.FunctionType((), ()))
    self.assert_types_equivalent(
        bf.client_processing.type_signature,
        computation_types.FunctionType(((), tf.int32), tf.int32))
    self.assertEqual(6, bf.client_processing((), 1))

    round_trip_comp = form_utils.get_computation_for_broadcast_form(bf)
    self.assert_types_equivalent(round_trip_comp.type_signature,
                                 add_five_at_clients.type_signature)
    self.assertEqual([10, 11, 12], round_trip_comp((), [5, 6, 7]))


class AsFunctionOfSingleSubparameterTest(test_case.TestCase):

  def assert_selected_param_to_result_type(self, old_lam, new_lam, index):
    old_type = old_lam.type_signature
    new_type = new_lam.type_signature
    old_type.check_function()
    new_type.check_function()
    self.assert_types_equivalent(
        new_type,
        computation_types.FunctionType(old_type.parameter[index],
                                       old_type.result))

  def test_selection(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0))
    new_lam = form_utils._as_function_of_single_subparameter(lam, 0)
    self.assert_selected_param_to_result_type(lam, new_lam, 0)

  def test_named_element_selection(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType([
        (None, fed_at_server),
        ('a', fed_at_clients),
    ])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), name='a'))
    new_lam = form_utils._as_function_of_single_subparameter(lam, 1)
    self.assert_selected_param_to_result_type(lam, new_lam, 1)


class AsFunctionOfSomeSubparametersTest(test_case.TestCase):

  def test_raises_on_non_tuple_parameter(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    with self.assertRaises(form_utils._ParameterSelectionError):
      form_utils._as_function_of_some_federated_subparameters(lam, [(0,)])

  def test_raises_on_selection_from_non_tuple(self):
    lam = building_blocks.Lambda('x', [tf.int32],
                                 building_blocks.Reference('x', [tf.int32]))
    with self.assertRaises(form_utils._ParameterSelectionError):
      form_utils._as_function_of_some_federated_subparameters(lam, [(0, 0)])

  def test_raises_on_non_federated_selection(self):
    lam = building_blocks.Lambda('x', [tf.int32],
                                 building_blocks.Reference('x', [tf.int32]))
    with self.assertRaises(form_utils._NonFederatedSelectionError):
      form_utils._as_function_of_some_federated_subparameters(lam, [(0,)])

  def test_raises_on_selections_at_different_placements(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0))
    with self.assertRaises(form_utils._MismatchedSelectionPlacementError):
      form_utils._as_function_of_some_federated_subparameters(lam, [(0,), (1,)])

  def test_single_element_selection(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0))

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0,)])
    expected_parameter_type = computation_types.at_clients((tf.int32,))
    self.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(expected_parameter_type,
                                       lam.result.type_signature))

  def test_single_named_element_selection(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType([
        ('a', fed_at_clients), ('b', fed_at_server)
    ])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), name='a'))

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0,)])
    expected_parameter_type = computation_types.at_clients((tf.int32,))
    self.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(expected_parameter_type,
                                       lam.result.type_signature))

  def test_single_element_selection_leaves_no_unbound_references(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0))
    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0,)])
    unbound_references = transformation_utils.get_map_of_unbound_references(
        new_lam)[new_lam]
    self.assertEmpty(unbound_references)

  def test_single_nested_element_selection(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType([[fed_at_clients],
                                                             fed_at_server])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Selection(
                building_blocks.Reference('x', tuple_of_federated_types),
                index=0),
            index=0))

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0, 0)])
    expected_parameter_type = computation_types.at_clients((tf.int32,))
    self.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(expected_parameter_type,
                                       lam.result.type_signature))

  def test_multiple_nested_element_selection(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType([[fed_at_clients],
                                                             fed_at_server,
                                                             [fed_at_clients]])
    first_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0),
        index=0)
    second_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=2),
        index=0)
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Struct([first_selection, second_selection]))

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0, 0), (2, 0)])

    expected_parameter_type = computation_types.at_clients((tf.int32, tf.int32))
    self.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(expected_parameter_type,
                                       lam.result.type_signature))

  def test_multiple_nested_named_element_selection(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType([
        ('a', [('a', fed_at_clients)]), ('b', fed_at_server),
        ('c', [('c', fed_at_clients)])
    ])
    first_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), name='a'),
        name='a')
    second_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), name='c'),
        name='c')
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Struct([first_selection, second_selection]))

    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0, 0), (2, 0)])

    expected_parameter_type = computation_types.at_clients((tf.int32, tf.int32))
    self.assert_types_equivalent(
        new_lam.type_signature,
        computation_types.FunctionType(expected_parameter_type,
                                       lam.result.type_signature))

  def test_binding_multiple_args_results_in_unique_names(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType([[fed_at_clients],
                                                             fed_at_server,
                                                             [fed_at_clients]])
    first_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0),
        index=0)
    second_selection = building_blocks.Selection(
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=2),
        index=0)
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Struct([first_selection, second_selection]))
    new_lam = form_utils._as_function_of_some_federated_subparameters(
        lam, [(0, 0), (2, 0)])
    tree_analysis.check_has_unique_names(new_lam)


if __name__ == '__main__':
  # The reference context is used here because it is currently the only context
  # which implements the `tff.federated_secure_sum_bitwidth` intrinsic.
  reference_context.set_reference_context()
  test_case.main()
