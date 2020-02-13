# Lint as: python3
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

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.mapreduce import canonical_form
from tensorflow_federated.python.core.backends.mapreduce import canonical_form_utils
from tensorflow_federated.python.core.backends.mapreduce import test_utils
from tensorflow_federated.python.core.backends.mapreduce import transformations
from tensorflow_federated.python.core.impl import reference_executor
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances
from tensorflow_federated.python.core.templates import iterative_process

tf.compat.v1.enable_v2_behavior()


def get_iterative_process_for_sum_example():
  """Returns an iterative process for a sum example.

  This iterative process contains all the components required to compile to
  `canonical_form.CanonicalForm`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.utils.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return [1, 1], []

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates, client_output = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_prepare():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_map` with a prepare
  function before the `federated_broadcast`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.utils.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return [1, 1], []

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    # No call to `federated_map` with a `prepare` function.
    client_input = intrinsics.federated_broadcast(server_state)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates, client_output = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_broadcast():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_broadcast`. As a
  result, this iterative process does not have a call to `federated_map` with a
  prepare function before the `federated_broadcast`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.utils.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return [1, 1], []

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    # No call to `federated_map` with prepare.
    # No call to `federated_broadcast`.
    client_updates, client_output = intrinsics.federated_map(work, client_data)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_client_output():
  """Returns an iterative process for a sum example.

  This iterative process does not return client output from the work function.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.utils.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return [1, 1]

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    # No client output.
    client_updates = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
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
    """The `init` function for `tff.utils.IterativeProcess`."""
    return intrinsics.federated_value(0, placements.SERVER)

  @computations.tf_computation(tf.int32)
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, tf.int32)
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, []

  @computations.tf_computation([tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
      computation_types.FederatedType(tf.int32, placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates, client_output = intrinsics.federated_map(work, c3)
    # No call to `federated_aggregate`.
    secure_update = intrinsics.federated_secure_sum(client_updates, 8)
    s6 = intrinsics.federated_zip([server_state, secure_update])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_federated_secure_sum():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_secure_sum`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.utils.IterativeProcess`."""
    return intrinsics.federated_value(0, placements.SERVER)

  @computations.tf_computation(tf.int32)
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, tf.int32)
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return 1, []

  @computations.tf_computation([tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
      computation_types.FederatedType(tf.int32, placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates, client_output = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates)
    # No call to `federated_secure_sum`.
    s6 = intrinsics.federated_zip([server_state, unsecure_update])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_update():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_map` with a prepare
  function before the `federated_broadcast`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.utils.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return [1, 1], []

  @computations.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates, client_output = intrinsics.federated_map(work, c3)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
    new_server_state = intrinsics.federated_zip(
        [unsecure_update, secure_update])
    # No call to `federated_map` with an `update` function.
    server_output = intrinsics.federated_value([], placements.SERVER)
    return new_server_state, server_output, client_output

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
    """The `init` function for `tff.utils.IterativeProcess`."""
    return intrinsics.federated_value([], placements.SERVER)

  @computations.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return [1, 1], []

  @computations.tf_computation([tf.int32, tf.int32])
  def update(global_update):
    return global_update

  @computations.federated_computation([
      computation_types.FederatedType([], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    del server_state  # Unused
    # No call to `federated_map` with prepare.
    # No call to `federated_broadcast`.
    client_updates, client_output = intrinsics.federated_map(work, client_data)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
    s5 = intrinsics.federated_zip([unsecure_update, secure_update])
    # Empty server state.
    new_server_state = intrinsics.federated_value([], placements.SERVER)
    server_output = intrinsics.federated_map(update, s5)
    return new_server_state, server_output, client_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_aggregation():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_aggregate` or
  `federated_secure_sum` and as a result it should fail to compile to
  `canonical_form.CanonicalForm`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.utils.IterativeProcess`."""
    return intrinsics.federated_value([0, 0], placements.SERVER)

  @computations.tf_computation([tf.int32, tf.int32])
  def prepare(server_state):
    return server_state

  @computations.tf_computation(tf.int32, [tf.int32, tf.int32])
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return [1, 1], []

  @computations.tf_computation([tf.int32, tf.int32], [tf.int32, tf.int32])
  def update(server_state, global_update):
    del server_state  # Unused
    return global_update, []

  @computations.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    _, client_output = intrinsics.federated_map(work, c3)
    # No call to `federated_aggregate`.
    unsecure_update = intrinsics.federated_value(1, placements.SERVER)
    # No call to `federated_secure_sum`.
    secure_update = intrinsics.federated_value(1, placements.SERVER)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_minimal_sum_example():
  """Returns an iterative process for a sum example.

  This iterative process contains the fewest components required to compile to
  `canonical_form.CanonicalForm`.
  """

  @computations.federated_computation
  def init_fn():
    """The `init` function for `tff.utils.IterativeProcess`."""
    zero = computations.tf_computation(lambda: [0, 0])
    return intrinsics.federated_eval(zero, placements.SERVER)

  @computations.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return [1, 1]

  @computations.federated_computation([
      computation_types.FederatedType([tf.int32, tf.int32], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `tff.utils.IterativeProcess`."""
    del server_state  # Unused
    # No call to `federated_map` with prepare.
    # No call to `federated_broadcast`.
    client_updates = intrinsics.federated_map(work, client_data)
    unsecure_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
    new_server_state = intrinsics.federated_zip(
        [unsecure_update, secure_update])
    # No call to `federated_map` with an `update` function.
    server_output = intrinsics.federated_value([], placements.SERVER)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


class CanonicalFormTestCase(test.TestCase):
  """A base class that overrides evaluate to handle various executors."""

  def evaluate(self, value):
    if tf.is_tensor(value):
      return super().evaluate(value)
    elif isinstance(value, (np.ndarray, np.number)):
      return value
    else:
      raise TypeError('Cannot evaluate value of type `{!s}`.'.format(
          type(value)))


class GetIterativeProcessForCanonicalFormTest(CanonicalFormTestCase):

  def test_with_temperature_sensor_example(self):
    cf = test_utils.get_temperature_sensor_example()
    it = canonical_form_utils.get_iterative_process_for_canonical_form(cf)

    state = it.initialize()
    self.assertLen(state, 1)
    self.assertAllEqual(anonymous_tuple.name_list(state), ['num_rounds'])
    self.assertEqual(state[0], 0)

    state, metrics, stats = it.next(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertLen(state, 1)
    self.assertAllEqual(anonymous_tuple.name_list(state), ['num_rounds'])
    self.assertEqual(state[0], 1)
    self.assertLen(metrics, 1)
    self.assertAllEqual(
        anonymous_tuple.name_list(metrics), ['ratio_over_threshold'])
    self.assertEqual(metrics[0], 0.5)
    self.assertCountEqual([self.evaluate(x.num_readings) for x in stats],
                          [1, 3])

    state, metrics, stats = it.next(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllEqual(state, (2,))
    self.assertAllClose(metrics, {'ratio_over_threshold': 0.75})
    self.assertCountEqual([x.num_readings for x in stats], [1, 1, 1, 1])


class CreateNextWithFakeClientOutputTest(test.TestCase):

  def test_returns_tree(self):
    ip = get_iterative_process_for_sum_example_with_no_client_output()
    old_next_tree = building_blocks.ComputationBuildingBlock.from_proto(
        ip.next._computation_proto)

    new_next_tree = canonical_form_utils._create_next_with_fake_client_output(
        old_next_tree)

    self.assertIsInstance(new_next_tree, building_blocks.Lambda)
    self.assertIsInstance(new_next_tree.result, building_blocks.Tuple)
    self.assertLen(new_next_tree.result, 3)
    self.assertEqual(new_next_tree.result[0].formatted_representation(),
                     old_next_tree.result[0].formatted_representation())
    self.assertEqual(new_next_tree.result[1].formatted_representation(),
                     old_next_tree.result[1].formatted_representation())

    # pyformat: disable
    self.assertEqual(
        new_next_tree.result[2].formatted_representation(),
        'federated_value_at_clients(<>)'
    )
    # pyformat: enable


class CreateBeforeAndAfterBroadcastForNoBroadcastTest(test.TestCase):

  def test_returns_tree(self):
    ip = get_iterative_process_for_sum_example_with_no_server_state()
    next_tree = building_blocks.ComputationBuildingBlock.from_proto(
        ip.next._computation_proto)

    before_broadcast, after_broadcast = canonical_form_utils._create_before_and_after_broadcast_for_no_broadcast(
        next_tree)

    # pyformat: disable
    self.assertEqual(
        before_broadcast.formatted_representation(),
        '(_var1 -> federated_value_at_server(<>))'
    )
    # pyformat: enable

    self.assertIsInstance(after_broadcast, building_blocks.Lambda)
    self.assertIsInstance(after_broadcast.result, building_blocks.Call)
    self.assertEqual(after_broadcast.result.function.formatted_representation(),
                     next_tree.formatted_representation())

    # pyformat: disable
    self.assertEqual(
        after_broadcast.result.argument.formatted_representation(),
        '_var2[0]'
    )
    # pyformat: enable


class CreateBeforeAndAfterAggregateForNoFederatedAggregateTest(test.TestCase):

  def test_returns_tree(self):
    ip = get_iterative_process_for_sum_example_with_no_federated_aggregate()
    next_tree = building_blocks.ComputationBuildingBlock.from_proto(
        ip.next._computation_proto)

    before_aggregate, after_aggregate = canonical_form_utils._create_before_and_after_aggregate_for_no_federated_aggregate(
        next_tree)

    before_federated_secure_sum, after_federated_secure_sum = (
        transformations.force_align_and_split_by_intrinsics(
            next_tree, [intrinsic_defs.FEDERATED_SECURE_SUM.uri]))
    self.assertIsInstance(before_aggregate, building_blocks.Lambda)
    self.assertIsInstance(before_aggregate.result, building_blocks.Tuple)
    self.assertLen(before_aggregate.result, 2)

    # pyformat: disable
    self.assertEqual(
        before_aggregate.result[0].formatted_representation(),
        '<\n'
        '  federated_value_at_clients(<>),\n'
        '  <>,\n'
        '  (_var1 -> <>),\n'
        '  (_var2 -> <>),\n'
        '  (_var3 -> <>)\n'
        '>'
    )
    # pyformat: enable

    self.assertEqual(
        before_aggregate.result[1].formatted_representation(),
        before_federated_secure_sum.result.formatted_representation())

    self.assertIsInstance(after_aggregate, building_blocks.Lambda)
    self.assertIsInstance(after_aggregate.result, building_blocks.Call)
    actual_tree, _ = tree_transformations.uniquify_reference_names(
        after_aggregate.result.function)
    expected_tree, _ = tree_transformations.uniquify_reference_names(
        after_federated_secure_sum)
    self.assertEqual(actual_tree.formatted_representation(),
                     expected_tree.formatted_representation())

    # pyformat: disable
    self.assertEqual(
        after_aggregate.result.argument.formatted_representation(),
        '<\n'
        '  _var4[0],\n'
        '  _var4[1][1]\n'
        '>'
    )
    # pyformat: enable


class CreateBeforeAndAfterAggregateForNoSecureSumTest(test.TestCase):

  def test_returns_tree(self):
    ip = get_iterative_process_for_sum_example_with_no_federated_secure_sum()
    next_tree = building_blocks.ComputationBuildingBlock.from_proto(
        ip.next._computation_proto)
    next_tree = canonical_form_utils._replace_intrinsics_with_bodies(next_tree)

    before_aggregate, after_aggregate = canonical_form_utils._create_before_and_after_aggregate_for_no_federated_secure_sum(
        next_tree)

    before_federated_aggregate, after_federated_aggregate = (
        transformations.force_align_and_split_by_intrinsics(
            next_tree, [intrinsic_defs.FEDERATED_AGGREGATE.uri]))
    self.assertIsInstance(before_aggregate, building_blocks.Lambda)
    self.assertIsInstance(before_aggregate.result, building_blocks.Tuple)
    self.assertLen(before_aggregate.result, 2)
    self.assertEqual(
        before_aggregate.result[0].formatted_representation(),
        before_federated_aggregate.result.formatted_representation())

    # pyformat: disable
    self.assertEqual(
        before_aggregate.result[1].formatted_representation(),
        '<\n'
        '  federated_value_at_clients(<>),\n'
        '  <>\n'
        '>'
    )
    # pyformat: enable

    self.assertIsInstance(after_aggregate, building_blocks.Lambda)
    self.assertIsInstance(after_aggregate.result, building_blocks.Call)
    actual_tree, _ = tree_transformations.uniquify_reference_names(
        after_aggregate.result.function)
    expected_tree, _ = tree_transformations.uniquify_reference_names(
        after_federated_aggregate)
    self.assertEqual(actual_tree.formatted_representation(),
                     expected_tree.formatted_representation())

    # pyformat: disable
    self.assertEqual(
        after_aggregate.result.argument.formatted_representation(),
        '<\n'
        '  _var1[0],\n'
        '  _var1[1][0]\n'
        '>'
    )
    # pyformat: enable


class GetTypeInfoTest(test.TestCase):

  def test_returns_type_info_for_sum_example(self):
    ip = get_iterative_process_for_sum_example()
    initialize_tree = building_blocks.ComputationBuildingBlock.from_proto(
        ip.initialize._computation_proto)
    next_tree = building_blocks.ComputationBuildingBlock.from_proto(
        ip.next._computation_proto)
    initialize_tree = canonical_form_utils._replace_intrinsics_with_bodies(
        initialize_tree)
    next_tree = canonical_form_utils._replace_intrinsics_with_bodies(next_tree)
    before_broadcast, after_broadcast = (
        transformations.force_align_and_split_by_intrinsics(
            next_tree, [intrinsic_defs.FEDERATED_BROADCAST.uri]))
    before_aggregate, after_aggregate = (
        transformations.force_align_and_split_by_intrinsics(
            after_broadcast, [
                intrinsic_defs.FEDERATED_AGGREGATE.uri,
                intrinsic_defs.FEDERATED_SECURE_SUM.uri,
            ]))

    type_info = canonical_form_utils._get_type_info(initialize_tree,
                                                    before_broadcast,
                                                    after_broadcast,
                                                    before_aggregate,
                                                    after_aggregate)

    actual = collections.OrderedDict([
        (label, type_signature.compact_representation())
        for label, type_signature in type_info.items()
    ])
    # Note: THE CONTENTS OF THIS DICTIONARY IS NOT IMPORTANT. The purpose of
    # this test is not to assert that this value returned by
    # `canonical_form_utils._get_type_info`, but instead to act as a signal when
    # refactoring the code involved in compiling an `tff.utils.IterativeProcess`
    # into a `tff.backends.mapreduce.CanonicalForm`. If you are sure this needs
    # to be updated, one recommendation is to print k=\'v\' while iterating
    # over the k-v pairs of the ordereddict.
    # pyformat: disable
    expected = collections.OrderedDict(
        initialize_type='( -> <int32,int32>)',
        s1_type='<int32,int32>@SERVER',
        c1_type='{int32}@CLIENTS',
        prepare_type='(<int32,int32> -> <<int32,int32>>)',
        s2_type='<<int32,int32>>@SERVER',
        c2_type='<<int32,int32>>@CLIENTS',
        c3_type='{<int32,<<int32,int32>>>}@CLIENTS',
        work_type='(<int32,<<int32,int32>>> -> <<<int32,int32,int32,int32,int32,int32>,<int32,int32,int32,int32,int32,int32>>,<>>)',
        c4_type='{<<<int32,int32,int32,int32,int32,int32>,<int32,int32,int32,int32,int32,int32>>,<>>}@CLIENTS',
        c5_type='{<<int32,int32,int32,int32,int32,int32>,<int32,int32,int32,int32,int32,int32>>}@CLIENTS',
        c6_type='{<int32,int32,int32,int32,int32,int32>}@CLIENTS',
        c7_type='{<int32,int32,int32,int32,int32,int32>}@CLIENTS',
        c8_type='{<>}@CLIENTS',
        zero_type='( -> <int32,int32,int32,int32,int32,int32>)',
        accumulate_type='(<<int32,int32,int32,int32,int32,int32>,<int32,int32,int32,int32,int32,int32>> -> <int32,int32,int32,int32,int32,int32>)',
        merge_type='(<<int32,int32,int32,int32,int32,int32>,<int32,int32,int32,int32,int32,int32>> -> <int32,int32,int32,int32,int32,int32>)',
        report_type='(<int32,int32,int32,int32,int32,int32> -> <int32,int32,int32,int32,int32,int32>)',
        s3_type='<int32,int32,int32,int32,int32,int32>@SERVER',
        bitwidth_type='( -> <int32,int32,int32,int32,int32,int32>)',
        s4_type='<int32,int32,int32,int32,int32,int32>@SERVER',
        s5_type='<<int32,int32,int32,int32,int32,int32>,<int32,int32,int32,int32,int32,int32>>@SERVER',
        s6_type='<<int32,int32>,<<int32,int32,int32,int32,int32,int32>,<int32,int32,int32,int32,int32,int32>>>@SERVER',
        update_type='(<<int32,int32>,<<int32,int32,int32,int32,int32,int32>,<int32,int32,int32,int32,int32,int32>>> -> <<int32,int32>,<>>)',
        s7_type='<<int32,int32>,<>>@SERVER',
        s8_type='<int32,int32>@SERVER',
        s9_type='<>@SERVER',
        )
    # pyformat: enable

    items = zip(actual.items(), expected.items())
    for (actual_key, actual_value), (expected_key, expected_value) in items:
      self.assertEqual(actual_key, expected_key)
      self.assertEqual(
          actual_value, expected_value,
          'The value of \'{}\' is not equal to the expected value'.format(
              actual_key))


class GetCanonicalFormForIterativeProcessTest(CanonicalFormTestCase,
                                              parameterized.TestCase):

  def test_next_computation_returning_tensor_fails_well(self):
    cf = test_utils.get_temperature_sensor_example()
    it = canonical_form_utils.get_iterative_process_for_canonical_form(cf)
    init_result = it.initialize.type_signature.result
    lam = building_blocks.Lambda('x', init_result,
                                 building_blocks.Reference('x', init_result))
    bad_it = iterative_process.IterativeProcess(
        it.initialize,
        computation_wrapper_instances.building_block_to_computation(lam))
    with self.assertRaises(TypeError):
      canonical_form_utils.get_canonical_form_for_iterative_process(bad_it)

  def test_broadcast_dependent_on_aggregate_fails_well(self):
    cf = test_utils.get_temperature_sensor_example()
    it = canonical_form_utils.get_iterative_process_for_canonical_form(cf)
    next_comp = test_utils.computation_to_building_block(it.next)
    top_level_param = building_blocks.Reference(next_comp.parameter_name,
                                                next_comp.parameter_type)
    first_result = building_blocks.Call(next_comp, top_level_param)
    middle_param = building_blocks.Tuple([
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
      canonical_form_utils.get_canonical_form_for_iterative_process(
          not_reducible_it)

  def test_constructs_canonical_form_from_mnist_training_example(self):
    it = canonical_form_utils.get_iterative_process_for_canonical_form(
        test_utils.get_mnist_training_example())
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(it)
    self.assertIsInstance(cf, canonical_form.CanonicalForm)

  def test_temperature_example_round_trip(self):
    it = canonical_form_utils.get_iterative_process_for_canonical_form(
        test_utils.get_temperature_sensor_example())
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(it)
    new_it = canonical_form_utils.get_iterative_process_for_canonical_form(cf)
    state = new_it.initialize()
    self.assertLen(state, 1)
    self.assertAllEqual(anonymous_tuple.name_list(state), ['num_rounds'])
    self.assertEqual(state[0], 0)

    state, metrics, stats = new_it.next(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertLen(state, 1)
    self.assertAllEqual(anonymous_tuple.name_list(state), ['num_rounds'])
    self.assertEqual(state[0], 1)
    self.assertLen(metrics, 1)
    self.assertAllEqual(
        anonymous_tuple.name_list(metrics), ['ratio_over_threshold'])
    self.assertEqual(metrics[0], 0.5)
    self.assertCountEqual([self.evaluate(x.num_readings) for x in stats],
                          [1, 3])

    state, metrics, stats = new_it.next(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllEqual(state, (2,))
    self.assertAllClose(metrics, {'ratio_over_threshold': 0.75})
    self.assertCountEqual([x.num_readings for x in stats], [1, 1, 1, 1])

  def test_mnist_training_round_trip(self):
    it = canonical_form_utils.get_iterative_process_for_canonical_form(
        test_utils.get_mnist_training_example())
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(it)
    new_it = canonical_form_utils.get_iterative_process_for_canonical_form(cf)
    state1 = it.initialize()
    state2 = new_it.initialize()
    self.assertEqual(str(state1), str(state2))
    dummy_x = np.array([[0.5] * 784], dtype=np.float32)
    dummy_y = np.array([1], dtype=np.int32)
    client_data = [collections.OrderedDict(x=dummy_x, y=dummy_y)]
    round_1 = it.next(state1, [client_data])
    state = round_1[0]
    metrics = round_1[1]
    alt_round_1 = new_it.next(state2, [client_data])
    alt_state = alt_round_1[0]
    alt_metrics = alt_round_1[1]
    self.assertAllEqual(
        anonymous_tuple.name_list(state), anonymous_tuple.name_list(alt_state))
    self.assertAllEqual(
        anonymous_tuple.name_list(metrics),
        anonymous_tuple.name_list(alt_metrics))
    self.assertAllClose(state, alt_state)
    self.assertAllClose(metrics, alt_metrics)

  def test_returns_canonical_form_from_tff_learning_structure(self):
    it = test_utils.construct_example_training_comp()
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(it)
    new_it = canonical_form_utils.get_iterative_process_for_canonical_form(cf)
    self.assertIsInstance(cf, canonical_form.CanonicalForm)
    self.assertEqual(it.initialize.type_signature,
                     new_it.initialize.type_signature)
    # Notice next type_signatures need not be equal, since we may have appended
    # an empty tuple as client side-channel outputs if none existed
    self.assertEqual(it.next.type_signature.parameter,
                     new_it.next.type_signature.parameter)
    self.assertEqual(it.next.type_signature.result[0],
                     new_it.next.type_signature.result[0])
    self.assertEqual(it.next.type_signature.result[1],
                     new_it.next.type_signature.result[1])

    state1 = it.initialize()
    state2 = new_it.initialize()

    sample_batch = collections.OrderedDict(
        x=np.array([[1., 1.]], dtype=np.float32),
        y=np.array([[0]], dtype=np.int32))
    client_data = [sample_batch]

    round_1 = it.next(state1, [client_data])
    state = round_1[0]
    state_names = anonymous_tuple.name_list(state)
    state_arrays = anonymous_tuple.flatten(state)
    metrics = round_1[1]
    metrics_names = [x[0] for x in anonymous_tuple.iter_elements(metrics)]
    metrics_arrays = anonymous_tuple.flatten(metrics)

    alt_round_1 = new_it.next(state2, [client_data])
    alt_state = alt_round_1[0]
    alt_state_names = anonymous_tuple.name_list(alt_state)
    alt_state_arrays = anonymous_tuple.flatten(alt_state)
    alt_metrics = alt_round_1[1]
    alt_metrics_names = [
        x[0] for x in anonymous_tuple.iter_elements(alt_metrics)
    ]
    alt_metrics_arrays = anonymous_tuple.flatten(alt_metrics)

    self.assertEmpty(state.delta_aggregate_state)
    self.assertEmpty(state.model_broadcast_state)
    self.assertAllEqual(state_names, alt_state_names)
    self.assertAllEqual(metrics_names, alt_metrics_names)
    self.assertAllClose(state_arrays, alt_state_arrays)
    self.assertAllClose(metrics_arrays[:2], alt_metrics_arrays[:2])
    # Final metric is execution time
    self.assertAlmostEqual(metrics_arrays[2], alt_metrics_arrays[2], delta=1e-3)

  # pyformat: disable
  @parameterized.named_parameters(
      ('sum_example',
       get_iterative_process_for_sum_example()),
      ('sum_example_with_no_prepare',
       get_iterative_process_for_sum_example_with_no_prepare()),
      ('sum_example_with_no_broadcast',
       get_iterative_process_for_sum_example_with_no_broadcast()),
      ('sum_example_with_no_client_output',
       get_iterative_process_for_sum_example_with_no_client_output()),
      ('sum_example_with_no_federated_aggregate',
       get_iterative_process_for_sum_example_with_no_federated_aggregate()),
      ('sum_example_with_no_federated_secure_sum',
       get_iterative_process_for_sum_example_with_no_federated_secure_sum()),
      ('sum_example_with_no_update',
       get_iterative_process_for_sum_example_with_no_update()),
      ('sum_example_with_no_server_state',
       get_iterative_process_for_sum_example_with_no_server_state()),
      ('minimal_sum_example',
       get_iterative_process_for_minimal_sum_example()),
      ('example_with_unused_lambda_arg',
       test_utils.get_iterative_process_for_example_with_unused_lambda_arg()),
      ('example_with_unused_tf_computation_arg',
       test_utils.get_iterative_process_for_example_with_unused_tf_computation_arg()),
  )
  # pyformat: enable
  def test_returns_canonical_form(self, ip):
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(ip)

    self.assertIsInstance(cf, canonical_form.CanonicalForm)

  def test_raises_value_error_for_sum_example_with_no_aggregation(self):
    ip = get_iterative_process_for_sum_example_with_no_aggregation()

    with self.assertRaises(ValueError):
      canonical_form_utils.get_canonical_form_for_iterative_process(ip)


if __name__ == '__main__':
  reference_executor = reference_executor.ReferenceExecutor()
  with context_stack_impl.context_stack.install(reference_executor):
    test.main()
