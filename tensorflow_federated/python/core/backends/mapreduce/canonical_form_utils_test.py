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

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.mapreduce import canonical_form
from tensorflow_federated.python.core.backends.mapreduce import canonical_form_utils
from tensorflow_federated.python.core.backends.mapreduce import test_utils
from tensorflow_federated.python.core.backends.mapreduce import transformations
from tensorflow_federated.python.core.backends.reference import reference_context
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances
from tensorflow_federated.python.core.templates import iterative_process


def get_iterative_process_for_sum_example():
  """Returns an iterative process for a sum example.

  This iterative process contains all the components required to compile to
  `canonical_form.CanonicalForm`.
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
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
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
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
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
    secure_update = intrinsics.federated_secure_sum(client_updates, 8)
    s6 = intrinsics.federated_zip([server_state, secure_update])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_federated_secure_sum():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_secure_sum`.
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
    # No call to `federated_secure_sum`.
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
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
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
    secure_update = intrinsics.federated_secure_sum(client_updates[1], 8)
    s5 = intrinsics.federated_zip([unsecure_update, secure_update])
    # Empty server state.
    new_server_state = intrinsics.federated_value([], placements.SERVER)
    server_output = intrinsics.federated_map(update, s5)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_aggregation():
  """Returns an iterative process for a sum example.

  This iterative process does not have a call to `federated_aggregate` or
  `federated_secure_sum` and as a result it should fail to compile to
  `canonical_form.CanonicalForm`.
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
    # No call to `federated_secure_sum`.
    secure_update = intrinsics.federated_value(1, placements.SERVER)
    s6 = intrinsics.federated_zip(
        [server_state, [unsecure_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return iterative_process.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_minimal_sum_example():
  """Returns an iterative process for a sum example.

  This iterative process contains the fewest components required to compile to
  `canonical_form.CanonicalForm`.
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
    self.assertAllEqual(state, collections.OrderedDict(num_rounds=0))

    state, metrics = it.next(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertAllEqual(state, collections.OrderedDict(num_rounds=1))
    self.assertAllClose(metrics,
                        collections.OrderedDict(ratio_over_threshold=0.5))

    state, metrics = it.next(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllClose(metrics,
                        collections.OrderedDict(ratio_over_threshold=0.75))


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
    self.assertIsInstance(before_aggregate.result, building_blocks.Struct)
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

    self.assertTrue(
        tree_analysis.trees_equal(before_aggregate.result[1],
                                  before_federated_secure_sum.result))

    self.assertIsInstance(after_aggregate, building_blocks.Lambda)
    self.assertIsInstance(after_aggregate.result, building_blocks.Call)
    actual_tree, _ = tree_transformations.uniquify_reference_names(
        after_aggregate.result.function)
    expected_tree, _ = tree_transformations.uniquify_reference_names(
        after_federated_secure_sum)
    self.assertTrue(tree_analysis.trees_equal(actual_tree, expected_tree))

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
    self.assertIsInstance(before_aggregate.result, building_blocks.Struct)
    self.assertLen(before_aggregate.result, 2)

    # trees_equal will fail if computations refer to unbound references, so we
    # create a new dummy computation to bind them.
    unbound_refs_in_before_agg_result = transformation_utils.get_map_of_unbound_references(
        before_aggregate.result[0])[before_aggregate.result[0]]
    unbound_refs_in_before_fed_agg_result = transformation_utils.get_map_of_unbound_references(
        before_federated_aggregate.result)[before_federated_aggregate.result]

    dummy_data = building_blocks.Data('data',
                                      computation_types.AbstractType('T'))

    blk_binding_refs_in_before_agg = building_blocks.Block(
        [(name, dummy_data) for name in unbound_refs_in_before_agg_result],
        before_aggregate.result[0])
    blk_binding_refs_in_before_fed_agg = building_blocks.Block(
        [(name, dummy_data) for name in unbound_refs_in_before_fed_agg_result],
        before_federated_aggregate.result)

    self.assertTrue(
        tree_analysis.trees_equal(blk_binding_refs_in_before_agg,
                                  blk_binding_refs_in_before_fed_agg))

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

    self.assertTrue(
        tree_analysis.trees_equal(after_aggregate.result.function,
                                  after_federated_aggregate))

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
    # refactoring the code involved in compiling an
    # `tff.templates.IterativeProcess` into a
    # `tff.backends.mapreduce.CanonicalForm`. If you are sure this needs to be
    # updated, one recommendation is to print 'k=\'v\',' while iterating over
    # the k-v pairs of the ordereddict.
    # pyformat: disable
    expected = collections.OrderedDict(
        initialize_type='( -> <int32,int32>)',
        s1_type='<int32,int32>@SERVER',
        c1_type='{int32}@CLIENTS',
        prepare_type='(<int32,int32> -> <int32,int32>)',
        s2_type='<int32,int32>@SERVER',
        c2_type='<int32,int32>@CLIENTS',
        c3_type='{<int32,<int32,int32>>}@CLIENTS',
        work_type='(<int32,<int32,int32>> -> <int32,int32>)',
        c4_type='{<int32,int32>}@CLIENTS',
        c5_type='{int32}@CLIENTS',
        c6_type='{int32}@CLIENTS',
        zero_type='( -> int32)',
        accumulate_type='(<int32,int32> -> int32)',
        merge_type='(<int32,int32> -> int32)',
        report_type='(int32 -> int32)',
        s3_type='int32@SERVER',
        bitwidth_type='( -> int32)',
        s4_type='int32@SERVER',
        s5_type='<int32,int32>@SERVER',
        s6_type='<<int32,int32>,<int32,int32>>@SERVER',
        update_type='(<<int32,int32>,<int32,int32>> -> <<int32,int32>,<>>)',
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
      canonical_form_utils.get_canonical_form_for_iterative_process(
          not_reducible_it)

  def test_constructs_canonical_form_from_mnist_training_example(self):
    it = canonical_form_utils.get_iterative_process_for_canonical_form(
        test_utils.get_mnist_training_example())
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(it)
    self.assertIsInstance(cf, canonical_form.CanonicalForm)

  def test_temperature_example_round_trip(self):
    # NOTE: the roundtrip through CanonicalForm->IterProc->CanonicalForm seems
    # to lose the python container annotations on the StructType.
    it = canonical_form_utils.get_iterative_process_for_canonical_form(
        test_utils.get_temperature_sensor_example())
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(it)
    new_it = canonical_form_utils.get_iterative_process_for_canonical_form(cf)
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
            test_utils.computation_to_building_block(it.next)),
        tree_analysis.count_tensorflow_variables_under(
            test_utils.computation_to_building_block(new_it.next)))

  def test_mnist_training_round_trip(self):
    it = canonical_form_utils.get_iterative_process_for_canonical_form(
        test_utils.get_mnist_training_example())
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(it)
    new_it = canonical_form_utils.get_iterative_process_for_canonical_form(cf)
    state1 = it.initialize()
    state2 = new_it.initialize()
    self.assertAllClose(state1, state2)
    dummy_x = np.array([[0.5] * 784], dtype=np.float32)
    dummy_y = np.array([1], dtype=np.int32)
    client_data = [collections.OrderedDict(x=dummy_x, y=dummy_y)]
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
            test_utils.computation_to_building_block(it.next)),
        tree_analysis.count_tensorflow_variables_under(
            test_utils.computation_to_building_block(new_it.next)))

  # pyformat: disable
  @parameterized.named_parameters(
      ('sum_example',
       get_iterative_process_for_sum_example()),
      ('sum_example_with_no_prepare',
       get_iterative_process_for_sum_example_with_no_prepare()),
      ('sum_example_with_no_broadcast',
       get_iterative_process_for_sum_example_with_no_broadcast()),
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

  # pyformat: disable
  @parameterized.named_parameters(
      ('sum_example',
       get_iterative_process_for_sum_example()),
      ('sum_example_with_no_prepare',
       get_iterative_process_for_sum_example_with_no_prepare()),
      ('sum_example_with_no_broadcast',
       get_iterative_process_for_sum_example_with_no_broadcast()),
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
  def test_returns_canonical_form_with_grappler_disabled(self, ip):
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(ip, None)

    self.assertIsInstance(cf, canonical_form.CanonicalForm)

  def test_raises_value_error_for_sum_example_with_no_aggregation(self):
    ip = get_iterative_process_for_sum_example_with_no_aggregation()

    with self.assertRaisesRegex(
        ValueError,
        r'Expected .* containing at least one `federated_aggregate` or '
        r'`federated_secure_sum`'):
      canonical_form_utils.get_canonical_form_for_iterative_process(ip)

  def test_returns_canonical_form_with_indirection_to_intrinsic(self):
    self.skipTest('b/160865930')
    ip = test_utils.get_iterative_process_for_example_with_lambda_returning_aggregation(
    )

    cf = canonical_form_utils.get_canonical_form_for_iterative_process(ip)

    self.assertIsInstance(cf, canonical_form.CanonicalForm)


if __name__ == '__main__':
  # The reference context is used here because it is currently the only context
  # which implements the `tff.federated_secure_sum` intrinsic.
  reference_context.set_reference_context()
  test.main()
