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

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test as common_test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.mapreduce import canonical_form
from tensorflow_federated.python.core.backends.mapreduce import canonical_form_utils
from tensorflow_federated.python.core.backends.mapreduce import test_utils
from tensorflow_federated.python.core.backends.mapreduce import transformations as mapreduce_transformations
from tensorflow_federated.python.core.impl import transformations as impl_transformations
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances
from tensorflow_federated.python.core.utils import computation_utils


def get_iterative_process_for_sum_example():
  """Returns an iterative process for a sum example."""

  @computations.federated_computation
  def init_fn():
    """The `init` function for `computation_utils.IterativeProcess`."""
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
    """The `next` function for `computation_utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates, client_output = intrinsics.federated_map(work, c3)
    federated_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.secure_sum(client_updates[1], 8)
    s6 = intrinsics.federated_zip(
        [server_state, [federated_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return computation_utils.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_server_state():
  """Returns an iterative process for a sum example."""

  @computations.federated_computation
  def init_fn():
    """The `init` function for `computation_utils.IterativeProcess`."""
    return intrinsics.federated_value([], placements.SERVER)

  @computations.tf_computation(tf.int32)
  def work(client_data):
    del client_data  # Unused
    return [1, 1], []

  @computations.tf_computation([tf.int32, tf.int32])
  def update(global_update):
    return [], global_update

  @computations.federated_computation([
      computation_types.FederatedType([], placements.SERVER),
      computation_types.FederatedType(tf.int32, placements.CLIENTS),
  ])
  def next_fn(server_state, client_data):
    """The `next` function for `computation_utils.IterativeProcess`."""
    del server_state  # Unused
    client_updates, client_output = intrinsics.federated_map(work, client_data)
    federated_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.secure_sum(client_updates[1], 8)
    s5 = intrinsics.federated_zip([federated_update, secure_update])
    new_server_state, server_output = intrinsics.federated_map(update, s5)
    return new_server_state, server_output, client_output

  return computation_utils.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_client_output():
  """Returns an iterative process for a sum example."""

  @computations.federated_computation
  def init_fn():
    """The `init` function for `computation_utils.IterativeProcess`."""
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
    """The `next` function for `computation_utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates = intrinsics.federated_map(work, c3)
    federated_update = intrinsics.federated_sum(client_updates[0])
    secure_update = intrinsics.secure_sum(client_updates[1], 8)
    s6 = intrinsics.federated_zip(
        [server_state, [federated_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output

  return computation_utils.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_federated_aggregate():
  """Returns an iterative process for a sum example."""

  @computations.federated_computation
  def init_fn():
    """The `init` function for `computation_utils.IterativeProcess`."""
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
    """The `next` function for `computation_utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates, client_output = intrinsics.federated_map(work, c3)
    secure_update = intrinsics.secure_sum(client_updates, 8)
    s6 = intrinsics.federated_zip([server_state, secure_update])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return computation_utils.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_secure_sum():
  """Returns an iterative process for a sum example."""

  @computations.federated_computation
  def init_fn():
    """The `init` function for `computation_utils.IterativeProcess`."""
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
    """The `next` function for `computation_utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    client_updates, client_output = intrinsics.federated_map(work, c3)
    federated_update = intrinsics.federated_sum(client_updates)
    s6 = intrinsics.federated_zip([server_state, federated_update])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return computation_utils.IterativeProcess(init_fn, next_fn)


def get_iterative_process_for_sum_example_with_no_aggregation():
  """Returns an iterative process for a sum example."""

  @computations.federated_computation
  def init_fn():
    """The `init` function for `computation_utils.IterativeProcess`."""
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
    """The `next` function for `computation_utils.IterativeProcess`."""
    s2 = intrinsics.federated_map(prepare, server_state)
    client_input = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([client_data, client_input])
    _, client_output = intrinsics.federated_map(work, c3)
    federated_update = intrinsics.federated_value(1, placements.SERVER)
    secure_update = intrinsics.federated_value(1, placements.SERVER)
    s6 = intrinsics.federated_zip(
        [server_state, [federated_update, secure_update]])
    new_server_state, server_output = intrinsics.federated_map(update, s6)
    return new_server_state, server_output, client_output

  return computation_utils.IterativeProcess(init_fn, next_fn)


class CanonicalFormTestCase(common_test.TestCase):
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


class CreateNextWithFakeClientOutputTest(common_test.TestCase):

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


class CreateBeforeAndAfterBroadcastForNoBroadcastTest(common_test.TestCase):

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


class CreateBeforeAndAfterAggregateForNoFederatedAggregateTest(
    common_test.TestCase):

  def test_returns_tree(self):
    ip = get_iterative_process_for_sum_example_with_no_federated_aggregate()
    next_tree = building_blocks.ComputationBuildingBlock.from_proto(
        ip.next._computation_proto)

    before_aggregate, after_aggregate = canonical_form_utils._create_before_and_after_aggregate_for_no_federated_aggregate(
        next_tree)

    before_secure_sum, after_secure_sum = (
        mapreduce_transformations.force_align_and_split_by_intrinsics(
            next_tree, [intrinsic_defs.SECURE_SUM.uri]))
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

    self.assertEqual(before_aggregate.result[1].formatted_representation(),
                     before_secure_sum.result.formatted_representation())

    self.assertIsInstance(after_aggregate, building_blocks.Lambda)
    self.assertIsInstance(after_aggregate.result, building_blocks.Call)
    actual_tree, _ = impl_transformations.uniquify_reference_names(
        after_aggregate.result.function)
    expected_tree, _ = impl_transformations.uniquify_reference_names(
        after_secure_sum)
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


class CreateBeforeAndAfterAggregateForNoSecureSumTest(common_test.TestCase):

  def test_returns_tree(self):
    ip = get_iterative_process_for_sum_example_with_no_secure_sum()
    next_tree = building_blocks.ComputationBuildingBlock.from_proto(
        ip.next._computation_proto)
    next_tree = canonical_form_utils._replace_intrinsics_with_bodies(next_tree)

    before_aggregate, after_aggregate = canonical_form_utils._create_before_and_after_aggregate_for_no_secure_sum(
        next_tree)

    before_federated_aggregate, after_federated_aggregate = (
        mapreduce_transformations.force_align_and_split_by_intrinsics(
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
    actual_tree, _ = impl_transformations.uniquify_reference_names(
        after_aggregate.result.function)
    expected_tree, _ = impl_transformations.uniquify_reference_names(
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


class GetCanonicalFormForIterativeProcessTest(CanonicalFormTestCase):

  def test_next_computation_returning_tensor_fails_well(self):
    cf = test_utils.get_temperature_sensor_example()
    it = canonical_form_utils.get_iterative_process_for_canonical_form(cf)
    init_result = it.initialize.type_signature.result
    lam = building_blocks.Lambda('x', init_result,
                                 building_blocks.Reference('x', init_result))
    bad_it = computation_utils.IterativeProcess(
        it.initialize,
        computation_wrapper_instances.building_block_to_computation(lam))
    with self.assertRaisesRegex(TypeError,
                                'instances of `tff.NamedTupleType`.'):
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
    not_reducible_it = computation_utils.IterativeProcess(
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

  def test_returns_canonical_form_with_next_fn_returning_call_directly(self):

    @computations.federated_computation
    def init_fn():
      return intrinsics.federated_value(42, placements.SERVER)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER),
        computation_types.FederatedType(
            computation_types.SequenceType(tf.float32), placements.CLIENTS))
    def next_fn(server_state, client_data):
      broadcast_state = intrinsics.federated_broadcast(server_state)

      @computations.tf_computation(tf.int32,
                                   computation_types.SequenceType(tf.float32))
      @tf.function
      def some_transform(x, y):
        del y  # Unused
        return x + 1

      client_update = intrinsics.federated_map(some_transform,
                                               (broadcast_state, client_data))
      aggregate_update = intrinsics.federated_sum(client_update)
      server_output = intrinsics.federated_value(1234, placements.SERVER)
      return aggregate_update, server_output

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER),
        computation_types.FederatedType(
            computation_types.SequenceType(tf.float32), placements.CLIENTS))
    def nested_next_fn(server_state, client_data):
      return next_fn(server_state, client_data)

    iterative_process = computation_utils.IterativeProcess(
        init_fn, nested_next_fn)
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(
        iterative_process)
    self.assertIsInstance(cf, canonical_form.CanonicalForm)

  def test_returns_canonical_form_with_unused_federated_arg(self):
    example_iterative_process = test_utils.get_unused_lambda_arg_iterative_process(
    )
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(
        example_iterative_process)
    self.assertIsInstance(cf, canonical_form.CanonicalForm)

  def test_returns_canonical_form_with_unused_tf_func_arg(self):
    example_iterative_process = test_utils.get_unused_tf_computation_arg_iterative_process(
    )
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(
        example_iterative_process)
    self.assertIsInstance(cf, canonical_form.CanonicalForm)

  def test_returns_canonical_form_with_no_broadcast(self):

    @computations.tf_computation(tf.int32)
    @tf.function
    def map_fn(client_val):
      del client_val  # unused
      return 1

    @computations.federated_computation
    def init_fn():
      return intrinsics.federated_value(False, placements.SERVER)

    @computations.federated_computation(
        computation_types.FederatedType(tf.bool, placements.SERVER),
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def next_fn(server_val, client_val):
      del server_val  # Unused
      result_on_clients = intrinsics.federated_map(map_fn, client_val)
      aggregated_result = intrinsics.federated_sum(result_on_clients)
      side_output = intrinsics.federated_value(False, placements.SERVER)
      return side_output, aggregated_result

    ip = computation_utils.IterativeProcess(init_fn, next_fn)
    cf = canonical_form_utils.get_canonical_form_for_iterative_process(ip)
    self.assertIsInstance(cf, canonical_form.CanonicalForm)


INIT_TYPE = computation_types.FunctionType(None, tf.float32)
S1_TYPE = computation_types.FederatedType(INIT_TYPE.result, placements.SERVER)
C1_TYPE = computation_types.FederatedType(tf.float32, placements.CLIENTS)
S6_TYPE = computation_types.FederatedType(tf.float64, placements.SERVER)
S7_TYPE = computation_types.FederatedType(tf.bool, placements.SERVER)
C6_TYPE = computation_types.FederatedType(tf.int64, placements.CLIENTS)
S2_TYPE = computation_types.FederatedType([tf.float32], placements.SERVER)
C2_TYPE = computation_types.FederatedType(S2_TYPE.member, placements.CLIENTS)
C5_TYPE = computation_types.FederatedType([tf.float64], placements.CLIENTS)
ZERO_TYPE = computation_types.TensorType(tf.int64)
ACCUMULATE_TYPE = computation_types.FunctionType([ZERO_TYPE, C5_TYPE.member],
                                                 ZERO_TYPE)
MERGE_TYPE = computation_types.FunctionType([ZERO_TYPE, ZERO_TYPE], ZERO_TYPE)
REPORT_TYPE = computation_types.FunctionType(ZERO_TYPE, tf.int64)
S3_TYPE = computation_types.FederatedType(REPORT_TYPE.result, placements.SERVER)


def _create_next_type_with_s1_type(x):
  param_type = computation_types.NamedTupleType([x, C1_TYPE])
  result_type = computation_types.NamedTupleType([S6_TYPE, S7_TYPE, C6_TYPE])
  return computation_types.FunctionType(param_type, result_type)


def _create_before_broadcast_type_with_s1_type(x):
  return computation_types.FunctionType(
      computation_types.NamedTupleType([x, C1_TYPE]), S2_TYPE)


def _create_before_aggregate_with_c2_type(x):
  return computation_types.FunctionType(
      [[S1_TYPE, C1_TYPE], x],
      [C5_TYPE, ZERO_TYPE, ACCUMULATE_TYPE, MERGE_TYPE, REPORT_TYPE])


def _create_after_aggregate_with_s3_type(x):
  return computation_types.FunctionType([[[S1_TYPE, C1_TYPE], C2_TYPE], x],
                                        [S6_TYPE, S7_TYPE, C6_TYPE])


class TypeCheckTest(CanonicalFormTestCase):

  def test_init_raises_non_federated_type(self):
    with self.assertRaisesRegex(TypeError, 'init'):
      canonical_form_utils.pack_initialize_comp_type_signature(tf.float32)

  def test_init_passes_with_float_at_server(self):
    cf_types = canonical_form_utils.pack_initialize_comp_type_signature(
        computation_types.FederatedType(tf.float32, placements.SERVER))
    self.assertEqual(cf_types['initialize_type'], INIT_TYPE)

  def test_next_succeeds_match_with_init_type(self):
    cf_types = {'initialize_type': INIT_TYPE}
    next_type = _create_next_type_with_s1_type(S1_TYPE)
    packed_types = canonical_form_utils.pack_next_comp_type_signature(
        next_type, cf_types)
    # Checking contents of the returned dict.
    self.assertEqual(packed_types['s1_type'], S1_TYPE)
    self.assertEqual(packed_types['c1_type'], C1_TYPE)
    self.assertEqual(packed_types['s6_type'], S6_TYPE)
    self.assertEqual(packed_types['s7_type'], S7_TYPE)
    self.assertEqual(packed_types['c6_type'], C6_TYPE)

  def test_next_fails_mismatch_with_init_type(self):
    cf_types = {'initialize_type': INIT_TYPE}
    next_type = _create_next_type_with_s1_type(
        computation_types.FederatedType(tf.int32, placements.SERVER))
    with self.assertRaisesRegex(TypeError, 'next'):
      canonical_form_utils.pack_next_comp_type_signature(next_type, cf_types)

  def test_before_broadcast_succeeds_match_with_next_type(self):
    cf_types = {'initialize_type': INIT_TYPE}
    next_type = _create_next_type_with_s1_type(S1_TYPE)
    cf_types = canonical_form_utils.pack_next_comp_type_signature(
        next_type, cf_types)
    good_before_broadcast_type = _create_before_broadcast_type_with_s1_type(
        S1_TYPE)
    packed_types = (
        canonical_form_utils.check_and_pack_before_broadcast_type_signature(
            good_before_broadcast_type, cf_types))
    # Checking contents of the returned dict.
    self.assertEqual(
        packed_types['s2_type'],
        computation_types.FederatedType(C2_TYPE.member, placements.SERVER))
    self.assertEqual(
        packed_types['prepare_type'],
        computation_types.FunctionType(S1_TYPE.member, S2_TYPE.member))

  def test_before_broadcast_fails_mismatch_with_next_type(self):
    cf_types = {'initialize_type': INIT_TYPE}
    next_type = _create_next_type_with_s1_type(S1_TYPE)
    cf_types = canonical_form_utils.pack_next_comp_type_signature(
        next_type, cf_types)
    bad_before_broadcast_type = _create_before_broadcast_type_with_s1_type(
        computation_types.FederatedType(tf.int32, placements.SERVER))
    with self.assertRaisesRegex(TypeError, 'before_broadcast'):
      canonical_form_utils.check_and_pack_before_broadcast_type_signature(
          bad_before_broadcast_type, cf_types)

  def test_before_aggregate_succeeds_and_packs(self):
    cf_types = {'initialize_type': INIT_TYPE}
    next_type = _create_next_type_with_s1_type(S1_TYPE)
    good_before_broadcast_type = _create_before_broadcast_type_with_s1_type(
        S1_TYPE)
    cf_types = canonical_form_utils.pack_next_comp_type_signature(
        next_type, cf_types)
    cf_types = (
        canonical_form_utils.check_and_pack_before_broadcast_type_signature(
            good_before_broadcast_type, cf_types))
    good_before_aggregate_type = _create_before_aggregate_with_c2_type(C2_TYPE)
    packed_types = (
        canonical_form_utils.check_and_pack_before_aggregate_type_signature(
            good_before_aggregate_type, cf_types))

    # Checking contents of the returned dict.
    self.assertEqual(packed_types['c5_type'], C5_TYPE)
    self.assertEqual(packed_types['zero_type'].result, ZERO_TYPE)
    self.assertEqual(packed_types['accumulate_type'], ACCUMULATE_TYPE)
    self.assertEqual(packed_types['merge_type'], MERGE_TYPE)
    self.assertEqual(packed_types['report_type'], REPORT_TYPE)

  def test_before_aggregate_fails_mismatch_with_before_broadcast_type(self):
    cf_types = {'initialize_type': INIT_TYPE}
    next_type = _create_next_type_with_s1_type(S1_TYPE)
    good_before_broadcast_type = _create_before_broadcast_type_with_s1_type(
        S1_TYPE)
    cf_types = canonical_form_utils.pack_next_comp_type_signature(
        next_type, cf_types)
    cf_types = (
        canonical_form_utils.check_and_pack_before_broadcast_type_signature(
            good_before_broadcast_type, cf_types))
    bad_before_aggregate_type = _create_before_aggregate_with_c2_type(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    with self.assertRaisesRegex(TypeError, 'before_aggregate'):
      canonical_form_utils.check_and_pack_before_aggregate_type_signature(
          bad_before_aggregate_type, cf_types)

  def test_after_aggregate_succeeds_and_packs(self):
    good_init_type = computation_types.FederatedType(tf.float32,
                                                     placements.SERVER)
    cf_types = canonical_form_utils.pack_initialize_comp_type_signature(
        good_init_type)
    next_type = _create_next_type_with_s1_type(S1_TYPE)
    good_before_broadcast_type = _create_before_broadcast_type_with_s1_type(
        S1_TYPE)
    cf_types = canonical_form_utils.pack_next_comp_type_signature(
        next_type, cf_types)
    cf_types = (
        canonical_form_utils.check_and_pack_before_broadcast_type_signature(
            good_before_broadcast_type, cf_types))
    good_before_aggregate_type = _create_before_aggregate_with_c2_type(C2_TYPE)
    cf_types = (
        canonical_form_utils.check_and_pack_before_aggregate_type_signature(
            good_before_aggregate_type, cf_types))
    good_after_aggregate_type = _create_after_aggregate_with_s3_type(S3_TYPE)
    packed_types = (
        canonical_form_utils.check_and_pack_after_aggregate_type_signature(
            good_after_aggregate_type, cf_types))
    # Checking contents of the returned dict.
    self.assertEqual(
        packed_types['s4_type'],
        computation_types.FederatedType([S1_TYPE.member, S3_TYPE.member],
                                        placements.SERVER))
    self.assertEqual(
        packed_types['c3_type'],
        computation_types.FederatedType([C1_TYPE.member, C2_TYPE.member],
                                        placements.CLIENTS))
    self.assertEqual(
        packed_types['update_type'],
        computation_types.FunctionType(packed_types['s4_type'].member,
                                       packed_types['s5_type'].member))

  def test_after_aggregate_raises_mismatch_with_before_aggregate(self):
    good_init_type = computation_types.FederatedType(tf.float32,
                                                     placements.SERVER)
    cf_types = canonical_form_utils.pack_initialize_comp_type_signature(
        good_init_type)
    next_type = _create_next_type_with_s1_type(
        computation_types.FederatedType(tf.float32, placements.SERVER))
    good_before_broadcast_type = _create_before_broadcast_type_with_s1_type(
        computation_types.FederatedType(tf.float32, placements.SERVER))
    cf_types = canonical_form_utils.pack_next_comp_type_signature(
        next_type, cf_types)
    cf_types = (
        canonical_form_utils.check_and_pack_before_broadcast_type_signature(
            good_before_broadcast_type, cf_types))
    good_before_aggregate_type = _create_before_aggregate_with_c2_type(C2_TYPE)
    cf_types = (
        canonical_form_utils.check_and_pack_before_aggregate_type_signature(
            good_before_aggregate_type, cf_types))
    bad_after_aggregate_type = _create_after_aggregate_with_s3_type(
        computation_types.FederatedType(tf.int32, placements.SERVER))

    with self.assertRaisesRegex(TypeError, 'after_aggregate'):
      canonical_form_utils.check_and_pack_after_aggregate_type_signature(
          bad_after_aggregate_type, cf_types)

if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  common_test.main()
