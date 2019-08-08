# Lint as: python2, python3
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
"""Tests for canonical_form_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest

import numpy as np
import tensorflow as tf

import tensorflow_federated as tff

from tensorflow_federated.python.core.backends.mapreduce import canonical_form_utils
from tensorflow_federated.python.core.backends.mapreduce import test_utils


class CanonicalFormUtilsTest(absltest.TestCase):

  def test_broadcast_dependent_on_aggregate_fails_well(self):
    cf = test_utils.get_temperature_sensor_example()
    it = tff.backends.mapreduce.get_iterative_process_for_canonical_form(cf)
    next_comp = test_utils.computation_to_building_block(it.next)
    top_level_param = tff.framework.Reference(next_comp.parameter_name,
                                              next_comp.parameter_type)
    first_result = tff.framework.Call(next_comp, top_level_param)
    middle_param = tff.framework.Tuple([
        tff.framework.Selection(first_result, index=0),
        tff.framework.Selection(top_level_param, index=1)
    ])
    second_result = tff.framework.Call(next_comp, middle_param)
    not_reducible = tff.framework.Lambda(next_comp.parameter_name,
                                         next_comp.parameter_type,
                                         second_result)
    not_reducible_it = tff.utils.IterativeProcess(
        it.initialize,
        tff.framework.building_block_to_computation(not_reducible))

    with self.assertRaisesRegex(ValueError, 'broadcast dependent on aggregate'):
      tff.backends.mapreduce.get_canonical_form_for_iterative_process(
          not_reducible_it)

  def test_get_iterative_process_for_canonical_form(self):
    cf = test_utils.get_temperature_sensor_example()
    it = tff.backends.mapreduce.get_iterative_process_for_canonical_form(cf)

    state = it.initialize()
    self.assertEqual(str(state), '<num_rounds=0>')

    state, metrics, stats = it.next(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertEqual(str(state), '<num_rounds=1>')
    self.assertEqual(str(metrics), '<ratio_over_threshold=0.5>')
    self.assertCountEqual([x.num_readings for x in stats], [1, 3])

    state, metrics, stats = it.next(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertEqual(str(state), '<num_rounds=2>')
    self.assertEqual(str(metrics), '<ratio_over_threshold=0.75>')
    self.assertCountEqual([x.num_readings for x in stats], [1, 1, 1, 1])

  def test_get_canonical_form_for_iterative_process(self):
    cf = test_utils.get_temperature_sensor_example()
    it = tff.backends.mapreduce.get_iterative_process_for_canonical_form(cf)
    cf = tff.backends.mapreduce.get_canonical_form_for_iterative_process(it)
    self.assertIsInstance(cf, tff.backends.mapreduce.CanonicalForm)

  def test_get_canonical_form_mnist_training(self):
    it = tff.backends.mapreduce.get_iterative_process_for_canonical_form(
        test_utils.get_mnist_training_example())
    cf = tff.backends.mapreduce.get_canonical_form_for_iterative_process(it)
    self.assertIsInstance(cf, tff.backends.mapreduce.CanonicalForm)

  def test_temperature_example_round_trip_(self):
    it = tff.backends.mapreduce.get_iterative_process_for_canonical_form(
        test_utils.get_temperature_sensor_example())
    cf = tff.backends.mapreduce.get_canonical_form_for_iterative_process(it)
    new_it = tff.backends.mapreduce.get_iterative_process_for_canonical_form(cf)
    state = new_it.initialize()
    self.assertEqual(str(state), '<num_rounds=0>')

    state, metrics, stats = new_it.next(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertEqual(str(state), '<num_rounds=1>')
    self.assertEqual(str(metrics), '<ratio_over_threshold=0.5>')
    self.assertCountEqual([x.num_readings for x in stats], [1, 3])

    state, metrics, stats = new_it.next(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertEqual(str(state), '<num_rounds=2>')
    self.assertEqual(str(metrics), '<ratio_over_threshold=0.75>')
    self.assertCountEqual([x.num_readings for x in stats], [1, 1, 1, 1])

  def test_mnist_training_round_trip(self):
    it = tff.backends.mapreduce.get_iterative_process_for_canonical_form(
        test_utils.get_mnist_training_example())
    cf = tff.backends.mapreduce.get_canonical_form_for_iterative_process(it)
    new_it = tff.backends.mapreduce.get_iterative_process_for_canonical_form(cf)
    state1 = it.initialize()
    state2 = new_it.initialize()
    self.assertEqual(str(state1), str(state2))
    dummy_x = np.array([[0.5] * 784], dtype=np.float32)
    dummy_y = np.array([1], dtype=np.int32)
    client_data = [collections.OrderedDict([('x', dummy_x), ('y', dummy_y)])]
    round_1 = it.next(state1, [client_data])
    state = round_1[0]
    metrics = round_1[1]
    alt_round_1 = new_it.next(state2, [client_data])
    alt_state = alt_round_1[0]
    alt_metrics = alt_round_1[1]
    self.assertEqual(str(round_1), str(alt_round_1))
    self.assertTrue(np.array_equal(state.model.weights, state.model.weights))
    self.assertTrue(np.array_equal(state.model.bias, alt_state.model.bias))
    self.assertTrue(np.array_equal(state.num_rounds, alt_state.num_rounds))
    self.assertTrue(np.array_equal(metrics.num_rounds, alt_metrics.num_rounds))
    self.assertTrue(
        np.array_equal(metrics.num_examples, alt_metrics.num_examples))
    self.assertTrue(np.array_equal(metrics.loss, alt_metrics.loss))

  def test_get_canonical_form_from_fl_api(self):
    it = test_utils.construct_example_training_comp()
    cf = tff.backends.mapreduce.get_canonical_form_for_iterative_process(it)
    new_it = tff.backends.mapreduce.get_iterative_process_for_canonical_form(cf)
    self.assertIsInstance(cf, tff.backends.mapreduce.CanonicalForm)
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
    self.assertEqual(str(state1), str(state2))

    sample_batch = collections.OrderedDict([('x',
                                             np.array([[1., 1.]],
                                                      dtype=np.float32)),
                                            ('y', np.array([[0]],
                                                           dtype=np.int32))])
    client_data = [sample_batch]

    round_1 = it.next(state1, [client_data])
    state = round_1[0]
    metrics = round_1[1]

    alt_round_1 = new_it.next(state2, [client_data])
    alt_state = alt_round_1[0]
    alt_metrics = alt_round_1[1]

    self.assertTrue(
        np.array_equal(state.model.trainable[0], alt_state.model.trainable[0]))
    self.assertTrue(
        np.array_equal(state.model.trainable[1], alt_state.model.trainable[1]))
    self.assertEqual(
        str(state.model.non_trainable), str(alt_state.model.non_trainable))
    self.assertEqual(state.optimizer_state[0], alt_state.optimizer_state[0])
    self.assertEmpty(state.delta_aggregate_state)
    self.assertEmpty(alt_state.delta_aggregate_state)
    self.assertEmpty(state.model_broadcast_state)
    self.assertEmpty(alt_state.model_broadcast_state)
    self.assertEqual(metrics.sparse_categorical_accuracy,
                     alt_metrics.sparse_categorical_accuracy)
    self.assertEqual(metrics.loss, alt_metrics.loss)


INIT_TYPE = tff.FederatedType(tf.float32, tff.SERVER)
S1_TYPE = INIT_TYPE
C1_TYPE = tff.FederatedType(tf.float32, tff.CLIENTS)
S6_TYPE = tff.FederatedType(tf.float64, tff.SERVER)
S7_TYPE = tff.FederatedType(tf.bool, tff.SERVER)
C6_TYPE = tff.FederatedType(tf.int64, tff.CLIENTS)
S2_TYPE = tff.FederatedType([tf.float32], tff.SERVER)
C2_TYPE = tff.FederatedType(S2_TYPE.member, tff.CLIENTS)
C5_TYPE = tff.FederatedType([tf.float64], tff.CLIENTS)
ZERO_TYPE = tff.TensorType(tf.int64)
ACCUMULATE_TYPE = tff.FunctionType([ZERO_TYPE, C5_TYPE.member], ZERO_TYPE)
MERGE_TYPE = tff.FunctionType([ZERO_TYPE, ZERO_TYPE], ZERO_TYPE)
REPORT_TYPE = tff.FunctionType(ZERO_TYPE, tf.int64)
S3_TYPE = tff.FederatedType(REPORT_TYPE.result, tff.SERVER)


def _create_next_type_with_s1_type(x):
  param_type = tff.NamedTupleType([x, C1_TYPE])
  result_type = tff.NamedTupleType([S6_TYPE, S7_TYPE, C6_TYPE])
  return tff.FunctionType(param_type, result_type)


def _create_before_broadcast_type_with_s1_type(x):
  return tff.FunctionType(tff.NamedTupleType([x, C1_TYPE]), S2_TYPE)


def _create_before_aggregate_with_c2_type(x):
  return tff.FunctionType(
      [[S1_TYPE, C1_TYPE], x],
      [C5_TYPE, ZERO_TYPE, ACCUMULATE_TYPE, MERGE_TYPE, REPORT_TYPE])


def _create_after_aggregate_with_s3_type(x):
  return tff.FunctionType([[[S1_TYPE, C1_TYPE], C2_TYPE], x],
                          [S6_TYPE, S7_TYPE, C6_TYPE])


class TypeCheckTest(absltest.TestCase):

  def test_init_raises_non_federated_type(self):
    with self.assertRaisesRegex(TypeError, 'init'):
      canonical_form_utils.pack_initialize_comp_type_signature(tf.float32)

  def test_init_passes_with_float_at_server(self):
    cf_types = canonical_form_utils.pack_initialize_comp_type_signature(
        tff.FederatedType(tf.float32, tff.SERVER))
    self.assertIsInstance(cf_types['initialize_type'], tff.FederatedType)
    self.assertEqual(cf_types['initialize_type'].placement, tff.SERVER)

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
        tff.FederatedType(tf.int32, tff.SERVER))
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
    self.assertEqual(packed_types['s2_type'],
                     tff.FederatedType(C2_TYPE.member, tff.SERVER))
    self.assertEqual(packed_types['prepare_type'],
                     tff.FunctionType(S1_TYPE.member, S2_TYPE.member))

  def test_before_broadcast_fails_mismatch_with_next_type(self):
    cf_types = {'initialize_type': INIT_TYPE}
    next_type = _create_next_type_with_s1_type(S1_TYPE)
    cf_types = canonical_form_utils.pack_next_comp_type_signature(
        next_type, cf_types)
    bad_before_broadcast_type = _create_before_broadcast_type_with_s1_type(
        tff.FederatedType(tf.int32, tff.SERVER))
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
        tff.FederatedType(tf.int32, tff.CLIENTS))
    with self.assertRaisesRegex(TypeError, 'before_aggregate'):
      canonical_form_utils.check_and_pack_before_aggregate_type_signature(
          bad_before_aggregate_type, cf_types)

  def test_after_aggregate_succeeds_and_packs(self):
    good_init_type = tff.FederatedType(tf.float32, tff.SERVER)
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
        tff.FederatedType([S1_TYPE.member, S3_TYPE.member], tff.SERVER))
    self.assertEqual(
        packed_types['c3_type'],
        tff.FederatedType([C1_TYPE.member, C2_TYPE.member], tff.CLIENTS))
    self.assertEqual(
        packed_types['update_type'],
        tff.FunctionType(packed_types['s4_type'].member,
                         packed_types['s5_type'].member))

  def test_after_aggregate_raises_mismatch_with_before_aggregate(self):
    good_init_type = tff.FederatedType(tf.float32, tff.SERVER)
    cf_types = canonical_form_utils.pack_initialize_comp_type_signature(
        good_init_type)
    next_type = _create_next_type_with_s1_type(
        tff.FederatedType(tf.float32, tff.SERVER))
    good_before_broadcast_type = _create_before_broadcast_type_with_s1_type(
        tff.FederatedType(tf.float32, tff.SERVER))
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
        tff.FederatedType(tf.int32, tff.SERVER))

    with self.assertRaisesRegex(TypeError, 'after_aggregate'):
      canonical_form_utils.check_and_pack_after_aggregate_type_signature(
          bad_after_aggregate_type, cf_types)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()
