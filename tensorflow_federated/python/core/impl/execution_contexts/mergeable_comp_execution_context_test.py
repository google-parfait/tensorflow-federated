# Copyright 2021, The TensorFlow Federated Authors.
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
"""Tests for single_agg_execution_context."""

from absl.testing import absltest

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types


def build_whimsy_up_to_merge_computation(
    server_arg_type: computation_types.FederatedType,
    clients_arg_type: computation_types.FederatedType
) -> computation_base.Computation:

  @computations.federated_computation(server_arg_type, clients_arg_type)
  def up_to_merge(server_arg, client_arg):
    del server_arg  # Unused
    return intrinsics.federated_sum(client_arg)

  return up_to_merge


def build_whimsy_merge_computation(
    arg_type: computation_types.Type) -> computation_base.Computation:

  @computations.federated_computation(arg_type, arg_type)
  def merge(arg0, arg1):
    del arg1  # Unused
    return arg0

  return merge


def build_whimsy_after_merge_computation(
    original_arg_type: computation_types.Type,
    merge_result_type: computation_types.Type) -> computation_base.Computation:

  @computations.federated_computation(
      original_arg_type, computation_types.at_server(merge_result_type))
  def after_merge(original_arg, merge_result):
    del merge_result  # Unused
    return original_arg

  return after_merge


class MergeableCompFormTest(absltest.TestCase):

  def test_raises_mismatched_up_to_merge_and_merge(self):
    up_to_merge = build_whimsy_up_to_merge_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))

    bad_merge = build_whimsy_merge_computation(tf.float32)

    @computations.federated_computation(up_to_merge.type_signature.parameter,
                                        computation_types.at_server(
                                            bad_merge.type_signature.result))
    def after_merge(x, y):
      return (x, y)

    with self.assertRaises(
        mergeable_comp_execution_context.MergeTypeNotAssignableError):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=up_to_merge, merge=bad_merge, after_merge=after_merge)

  def test_raises_merge_computation_not_assignable_result(self):
    up_to_merge = build_whimsy_up_to_merge_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))

    @computations.tf_computation(tf.int32, tf.int32)
    def bad_merge(x, y):
      del x, y  # Unused
      return 1.  # of type float.

    @computations.federated_computation(up_to_merge.type_signature.parameter,
                                        computation_types.at_server(
                                            bad_merge.type_signature.result))
    def after_merge(x, y):
      return (x, y)

    with self.assertRaises(
        mergeable_comp_execution_context.MergeTypeNotAssignableError):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=up_to_merge, merge=bad_merge, after_merge=after_merge)

  def test_raises_no_top_level_argument_in_after_agg(self):
    up_to_merge = build_whimsy_up_to_merge_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))

    merge = build_whimsy_merge_computation(tf.int32)

    @computations.federated_computation(
        computation_types.at_server(merge.type_signature.result))
    def bad_after_merge(x):
      return x

    with self.assertRaises(computation_types.TypeNotAssignableError):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=up_to_merge, merge=merge, after_merge=bad_after_merge)

  def test_raises_up_to_merge_returns_non_server_placed_result(self):

    @computations.federated_computation(computation_types.at_server(tf.int32))
    def bad_up_to_merge(x):
      # Returns non SERVER-placed result.
      return x, x

    merge = build_whimsy_merge_computation(tf.int32)

    after_merge = build_whimsy_after_merge_computation(
        bad_up_to_merge.type_signature.parameter, merge.type_signature.result)

    with self.assertRaises(mergeable_comp_execution_context.UpToMergeTypeError):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=bad_up_to_merge, merge=merge, after_merge=after_merge)

  def test_raises_with_aggregation_in_after_agg(self):
    up_to_merge = build_whimsy_up_to_merge_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))

    merge = build_whimsy_merge_computation(tf.int32)

    @computations.federated_computation(up_to_merge.type_signature.parameter,
                                        computation_types.at_server(
                                            merge.type_signature.result))
    def after_merge_with_sum(original_arg, merged_arg):
      del merged_arg  # Unused
      # Second element in original arg is the clients-placed value.
      return intrinsics.federated_sum(original_arg[1])

    with self.assertRaisesRegex(
        mergeable_comp_execution_context.AfterMergeStructureError,
        'federated_sum'):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=up_to_merge,
          merge=merge,
          after_merge=after_merge_with_sum)

  def test_raises_with_federated_collect_in_after_agg(self):
    up_to_merge = build_whimsy_up_to_merge_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))

    merge = build_whimsy_merge_computation(tf.int32)

    @computations.federated_computation(up_to_merge.type_signature.parameter,
                                        computation_types.at_server(
                                            merge.type_signature.result))
    def after_merge_with_collect(original_arg, merged_arg):
      del merged_arg  # Unused
      # Second element in original arg is the clients-placed value.
      return intrinsics.federated_collect(original_arg[1])

    with self.assertRaisesRegex(
        mergeable_comp_execution_context.AfterMergeStructureError,
        'federated_collect'):
      mergeable_comp_execution_context.MergeableCompForm(
          up_to_merge=up_to_merge,
          merge=merge,
          after_merge=after_merge_with_collect)

  def test_passes_with_correct_signatures(self):
    up_to_merge = build_whimsy_up_to_merge_computation(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))
    merge = build_whimsy_merge_computation(tf.int32)
    after_merge = build_whimsy_after_merge_computation(
        up_to_merge.type_signature.parameter, merge.type_signature.result)
    single_agg_form = mergeable_comp_execution_context.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge)

    self.assertIsInstance(single_agg_form,
                          mergeable_comp_execution_context.MergeableCompForm)


if __name__ == '__main__':
  absltest.main()
