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

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

_NUM_EXPLICIT_SUBROUNDS = 50


def build_sum_client_arg_computation(
    server_arg_type: tff.FederatedType,
    clients_arg_type: tff.FederatedType,
) -> tff.Computation:
  @tff.federated_computation(server_arg_type, clients_arg_type)
  def up_to_merge(server_arg, client_arg):
    del server_arg  # Unused
    return tff.federated_sum(client_arg)

  return up_to_merge


def build_noarg_count_clients_computation() -> tff.Computation:
  @tff.federated_computation()
  def up_to_merge():
    return tff.federated_sum(tff.federated_value(1, tff.CLIENTS))

  return up_to_merge


def build_whimsy_merge_computation(
    arg_type: tff.Type,
) -> tff.Computation:
  @tff.federated_computation(arg_type, arg_type)
  def merge(arg0, arg1):
    del arg1  # Unused
    return arg0

  return merge


def build_sum_merge_computation(
    arg_type: tff.Type,
) -> tff.Computation:
  @tff.tf_computation(arg_type, arg_type)
  def merge(arg0, arg1):
    return arg0 + arg1

  return merge


def build_whimsy_after_merge_computation(
    original_arg_type: tff.Type,
    merge_result_type: tff.Type,
) -> tff.Computation:
  if original_arg_type is not None:

    @tff.federated_computation(
        original_arg_type, tff.type_at_server(merge_result_type)
    )
    def after_merge(original_arg, merge_result):
      del merge_result  # Unused
      return original_arg

  else:

    @tff.federated_computation(tff.type_at_server(merge_result_type))
    def after_merge(merge_result):
      return merge_result

  return after_merge


def build_return_merge_result_computation(
    original_arg_type: tff.Type,
    merge_result_type: tff.Type,
) -> tff.Computation:
  @tff.federated_computation(
      original_arg_type, tff.type_at_server(merge_result_type)
  )
  def after_merge(original_arg, merge_result):
    del original_arg  # Unused
    return merge_result

  return after_merge


def build_return_merge_result_with_no_first_arg_computation(
    merge_result_type: tff.Type,
) -> tff.Computation:
  @tff.federated_computation(tff.type_at_server(merge_result_type))
  def after_merge(merge_result):
    return merge_result

  return after_merge


def build_sum_merge_with_first_arg_computation(
    original_arg_type: tff.Type,
    merge_result_type: tff.Type,
) -> tff.Computation:
  """Assumes original_arg_type is federated, and compatible with summing with merge_result_type."""

  @tff.tf_computation(original_arg_type[0].member, merge_result_type)
  def add(x, y):
    return x + y

  @tff.federated_computation(
      original_arg_type, tff.type_at_server(merge_result_type)
  )
  def after_merge(original_arg, merge_result):
    return tff.federated_map(add, (original_arg[0], merge_result))

  return after_merge


class MergeableCompExecutionContextTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          '100_clients_five_contexts_inferred_subrounds',
          (0, list(range(100))),
          None,
          5,
      ),
      (
          '100_clients_sixty_contexts_inferred_subrounds',
          (0, list(range(100))),
          None,
          60,
      ),
      (
          '100_clients_five_contexts_fifty_subrounds',
          (0, list(range(100))),
          _NUM_EXPLICIT_SUBROUNDS,
          5,
      ),
      (
          '100_clients_sixty_contexts_fifty_subrounds',
          (0, list(range(100))),
          _NUM_EXPLICIT_SUBROUNDS,
          60,
      ),
      (
          'fewer_clients_and_contexts_than_inferred_subrounds',
          (0, [0]),
          None,
          int(_NUM_EXPLICIT_SUBROUNDS / 2),
      ),
      (
          'fewer_clients_more_contexts_than_inferred_subrounds',
          (0, [0]),
          None,
          int(_NUM_EXPLICIT_SUBROUNDS * 2),
      ),
      (
          'fewer_clients_and_contexts_than_explicit_subrounds',
          (0, [0]),
          _NUM_EXPLICIT_SUBROUNDS,
          int(_NUM_EXPLICIT_SUBROUNDS / 2),
      ),
      (
          'fewer_clients_more_contexts_than_explicit_subrounds',
          (0, [0]),
          _NUM_EXPLICIT_SUBROUNDS,
          int(_NUM_EXPLICIT_SUBROUNDS * 2),
      ),
  )
  def test_runs_computation_with_clients_placed_return_values(
      self, arg, num_subrounds, num_contexts
  ):
    up_to_merge = build_sum_client_arg_computation(
        tff.type_at_server(tf.int32),
        tff.type_at_clients(tf.int32),
    )
    merge = build_whimsy_merge_computation(tf.int32)
    after_merge = build_whimsy_after_merge_computation(
        up_to_merge.type_signature.parameter, merge.type_signature.result
    )

    # Simply returns the original argument
    mergeable_comp_form = tff.framework.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge
    )
    contexts = []
    for _ in range(num_contexts):
      context = tff.framework.AsyncExecutionContext(
          executor_fn=tff.framework.local_cpp_executor_factory(
              max_concurrent_computation_calls=1
          ),
          compiler_fn=tff.backends.native.desugar_and_transform_to_native,
      )
      contexts.append(context)
    mergeable_comp_context = tff.framework.MergeableCompExecutionContext(
        contexts, num_subrounds=num_subrounds
    )
    # We preemptively package as a struct to work around shortcircuiting in
    # type_to_py_container in a non-Struct argument case.
    arg = tff.structure.Struct.unnamed(*arg)
    expected_result = tff.types.type_to_py_container(
        arg, after_merge.type_signature.result
    )
    result = mergeable_comp_context.invoke(mergeable_comp_form, arg)

    # We assert on the sets representing clients-placed values, rather than the
    # values themselves, since clients-placed values are represented as lists
    # and this test would therefore be dependent on the order in which these
    # values were returned.
    self.assertEqual(result['server_arg'], expected_result['server_arg'])
    self.assertSameElements(result['client_arg'], expected_result['client_arg'])
    self.assertCountEqual(result['client_arg'], expected_result['client_arg'])

  @parameterized.named_parameters(
      (
          '500_clients_inferred_subrounds',
          (0, list(range(500))),
          sum(range(500)),
          None,
      ),
      (
          '500_clients_nonzero_server_values_inferred_subrounds',
          (1, list(range(500))),
          sum(range(500)),
          None,
      ),
      ('fewer_clients_than__inferred_subrounds', (0, [1]), 1, None),
      (
          '500_clients_explicit_subrounds',
          (0, list(range(500))),
          sum(range(500)),
          _NUM_EXPLICIT_SUBROUNDS,
      ),
      (
          '500_clients_nonzero_server_values_explicit_subrounds',
          (1, list(range(500))),
          sum(range(500)),
          _NUM_EXPLICIT_SUBROUNDS,
      ),
      (
          'fewer_clients_than_explicit_subrounds',
          (0, [1]),
          1,
          _NUM_EXPLICIT_SUBROUNDS,
      ),
  )
  def test_computes_sum_of_clients_values(
      self, arg, expected_sum, num_subrounds
  ):
    up_to_merge = build_sum_client_arg_computation(
        tff.type_at_server(tf.int32),
        tff.type_at_clients(tf.int32),
    )
    merge = build_sum_merge_computation(tf.int32)
    after_merge = build_return_merge_result_computation(
        up_to_merge.type_signature.parameter, merge.type_signature.result
    )

    mergeable_comp_form = tff.framework.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge
    )
    contexts = []
    for _ in range(5):
      context = tff.framework.AsyncExecutionContext(
          executor_fn=tff.framework.local_cpp_executor_factory(
              max_concurrent_computation_calls=1
          ),
          compiler_fn=tff.backends.native.desugar_and_transform_to_native,
      )
      contexts.append(context)
    mergeable_comp_context = tff.framework.MergeableCompExecutionContext(
        contexts, num_subrounds=num_subrounds
    )

    expected_result = tff.types.type_to_py_container(
        expected_sum, after_merge.type_signature.result
    )
    result = mergeable_comp_context.invoke(mergeable_comp_form, arg)

    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      (
          '100_clients_inferred_subrounds',
          (100, list(range(100))),
          sum(range(101)),
          None,
      ),
      (
          '100_clients_explicit_subrounds',
          (100, list(range(100))),
          sum(range(101)),
          _NUM_EXPLICIT_SUBROUNDS,
      ),
      ('fewer_clients_than_inferred_subrounds', (1, [1]), 2, None),
      (
          'fewer_clients_than_explicit_subrounds',
          (1, [1]),
          2,
          _NUM_EXPLICIT_SUBROUNDS,
      ),
  )
  def test_computes_sum_of_all_values(self, arg, expected_sum, num_subrounds):
    up_to_merge = build_sum_client_arg_computation(
        tff.type_at_server(tf.int32),
        tff.type_at_clients(tf.int32),
    )
    merge = build_sum_merge_computation(tf.int32)
    after_merge = build_sum_merge_with_first_arg_computation(
        up_to_merge.type_signature.parameter, merge.type_signature.result
    )

    mergeable_comp_form = tff.framework.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge
    )
    contexts = []
    for _ in range(1):
      context = tff.framework.AsyncExecutionContext(
          executor_fn=tff.framework.local_cpp_executor_factory(
              max_concurrent_computation_calls=1
          ),
          compiler_fn=tff.backends.native.desugar_and_transform_to_native,
      )
      contexts.append(context)
    mergeable_comp_context = tff.framework.MergeableCompExecutionContext(
        contexts, num_subrounds=num_subrounds
    )

    expected_result = tff.types.type_to_py_container(
        expected_sum, after_merge.type_signature.result
    )
    result = mergeable_comp_context.invoke(mergeable_comp_form, arg)
    self.assertEqual(expected_result, result)

  @parameterized.named_parameters(
      ('inferred_subrounds', None),
      ('explicit_subrounds', _NUM_EXPLICIT_SUBROUNDS),
  )
  def test_counts_clients_with_noarg_computation(self, num_subrounds):
    num_clients = 100
    num_executors = 5
    up_to_merge = build_noarg_count_clients_computation()
    merge = build_sum_merge_computation(tf.int32)
    after_merge = build_return_merge_result_with_no_first_arg_computation(
        merge.type_signature.result
    )

    mergeable_comp_form = tff.framework.MergeableCompForm(
        up_to_merge=up_to_merge, merge=merge, after_merge=after_merge
    )
    contexts = []
    for _ in range(num_executors):
      context = tff.framework.AsyncExecutionContext(
          executor_fn=tff.framework.local_cpp_executor_factory(
              default_num_clients=int(num_clients / num_executors),
              max_concurrent_computation_calls=1,
          ),
          compiler_fn=tff.backends.native.desugar_and_transform_to_native,
      )
      contexts.append(context)
    mergeable_comp_context = tff.framework.MergeableCompExecutionContext(
        contexts, num_subrounds=num_subrounds
    )

    expected_result = num_clients
    result = mergeable_comp_context.invoke(mergeable_comp_form, None)
    self.assertEqual(result, expected_result)


if __name__ == '__main__':
  absltest.main()
