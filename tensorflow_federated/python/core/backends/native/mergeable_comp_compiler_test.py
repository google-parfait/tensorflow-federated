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

import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import mergeable_comp_compiler
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def build_whimsy_computation_with_aggregation_and_after(server_arg_type,
                                                        clients_arg_type):

  @computations.tf_computation(server_arg_type.member, clients_arg_type.member)
  def compute_sum(x, y):
    return x + y

  @computations.federated_computation(server_arg_type, clients_arg_type)
  def aggregation_comp(server_arg, client_arg):
    summed_client_value = intrinsics.federated_sum(client_arg)
    return intrinsics.federated_map(compute_sum,
                                    (server_arg, summed_client_value))

  return aggregation_comp


def build_whimsy_computation_with_before_aggregation_work(
    server_arg_type, clients_arg_type):

  @computations.tf_computation(clients_arg_type.member)
  def compute_tuple_sum(x):
    return x[0] + x[1]

  @computations.tf_computation(server_arg_type.member,
                               clients_arg_type.member[0])
  def compute_sum(x, y):
    return x + y

  @computations.federated_computation(server_arg_type, clients_arg_type)
  def aggregation_comp(server_arg, client_arg):
    client_sums = intrinsics.federated_map(compute_tuple_sum, client_arg)
    summed_client_value = intrinsics.federated_sum(client_sums)
    return intrinsics.federated_map(compute_sum,
                                    (server_arg, summed_client_value))

  return aggregation_comp


@computations.tf_computation(tf.int32, tf.int32)
def tf_multiply_int(x, y):
  return x * y


@computations.federated_computation(
    computation_types.at_server([tf.int32, tf.int32]))
def server_placed_mult(arg):
  return intrinsics.federated_map(tf_multiply_int, arg)


class MergeableCompCompilerTest(test_case.TestCase):

  def setUp(self):
    ex_factory = executor_stacks.local_executor_factory(default_num_clients=1)
    self._mergeable_comp_context = mergeable_comp_execution_context.MergeableCompExecutionContext(
        [ex_factory])
    super().setUp()

  def _invoke_mergeable_form_on_arg(
      self, mergeable_form: mergeable_comp_execution_context.MergeableCompForm,
      arg):
    arg_type = mergeable_form.up_to_merge.type_signature.parameter
    if arg is not None:
      ingested_val = self._mergeable_comp_context.ingest(arg, arg_type)
    else:
      ingested_val = arg
    return self._mergeable_comp_context.invoke(mergeable_form, ingested_val)

  def test_raises_two_dependent_aggregates(self):

    @computations.federated_computation(computation_types.at_server(tf.int32))
    def dependent_agg_comp(server_arg):
      arg_at_clients = intrinsics.federated_broadcast(server_arg)
      sum_result = intrinsics.federated_sum(arg_at_clients)
      rebroadcast_sum = intrinsics.federated_broadcast(sum_result)
      return intrinsics.federated_sum(rebroadcast_sum)

    with self.assertRaisesRegex(ValueError,
                                'one aggregate dependent on another'):
      mergeable_comp_compiler.compile_to_mergeable_comp_form(dependent_agg_comp)

  def test_compiles_standalone_tensorflow_computation(self):
    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        tf_multiply_int)

    self.assertIsInstance(mergeable_form,
                          mergeable_comp_execution_context.MergeableCompForm)

  def test_compilation_preserves_semantics_standalone_tf(self):
    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        tf_multiply_int)

    expected_zero = self._invoke_mergeable_form_on_arg(mergeable_form, (1, 0))
    expected_two = self._invoke_mergeable_form_on_arg(mergeable_form, (1, 2))
    expected_six = self._invoke_mergeable_form_on_arg(mergeable_form, (2, 3))

    self.assertEqual(expected_zero, 0)
    self.assertEqual(expected_two, 2)
    self.assertEqual(expected_six, 6)

  def test_compiles_simple_noarg_computation(self):
    self.skipTest('b/203780753')

    @computations.federated_computation()
    def return_server_value():
      return intrinsics.federated_value(0, placements.SERVER)

    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        return_server_value)

    self.assertIsInstance(mergeable_form,
                          mergeable_comp_execution_context.MergeableCompForm)

  def test_preserves_semantics_of_noarg_computation(self):
    self.skipTest('b/203780753')

    @computations.federated_computation()
    def return_server_value():
      return intrinsics.federated_value(0, placements.SERVER)

    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        return_server_value)

    result = self._invoke_mergeable_form_on_arg(mergeable_form, None)
    self.assertEqual(result, 0)

  def test_compiles_server_placed_computation(self):
    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        server_placed_mult)

    self.assertIsInstance(mergeable_form,
                          mergeable_comp_execution_context.MergeableCompForm)

  def test_compilation_preserves_semantics_server_placed_computation(self):
    self.skipTest('b/200970992')
    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        server_placed_mult)

    expected_zero = self._invoke_mergeable_form_on_arg(mergeable_form, (1, 0))
    expected_two = self._invoke_mergeable_form_on_arg(mergeable_form, (1, 2))
    expected_six = self._invoke_mergeable_form_on_arg(mergeable_form, (2, 3))

    self.assertEqual(expected_zero, 0)
    self.assertEqual(expected_two, 2)
    self.assertEqual(expected_six, 6)

  def test_compiles_computation_with_aggregation_and_after(self):
    incoming_comp = build_whimsy_computation_with_aggregation_and_after(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))
    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        incoming_comp)

    self.assertIsInstance(mergeable_form,
                          mergeable_comp_execution_context.MergeableCompForm)

  def test_compilation_preserves_semantics_aggregation_and_after(self):
    incoming_comp = build_whimsy_computation_with_aggregation_and_after(
        computation_types.at_server(tf.int32),
        computation_types.at_clients(tf.int32))
    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        incoming_comp)
    arg = (100, list(range(100)))
    result = self._invoke_mergeable_form_on_arg(mergeable_form, arg)
    # Expected result is the sum of all the arguments, IE the sum of all
    # integers from 0 to 100, which is 101 * 100 / 2.
    self.assertEqual(result, 101 * 100 / 2)

  def test_compiles_computation_with_before_aggregation_work(self):
    incoming_comp = build_whimsy_computation_with_before_aggregation_work(
        computation_types.at_server(tf.int32),
        computation_types.at_clients([tf.int32, tf.int32]))
    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        incoming_comp)

    self.assertIsInstance(mergeable_form,
                          mergeable_comp_execution_context.MergeableCompForm)

  def test_compilation_preserves_semantics_before_agg_work(self):
    incoming_comp = build_whimsy_computation_with_before_aggregation_work(
        computation_types.at_server(tf.int32),
        computation_types.at_clients([tf.int32, tf.int32]))
    mergeable_form = mergeable_comp_compiler.compile_to_mergeable_comp_form(
        incoming_comp)
    arg = (100, [(x, x) for x in range(100)])
    result = self._invoke_mergeable_form_on_arg(mergeable_form, arg)
    # Expected result is again the sum of all arguments, which in this case is
    # 2 * 99 * 100 / 2 + 100
    self.assertEqual(result, 99 * 100 + 100)


if __name__ == '__main__':
  test_case.main()
