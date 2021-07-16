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

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.mapreduce import form_utils
from tensorflow_federated.python.core.backends.mapreduce import test_utils as mapreduce_test_utils
from tensorflow_federated.python.core.backends.mapreduce import transformations
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils as compiler_test_utils
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import transformations as compiler_transformations
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_transformations
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances

DEFAULT_GRAPPLER_CONFIG = tf.compat.v1.ConfigProto()


class CheckExtractionResultTest(test_case.TestCase):

  def get_function_from_first_symbol_binding_in_lambda_result(self, tree):
    """Unwraps a function from a series of nested calls, lambdas and blocks.

    The specific shape being unwrapped here is:

    (_ -> (let (_=_, ...) in _))
                  ^ This is the computation being returned.

    Args:
      tree: A series of nested calls and lambdas as described above.

    Returns:
      Inner function value described above.
    """
    self.assertIsInstance(tree, building_blocks.Lambda)
    self.assertIsNone(tree.parameter_type)
    self.assertIsInstance(tree.result, building_blocks.Block)
    comp_to_return = tree.result.locals[0][1]
    self.assertIsInstance(comp_to_return, building_blocks.Call)
    return comp_to_return.function

  def compiled_computation_for_initialize(self, initialize):
    block = initialize.to_building_block()
    return self.get_function_from_first_symbol_binding_in_lambda_result(block)

  def test_raises_on_none_args(self):
    with self.assertRaisesRegex(TypeError, 'None'):
      transformations.check_extraction_result(
          None, building_blocks.Reference('x', tf.int32))
    with self.assertRaisesRegex(TypeError, 'None'):
      transformations.check_extraction_result(
          building_blocks.Reference('x', tf.int32), None)

  def test_raises_function_and_call(self):
    function = building_blocks.Reference(
        'f', computation_types.FunctionType(tf.int32, tf.int32))
    integer_ref = building_blocks.Reference('x', tf.int32)
    call = building_blocks.Call(function, integer_ref)
    with self.assertRaisesRegex(transformations.MapReduceFormCompilationError,
                                'we have the functional type'):
      transformations.check_extraction_result(function, call)

  def test_raises_non_function_and_compiled_computation(self):
    init = form_utils.get_iterative_process_for_map_reduce_form(
        mapreduce_test_utils.get_temperature_sensor_example()).initialize
    compiled_computation = self.compiled_computation_for_initialize(init)
    integer_ref = building_blocks.Reference('x', tf.int32)
    with self.assertRaisesRegex(transformations.MapReduceFormCompilationError,
                                'we have the non-functional type'):
      transformations.check_extraction_result(integer_ref, compiled_computation)

  def test_raises_function_and_compiled_computation_of_different_type(self):
    init = form_utils.get_iterative_process_for_map_reduce_form(
        mapreduce_test_utils.get_temperature_sensor_example()).initialize
    compiled_computation = self.compiled_computation_for_initialize(init)
    function = building_blocks.Reference(
        'f', computation_types.FunctionType(tf.int32, tf.int32))
    with self.assertRaisesRegex(transformations.MapReduceFormCompilationError,
                                'incorrect TFF type'):
      transformations.check_extraction_result(function, compiled_computation)

  def test_raises_tensor_and_call_to_not_compiled_computation(self):
    function = building_blocks.Reference(
        'f', computation_types.FunctionType(tf.int32, tf.int32))
    ref_to_int = building_blocks.Reference('x', tf.int32)
    called_fn = building_blocks.Call(function, ref_to_int)
    with self.assertRaisesRegex(transformations.MapReduceFormCompilationError,
                                'missing'):
      transformations.check_extraction_result(ref_to_int, called_fn)

  def test_passes_function_and_compiled_computation_of_same_type(self):
    init = form_utils.get_iterative_process_for_map_reduce_form(
        mapreduce_test_utils.get_temperature_sensor_example()).initialize
    compiled_computation = self.compiled_computation_for_initialize(init)
    function = building_blocks.Reference('f',
                                         compiled_computation.type_signature)
    transformations.check_extraction_result(function, compiled_computation)


class ConsolidateAndExtractTest(test_case.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      transformations.consolidate_and_extract_local_processing(
          None, DEFAULT_GRAPPLER_CONFIG)

  def test_already_reduced_case(self):
    init = form_utils.get_iterative_process_for_map_reduce_form(
        mapreduce_test_utils.get_temperature_sensor_example()).initialize

    comp = init.to_building_block()

    result = transformations.consolidate_and_extract_local_processing(
        comp, DEFAULT_GRAPPLER_CONFIG)

    self.assertIsInstance(result, building_blocks.CompiledComputation)
    self.assertIsInstance(result.proto, computation_pb2.Computation)
    self.assertEqual(result.proto.WhichOneof('computation'), 'tensorflow')

  def test_reduces_unplaced_lambda_leaving_type_signature_alone(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        lam, DEFAULT_GRAPPLER_CONFIG)
    self.assertIsInstance(extracted_tf, building_blocks.CompiledComputation)
    self.assertEqual(extracted_tf.type_signature, lam.type_signature)

  def test_reduces_unplaced_lambda_to_equivalent_tf(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        lam, DEFAULT_GRAPPLER_CONFIG)
    executable_tf = computation_wrapper_instances.building_block_to_computation(
        extracted_tf)
    executable_lam = computation_wrapper_instances.building_block_to_computation(
        lam)
    for k in range(10):
      self.assertEqual(executable_tf(k), executable_lam(k))

  def test_reduces_federated_identity_to_member_identity(self):
    fed_int_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    lam = building_blocks.Lambda('x', fed_int_type,
                                 building_blocks.Reference('x', fed_int_type))
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        lam, DEFAULT_GRAPPLER_CONFIG)
    self.assertIsInstance(extracted_tf, building_blocks.CompiledComputation)
    unplaced_function_type = computation_types.FunctionType(
        fed_int_type.member, fed_int_type.member)
    self.assertEqual(extracted_tf.type_signature, unplaced_function_type)

  def test_reduces_federated_map_to_equivalent_function(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Reference('arg', arg_type)
    map_block = building_block_factory.create_federated_map_or_apply(lam, arg)
    mapping_fn = building_blocks.Lambda('arg', arg_type, map_block)
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        mapping_fn, DEFAULT_GRAPPLER_CONFIG)
    self.assertIsInstance(extracted_tf, building_blocks.CompiledComputation)
    executable_tf = computation_wrapper_instances.building_block_to_computation(
        extracted_tf)
    executable_lam = computation_wrapper_instances.building_block_to_computation(
        lam)
    for k in range(10):
      self.assertEqual(executable_tf(k), executable_lam(k))

  def test_reduces_federated_apply_to_equivalent_function(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Reference('arg', arg_type)
    map_block = building_block_factory.create_federated_map_or_apply(lam, arg)
    mapping_fn = building_blocks.Lambda('arg', arg_type, map_block)
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        mapping_fn, DEFAULT_GRAPPLER_CONFIG)
    self.assertIsInstance(extracted_tf, building_blocks.CompiledComputation)
    executable_tf = computation_wrapper_instances.building_block_to_computation(
        extracted_tf)
    executable_lam = computation_wrapper_instances.building_block_to_computation(
        lam)
    for k in range(10):
      self.assertEqual(executable_tf(k), executable_lam(k))

  def test_reduces_federated_value_at_server_to_equivalent_noarg_function(self):
    zero = building_block_factory.create_tensorflow_constant(
        computation_types.TensorType(tf.int32, shape=[]), 0)
    federated_value = building_block_factory.create_federated_value(
        zero, placements.SERVER)
    federated_value_func = building_blocks.Lambda(None, None, federated_value)
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        federated_value_func, DEFAULT_GRAPPLER_CONFIG)
    executable_tf = computation_wrapper_instances.building_block_to_computation(
        extracted_tf)
    self.assertEqual(executable_tf(), 0)

  def test_reduces_federated_value_at_clients_to_equivalent_noarg_function(
      self):
    zero = building_block_factory.create_tensorflow_constant(
        computation_types.TensorType(tf.int32, shape=[]), 0)
    federated_value = building_block_factory.create_federated_value(
        zero, placements.CLIENTS)
    federated_value_func = building_blocks.Lambda(None, None, federated_value)
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        federated_value_func, DEFAULT_GRAPPLER_CONFIG)
    executable_tf = computation_wrapper_instances.building_block_to_computation(
        extracted_tf)
    self.assertEqual(executable_tf(), 0)

  def test_reduces_lambda_returning_empty_tuple_to_tf(self):
    empty_tuple = building_blocks.Struct([])
    lam = building_blocks.Lambda('x', tf.int32, empty_tuple)
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        lam, DEFAULT_GRAPPLER_CONFIG)
    self.assertIsInstance(extracted_tf, building_blocks.CompiledComputation)


def _remove_client_all_equals_from_type(type_signature):

  def _transform(inner_type):
    if (inner_type.is_federated() and inner_type.placement.is_clients() and
        inner_type.all_equal):
      return computation_types.FederatedType(inner_type.member,
                                             inner_type.placement, False), True
    return inner_type, False

  return type_transformations.transform_type_postorder(type_signature,
                                                       _transform)[0]


class ForceAlignAndSplitByIntrinsicTest(test_case.TestCase):

  def assert_splits_on(self, comp, calls):
    """Asserts that `force_align_and_split_by_intrinsics` removes intrinsics."""
    if not isinstance(calls, list):
      calls = [calls]
    uris = [call.function.uri for call in calls]
    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, calls)

    # Ensure that the resulting computations no longer contain the split
    # intrinsics.
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uris))
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uris))
    # Removal isn't interesting to test for if it wasn't there to begin with.
    self.assertTrue(tree_analysis.contains_called_intrinsic(comp, uris))

    self.assert_types_equivalent(comp.parameter_type, before.parameter_type)
    # THere must be one parameter for each intrinsic in `calls`.
    before.type_signature.result.check_struct()
    self.assertLen(before.type_signature.result, len(calls))

    # Check that `after`'s parameter is a structure like:
    # {
    #   'original_arg': comp.parameter_type,
    #   'intrinsic_results': [...],
    # }
    after.parameter_type.check_struct()
    self.assertLen(after.parameter_type, 2)
    self.assert_types_equivalent(comp.parameter_type,
                                 after.parameter_type.original_arg)
    # There must be one result for each intrinsic in `calls`.
    self.assertLen(after.parameter_type.intrinsic_results, len(calls))

    # Check that each pair of (param, result) is a valid type substitution
    # for the intrinsic in question.
    for i in range(len(calls)):
      concrete_signature = computation_types.FunctionType(
          before.type_signature.result[i],
          after.parameter_type.intrinsic_results[i])
      abstract_signature = calls[i].function.intrinsic_def().type_signature
      # `force_align_and_split_by_intrinsics` loses all-equal data due to
      # zipping and unzipping. This is okay because the resulting computations
      # are not used together directly, but are compiled into unplaced TF code.
      abstract_signature = _remove_client_all_equals_from_type(
          abstract_signature)
      concrete_signature = _remove_client_all_equals_from_type(
          concrete_signature)
      type_analysis.check_concrete_instance_of(concrete_signature,
                                               abstract_signature)

  def test_cannot_split_on_chained_intrinsic(self):
    int_type = computation_types.TensorType(tf.int32)
    client_int_type = computation_types.at_clients(int_type)
    int_ref = lambda name: building_blocks.Reference(name, int_type)
    client_int_ref = (
        lambda name: building_blocks.Reference(name, client_int_type))
    body = building_blocks.Block([
        ('a',
         building_block_factory.create_federated_map(
             building_blocks.Lambda('p1', int_type, int_ref('p1')),
             client_int_ref('param'))),
        ('b',
         building_block_factory.create_federated_map(
             building_blocks.Lambda('p2', int_type, int_ref('p2')),
             client_int_ref('a'))),
    ], client_int_ref('b'))
    comp = building_blocks.Lambda('param', int_type, body)
    with self.assertRaises(transformations._NonAlignableAlongIntrinsicError):
      transformations.force_align_and_split_by_intrinsics(
          comp, [building_block_factory.create_null_federated_map()])

  def test_splits_on_selected_intrinsic_broadcast(self):
    federated_broadcast = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([federated_broadcast])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_nested_in_tuple_broadcast(self):
    first_broadcast = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    packed_broadcast = building_blocks.Struct([
        building_blocks.Data('a', computation_types.at_server(tf.int32)),
        first_broadcast
    ])
    sel = building_blocks.Selection(packed_broadcast, index=0)
    second_broadcast = building_block_factory.create_federated_broadcast(sel)
    result, _ = compiler_transformations.transform_to_call_dominant(
        second_broadcast)
    comp = building_blocks.Lambda('a', tf.int32, result)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_broadcast(self):
    federated_broadcast = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([
        federated_broadcast,
        federated_broadcast,
    ])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_aggregate(self):
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([federated_aggregate])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_aggregate()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_aggregate(self):
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_aggregate,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_aggregate()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_secure_sum_bitwidth(self):
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([federated_secure_sum_bitwidth])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_secure_sum_bitwidth()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_secure_sum_bitwidths(self):
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([
        federated_secure_sum_bitwidth,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_secure_sum_bitwidth()
    self.assert_splits_on(comp, call)

  def test_removes_selected_intrinsic_leaving_remaining_intrinsic(self):
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    null_aggregate = building_block_factory.create_null_federated_aggregate()
    secure_sum_bitwidth_uri = federated_secure_sum_bitwidth.function.uri
    aggregate_uri = null_aggregate.function.uri
    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, [null_aggregate])
    self.assertTrue(
        tree_analysis.contains_called_intrinsic(comp, secure_sum_bitwidth_uri))
    self.assertTrue(
        tree_analysis.contains_called_intrinsic(comp, aggregate_uri))
    self.assertFalse(
        tree_analysis.contains_called_intrinsic(before, aggregate_uri))
    self.assertFalse(
        tree_analysis.contains_called_intrinsic(after, aggregate_uri))
    self.assertTrue(
        tree_analysis.contains_called_intrinsic(before,
                                                secure_sum_bitwidth_uri) or
        tree_analysis.contains_called_intrinsic(after, secure_sum_bitwidth_uri))

  def test_splits_on_two_intrinsics(self):
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    self.assert_splits_on(comp, [
        building_block_factory.create_null_federated_aggregate(),
        building_block_factory.create_null_federated_secure_sum_bitwidth()
    ])

  def test_splits_on_multiple_instances_of_two_intrinsics(self):
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_aggregate,
        federated_secure_sum_bitwidth,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    self.assert_splits_on(comp, [
        building_block_factory.create_null_federated_aggregate(),
        building_block_factory.create_null_federated_secure_sum_bitwidth()
    ])

  def test_splits_even_when_selected_intrinsic_is_not_present(self):
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([federated_aggregate])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    transformations.force_align_and_split_by_intrinsics(comp, [
        building_block_factory.create_null_federated_aggregate(),
        building_block_factory.create_null_federated_secure_sum_bitwidth(),
    ])


class SelectFederatedOutputFromLambdaTest(test_case.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      transformations.select_output_from_lambda(None, 0)

  def test_raises_on_non_lambda(self):
    fed_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ref = building_blocks.Reference('x', [fed_type])
    with self.assertRaises(TypeError):
      transformations.select_output_from_lambda(ref, 0)

  def test_selects_single_federated_output(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    ref = building_blocks.Reference('x', [fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda('x', ref.type_signature, ref)
    zero_selected = transformations.select_output_from_lambda(lam, 0)
    self.assertEqual(zero_selected.type_signature.parameter,
                     lam.type_signature.parameter)
    self.assertEqual(zero_selected.type_signature.result,
                     lam.type_signature.result[0])
    self.assertEqual(str(zero_selected), '(x -> x[0])')

  def test_selects_single_federated_output_by_str_name(self):
    fed_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ref = building_blocks.Reference('x', [('a', fed_type)])
    lam = building_blocks.Lambda('x', ref.type_signature, ref)
    selected = transformations.select_output_from_lambda(lam, 'a')
    self.assert_types_equivalent(
        selected.type_signature,
        computation_types.FunctionType(lam.parameter_type,
                                       lam.type_signature.result['a']))

  def test_selects_tuple_of_federated_outputs(self):
    fed_at_clients = computation_types.at_clients(tf.int32)
    fed_at_server = computation_types.at_server(tf.int32)
    ref = building_blocks.Reference(
        'x', [fed_at_clients, fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda('x', ref.type_signature, ref)
    tuple_selected = transformations.select_output_from_lambda(lam, (0, 1))
    self.assertEqual(tuple_selected.type_signature.parameter,
                     lam.type_signature.parameter)
    self.assertEqual(
        tuple_selected.type_signature.result,
        computation_types.StructType(
            [lam.type_signature.result[0], lam.type_signature.result[1]]))
    self.assertEqual(str(tuple_selected), '(x -> <x[0],x[1]>)')

  def test_selects_tuple_of_federated_outputs_by_str_name(self):
    fed_at_clients = computation_types.at_clients(tf.int32)
    fed_at_server = computation_types.at_server(tf.int32)
    ref = building_blocks.Reference('x', [('a', fed_at_clients),
                                          ('b', fed_at_clients),
                                          ('c', fed_at_server)])
    lam = building_blocks.Lambda('x', ref.type_signature, ref)
    selected = transformations.select_output_from_lambda(lam, ('a', 'b'))
    self.assert_types_equivalent(
        selected.type_signature,
        computation_types.FunctionType(
            lam.parameter_type,
            computation_types.StructType(
                [lam.type_signature.result[0], lam.type_signature.result[1]])))

  def test_selects_list_of_federated_outputs(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    ref = building_blocks.Reference(
        'x', [fed_at_clients, fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda('x', ref.type_signature, ref)
    tuple_selected = transformations.select_output_from_lambda(lam, [0, 1])
    self.assertEqual(tuple_selected.type_signature.parameter,
                     lam.type_signature.parameter)
    self.assertEqual(
        tuple_selected.type_signature.result,
        computation_types.StructType(
            [lam.type_signature.result[0], lam.type_signature.result[1]]))
    self.assertEqual(str(tuple_selected), '(x -> <x[0],x[1]>)')

  def test_selects_single_unplaced_output(self):
    ref = building_blocks.Reference('x', [tf.int32, tf.float32, tf.int32])
    lam = building_blocks.Lambda('x', ref.type_signature, ref)
    int_selected = transformations.select_output_from_lambda(lam, 0)
    self.assertEqual(int_selected.type_signature.parameter,
                     lam.type_signature.parameter)
    self.assertEqual(int_selected.type_signature.result,
                     lam.type_signature.result[0])

  def test_selects_multiple_unplaced_outputs(self):
    ref = building_blocks.Reference('x', [tf.int32, tf.float32, tf.int32])
    lam = building_blocks.Lambda('x', ref.type_signature, ref)
    tuple_selected = transformations.select_output_from_lambda(lam, [0, 1])
    self.assertEqual(tuple_selected.type_signature.parameter,
                     lam.type_signature.parameter)
    self.assertEqual(
        tuple_selected.type_signature.result,
        computation_types.StructType(
            [lam.type_signature.result[0], lam.type_signature.result[1]]))
    self.assertEqual(str(tuple_selected), '(x -> <x[0],x[1]>)')


class ConcatenateFunctionOutputsTest(test_case.TestCase):

  def test_raises_on_non_lambda_args(self):
    reference = building_blocks.Reference('x', tf.int32)
    tff_lambda = building_blocks.Lambda('x', tf.int32, reference)
    with self.assertRaises(TypeError):
      transformations.concatenate_function_outputs(tff_lambda, reference)
    with self.assertRaises(TypeError):
      transformations.concatenate_function_outputs(reference, tff_lambda)

  def test_raises_on_non_unique_names(self):
    reference = building_blocks.Reference('x', tf.int32)
    good_lambda = building_blocks.Lambda('x', tf.int32, reference)
    bad_lambda = building_blocks.Lambda('x', tf.int32, good_lambda)
    with self.assertRaises(ValueError):
      transformations.concatenate_function_outputs(good_lambda, bad_lambda)
    with self.assertRaises(ValueError):
      transformations.concatenate_function_outputs(bad_lambda, good_lambda)

  def test_raises_on_different_parameter_types(self):
    int_reference = building_blocks.Reference('x', tf.int32)
    int_lambda = building_blocks.Lambda('x', tf.int32, int_reference)
    float_reference = building_blocks.Reference('x', tf.float32)
    float_lambda = building_blocks.Lambda('x', tf.float32, float_reference)
    with self.assertRaises(TypeError):
      transformations.concatenate_function_outputs(int_lambda, float_lambda)

  def test_parameters_are_mapped_together(self):
    x_reference = building_blocks.Reference('x', tf.int32)
    x_lambda = building_blocks.Lambda('x', tf.int32, x_reference)
    y_reference = building_blocks.Reference('y', tf.int32)
    y_lambda = building_blocks.Lambda('y', tf.int32, y_reference)
    concatenated = transformations.concatenate_function_outputs(
        x_lambda, y_lambda)
    parameter_name = concatenated.parameter_name

    def _raise_on_other_name_reference(comp):
      if isinstance(comp,
                    building_blocks.Reference) and comp.name != parameter_name:
        raise ValueError
      return comp, True

    tree_analysis.check_has_unique_names(concatenated)
    transformation_utils.transform_postorder(concatenated,
                                             _raise_on_other_name_reference)

  def test_concatenates_identities(self):
    x_reference = building_blocks.Reference('x', tf.int32)
    x_lambda = building_blocks.Lambda('x', tf.int32, x_reference)
    y_reference = building_blocks.Reference('y', tf.int32)
    y_lambda = building_blocks.Lambda('y', tf.int32, y_reference)
    concatenated = transformations.concatenate_function_outputs(
        x_lambda, y_lambda)
    self.assertEqual(str(concatenated), '(_var1 -> <_var1,_var1>)')


class NormalizedBitTest(test_case.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      transformations.normalize_all_equal_bit(None)

  def test_converts_all_equal_at_clients_reference_to_not_equal(self):
    fed_type_all_equal = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    normalized_comp = transformations.normalize_all_equal_bit(
        building_blocks.Reference('x', fed_type_all_equal))
    self.assertEqual(
        normalized_comp.type_signature,
        computation_types.FederatedType(
            tf.int32, placements.CLIENTS, all_equal=False))
    self.assertIsInstance(normalized_comp, building_blocks.Reference)
    self.assertEqual(str(normalized_comp), 'x')

  def test_converts_not_all_equal_at_server_reference_to_equal(self):
    fed_type_not_all_equal = computation_types.FederatedType(
        tf.int32, placements.SERVER, all_equal=False)
    normalized_comp = transformations.normalize_all_equal_bit(
        building_blocks.Reference('x', fed_type_not_all_equal))
    self.assertEqual(
        normalized_comp.type_signature,
        computation_types.FederatedType(
            tf.int32, placements.SERVER, all_equal=True))
    self.assertIsInstance(normalized_comp, building_blocks.Reference)
    self.assertEqual(str(normalized_comp), 'x')

  def test_converts_all_equal_at_clients_lambda_parameter_to_not_equal(self):
    fed_type_all_equal = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    normalized_fed_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS)
    ref = building_blocks.Reference('x', fed_type_all_equal)
    lam = building_blocks.Lambda('x', fed_type_all_equal, ref)
    normalized_lambda = transformations.normalize_all_equal_bit(lam)
    self.assertEqual(
        lam.type_signature,
        computation_types.FunctionType(fed_type_all_equal, fed_type_all_equal))
    self.assertIsInstance(normalized_lambda, building_blocks.Lambda)
    self.assertEqual(str(normalized_lambda), '(x -> x)')
    self.assertEqual(
        normalized_lambda.type_signature,
        computation_types.FunctionType(normalized_fed_type,
                                       normalized_fed_type))

  def test_converts_not_all_equal_at_server_lambda_parameter_to_equal(self):
    fed_type_not_all_equal = computation_types.FederatedType(
        tf.int32, placements.SERVER, all_equal=False)
    normalized_fed_type = computation_types.FederatedType(
        tf.int32, placements.SERVER)
    ref = building_blocks.Reference('x', fed_type_not_all_equal)
    lam = building_blocks.Lambda('x', fed_type_not_all_equal, ref)
    normalized_lambda = transformations.normalize_all_equal_bit(lam)
    self.assertEqual(
        lam.type_signature,
        computation_types.FunctionType(fed_type_not_all_equal,
                                       fed_type_not_all_equal))
    self.assertIsInstance(normalized_lambda, building_blocks.Lambda)
    self.assertEqual(str(normalized_lambda), '(x -> x)')
    self.assertEqual(
        normalized_lambda.type_signature,
        computation_types.FunctionType(normalized_fed_type,
                                       normalized_fed_type))

  def test_converts_federated_map_all_equal_to_federated_map(self):
    fed_type_all_equal = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    normalized_fed_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS)
    int_ref = building_blocks.Reference('x', tf.int32)
    int_identity = building_blocks.Lambda('x', tf.int32, int_ref)
    federated_int_ref = building_blocks.Reference('y', fed_type_all_equal)
    called_federated_map_all_equal = building_block_factory.create_federated_map_all_equal(
        int_identity, federated_int_ref)
    normalized_federated_map = transformations.normalize_all_equal_bit(
        called_federated_map_all_equal)
    self.assertEqual(called_federated_map_all_equal.function.uri,
                     intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri)
    self.assertIsInstance(normalized_federated_map, building_blocks.Call)
    self.assertIsInstance(normalized_federated_map.function,
                          building_blocks.Intrinsic)
    self.assertEqual(normalized_federated_map.function.uri,
                     intrinsic_defs.FEDERATED_MAP.uri)
    self.assertEqual(normalized_federated_map.type_signature,
                     normalized_fed_type)


if __name__ == '__main__':
  factory = executor_stacks.local_executor_factory()
  context = sync_execution_context.ExecutionContext(executor_fn=factory)
  set_default_context.set_default_context(context)
  test_case.main()
