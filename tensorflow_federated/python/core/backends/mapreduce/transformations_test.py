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

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.mapreduce import canonical_form_utils
from tensorflow_federated.python.core.backends.mapreduce import test_utils as mapreduce_test_utils
from tensorflow_federated.python.core.backends.mapreduce import transformations
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils as compiler_test_utils
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances


DEFAULT_GRAPPLER_CONFIG = tf.compat.v1.ConfigProto()


class CheckExtractionResultTest(absltest.TestCase):

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
    block = mapreduce_test_utils.computation_to_building_block(initialize)
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
    with self.assertRaisesRegex(transformations.CanonicalFormCompilationError,
                                'we have the functional type'):
      transformations.check_extraction_result(function, call)

  def test_raises_non_function_and_compiled_computation(self):
    init = canonical_form_utils.get_iterative_process_for_canonical_form(
        mapreduce_test_utils.get_temperature_sensor_example()).initialize
    compiled_computation = self.compiled_computation_for_initialize(init)
    integer_ref = building_blocks.Reference('x', tf.int32)
    with self.assertRaisesRegex(transformations.CanonicalFormCompilationError,
                                'we have the non-functional type'):
      transformations.check_extraction_result(integer_ref, compiled_computation)

  def test_raises_function_and_compiled_computation_of_different_type(self):
    init = canonical_form_utils.get_iterative_process_for_canonical_form(
        mapreduce_test_utils.get_temperature_sensor_example()).initialize
    compiled_computation = self.compiled_computation_for_initialize(init)
    function = building_blocks.Reference(
        'f', computation_types.FunctionType(tf.int32, tf.int32))
    with self.assertRaisesRegex(transformations.CanonicalFormCompilationError,
                                'incorrect TFF type'):
      transformations.check_extraction_result(function, compiled_computation)

  def test_raises_tensor_and_call_to_not_compiled_computation(self):
    function = building_blocks.Reference(
        'f', computation_types.FunctionType(tf.int32, tf.int32))
    ref_to_int = building_blocks.Reference('x', tf.int32)
    called_fn = building_blocks.Call(function, ref_to_int)
    with self.assertRaisesRegex(transformations.CanonicalFormCompilationError,
                                'missing'):
      transformations.check_extraction_result(ref_to_int, called_fn)

  def test_passes_function_and_compiled_computation_of_same_type(self):
    init = canonical_form_utils.get_iterative_process_for_canonical_form(
        mapreduce_test_utils.get_temperature_sensor_example()).initialize
    compiled_computation = self.compiled_computation_for_initialize(init)
    function = building_blocks.Reference('f',
                                         compiled_computation.type_signature)
    transformations.check_extraction_result(function, compiled_computation)


class ConsolidateAndExtractTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      transformations.consolidate_and_extract_local_processing(
          None, DEFAULT_GRAPPLER_CONFIG)

  def test_raises_reference_to_functional_type(self):
    function_type = computation_types.FunctionType(tf.int32, tf.int32)
    ref = building_blocks.Reference('x', function_type)
    with self.assertRaisesRegex(ValueError, 'of functional type passed'):
      transformations.consolidate_and_extract_local_processing(
          ref, DEFAULT_GRAPPLER_CONFIG)

  def test_already_reduced_case(self):
    init = canonical_form_utils.get_iterative_process_for_canonical_form(
        mapreduce_test_utils.get_temperature_sensor_example()).initialize

    comp = mapreduce_test_utils.computation_to_building_block(init)

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
    arg = building_blocks.Reference(
        'arg', computation_types.FederatedType(tf.int32, placements.CLIENTS))
    mapped_fn = building_block_factory.create_federated_map_or_apply(lam, arg)
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        mapped_fn, DEFAULT_GRAPPLER_CONFIG)
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
    arg = building_blocks.Reference(
        'arg', computation_types.FederatedType(tf.int32, placements.CLIENTS))
    mapped_fn = building_block_factory.create_federated_map_or_apply(lam, arg)
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        mapped_fn, DEFAULT_GRAPPLER_CONFIG)
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
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        federated_value, DEFAULT_GRAPPLER_CONFIG)
    executable_tf = computation_wrapper_instances.building_block_to_computation(
        extracted_tf)
    self.assertEqual(executable_tf(), 0)

  def test_reduces_federated_value_at_clients_to_equivalent_noarg_function(
      self):
    zero = building_block_factory.create_tensorflow_constant(
        computation_types.TensorType(tf.int32, shape=[]), 0)
    federated_value = building_block_factory.create_federated_value(
        zero, placements.CLIENTS)
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        federated_value, DEFAULT_GRAPPLER_CONFIG)
    executable_tf = computation_wrapper_instances.building_block_to_computation(
        extracted_tf)
    self.assertEqual(executable_tf(), 0)

  def test_reduces_lambda_returning_empty_tuple_to_tf(self):
    empty_tuple = building_blocks.Struct([])
    lam = building_blocks.Lambda('x', tf.int32, empty_tuple)
    extracted_tf = transformations.consolidate_and_extract_local_processing(
        lam, DEFAULT_GRAPPLER_CONFIG)
    self.assertIsInstance(extracted_tf, building_blocks.CompiledComputation)


class ForceAlignAndSplitByIntrinsicTest(absltest.TestCase):

  def test_returns_trees_with_one_federated_broadcast(self):
    federated_broadcast = compiler_test_utils.create_dummy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([federated_broadcast])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    uri = [intrinsic_defs.FEDERATED_BROADCAST.uri]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_handles_federated_broadcasts_nested_in_tuple(self):
    first_broadcast = compiler_test_utils.create_dummy_called_federated_broadcast(
    )
    packed_broadcast = building_blocks.Struct([
        building_blocks.Data(
            'a',
            computation_types.FederatedType(
                computation_types.TensorType(tf.int32), placements.SERVER)),
        first_broadcast
    ])
    sel = building_blocks.Selection(packed_broadcast, index=0)
    second_broadcast = building_block_factory.create_federated_broadcast(sel)
    comp = building_blocks.Lambda('a', tf.int32, second_broadcast)
    uri = [intrinsic_defs.FEDERATED_BROADCAST.uri]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_two_federated_broadcast(self):
    federated_broadcast = compiler_test_utils.create_dummy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([
        federated_broadcast,
        federated_broadcast,
    ])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    uri = [intrinsic_defs.FEDERATED_BROADCAST.uri]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_one_federated_aggregate(self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([federated_aggregate])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [intrinsic_defs.FEDERATED_AGGREGATE.uri]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_two_federated_aggregates(self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_aggregate,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [intrinsic_defs.FEDERATED_AGGREGATE.uri]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_one_federated_secure_sum(self):
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([federated_secure_sum])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    uri = [intrinsic_defs.FEDERATED_SECURE_SUM.uri]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_two_federated_secure_sums(self):
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([
        federated_secure_sum,
        federated_secure_sum,
    ])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    uri = [intrinsic_defs.FEDERATED_SECURE_SUM.uri]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_one_federated_aggregate_and_one_federated_secure_sum_for_federated_aggregate_only(
      self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [intrinsic_defs.FEDERATED_AGGREGATE.uri]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_one_federated_aggregate_and_one_federated_secure_sum_for_federated_secure_sum_only(
      self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [intrinsic_defs.FEDERATED_SECURE_SUM.uri]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_one_federated_aggregate_and_one_federated_secure_sum_for_federated_aggregate_first(
      self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
        intrinsic_defs.FEDERATED_SECURE_SUM.uri,
    ]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_one_federated_aggregate_and_one_federated_secure_sum_for_federated_secure_sum_first(
      self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [
        intrinsic_defs.FEDERATED_SECURE_SUM.uri,
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
    ]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_two_federated_aggregates_and_one_federated_secure_sum(
      self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_aggregate,
        federated_secure_sum,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
        intrinsic_defs.FEDERATED_SECURE_SUM.uri,
    ]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_one_federated_aggregate_and_two_federated_secure_sums(
      self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum,
        federated_secure_sum,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
        intrinsic_defs.FEDERATED_SECURE_SUM.uri,
    ]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_two_federated_secure_sums_and_one_federated_aggregate(
      self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([
        federated_secure_sum,
        federated_secure_sum,
        federated_aggregate,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
        intrinsic_defs.FEDERATED_SECURE_SUM.uri,
    ]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_returns_trees_with_one_federated_secure_sum_and_two_federated_aggregates(
      self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum = compiler_test_utils.create_dummy_called_federated_secure_sum(
    )
    called_intrinsics = building_blocks.Struct([
        federated_secure_sum,
        federated_aggregate,
        federated_aggregate,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
        intrinsic_defs.FEDERATED_SECURE_SUM.uri,
    ]

    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, uri)

    self.assertIsInstance(before, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uri))
    self.assertIsInstance(after, building_blocks.Lambda)
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uri))

  def test_raises_value_error_for_expected_uri(self):
    federated_aggregate = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([federated_aggregate])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    uri = [
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
        intrinsic_defs.FEDERATED_SECURE_SUM.uri,
    ]

    with self.assertRaises(ValueError):
      transformations.force_align_and_split_by_intrinsics(comp, uri)


class BindSingleSelectionAsArgumentToLowerLevelLambdaTest(absltest.TestCase):

  def test_binds_single_argument_to_lower_lambda(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0))
    zeroth_index_extracted = transformations.bind_single_selection_as_argument_to_lower_level_lambda(
        lam, 0)
    self.assertEqual(zeroth_index_extracted.type_signature, lam.type_signature)
    self.assertIsInstance(zeroth_index_extracted, building_blocks.Lambda)
    self.assertIsInstance(zeroth_index_extracted.result, building_blocks.Call)
    self.assertIsInstance(zeroth_index_extracted.result.function,
                          building_blocks.Lambda)
    self.assertRegex(
        str(zeroth_index_extracted.result.function), r'\((.{4})1 -> (\1)1\)')
    self.assertEqual(str(zeroth_index_extracted.result.argument), '_var1[0]')

  def test_binds_single_argument_to_lower_lambda_with_selection_by_name(self):
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
    zeroth_index_extracted = transformations.bind_single_selection_as_argument_to_lower_level_lambda(
        lam, 1)
    self.assertEqual(zeroth_index_extracted.type_signature, lam.type_signature)
    self.assertIsInstance(zeroth_index_extracted, building_blocks.Lambda)
    self.assertIsInstance(zeroth_index_extracted.result, building_blocks.Call)
    self.assertIsInstance(zeroth_index_extracted.result.function,
                          building_blocks.Lambda)
    self.assertRegex(
        str(zeroth_index_extracted.result.function), r'\((.{4})1 -> (\1)1\)')
    self.assertEqual(str(zeroth_index_extracted.result.argument), '_var1[1]')


class ZipSelectionAsArgumentToLowerLevelLambdaTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      transformations.zip_selection_as_argument_to_lower_level_lambda(None, [0])

  def test_raises_on_non_lambda_comp(self):
    ref = building_blocks.Reference('x', [tf.int32])
    with self.assertRaises(TypeError):
      transformations.zip_selection_as_argument_to_lower_level_lambda(ref, [0])

  def test_raises_on_none_selections(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', [tf.int32]))
    with self.assertRaises(TypeError):
      transformations.zip_selection_as_argument_to_lower_level_lambda(lam, None)

  def test_raises_on_selection_tuple(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', [tf.int32]))
    with self.assertRaises(TypeError):
      transformations.zip_selection_as_argument_to_lower_level_lambda(lam, (0))

  def test_raises_on_non_tuple_parameter(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    with self.assertRaises(TypeError):
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          lam, [[0]])

  def test_raises_on_selection_from_non_tuple(self):
    lam = building_blocks.Lambda('x', [tf.int32],
                                 building_blocks.Reference('x', [tf.int32]))
    with self.assertRaisesRegex(TypeError, 'nonexistent index'):
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          lam, [[0, 0]])

  def test_raises_on_non_int_index(self):
    lam = building_blocks.Lambda(
        'x', [tf.int32], building_blocks.Reference('x', [('a', tf.int32)]))
    with self.assertRaises(TypeError):
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          lam, [['a']])

  def test_raises_on_non_federated_selection(self):
    lam = building_blocks.Lambda('x', [tf.int32],
                                 building_blocks.Reference('x', [tf.int32]))
    with self.assertRaises(TypeError):
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          lam, [[0]])

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
    with self.assertRaisesRegex(ValueError, 'at the same placement.'):
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          lam, [[0], [1]])

  def test_zips_single_element_tuple_to_lower_lambda(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0))
    expected_fn_regex = (r'\(_([a-z]{3})2 -> federated_map\(<\(_(\1)3 -> '
                         r'_(\1)3\[0\]\),_(\1)2>\)\)')
    expected_arg_regex = (r'federated_map\(<\(_([a-z]{3})4 -> '
                          r'<_(\1)4>\),<_(\1)1\[0\]>\[0\]>\)')

    zeroth_index_extracted = (
        transformations.zip_selection_as_argument_to_lower_level_lambda(
            lam, [[0]]))

    self.assertEqual(zeroth_index_extracted.type_signature, lam.type_signature)
    self.assertIsInstance(zeroth_index_extracted, building_blocks.Lambda)
    self.assertIsInstance(zeroth_index_extracted.result, building_blocks.Call)
    self.assertIsInstance(zeroth_index_extracted.result.function,
                          building_blocks.Lambda)
    self.assertRegexMatch(
        zeroth_index_extracted.result.function.compact_representation(),
        [expected_fn_regex])
    self.assertRegexMatch(
        zeroth_index_extracted.result.argument.compact_representation(),
        [expected_arg_regex])

  def test_zips_single_element_tuple_to_lower_lambda_selection_by_name(self):
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
    expected_fn_regex = (r'\(_([a-z]{3})2 -> federated_map\(<\(_(\1)3 -> '
                         r'_(\1)3\[0\]\),_(\1)2>\)\)')
    expected_arg_regex = (r'federated_map\(<\(_([a-z]{3})4 -> '
                          r'<_(\1)4>\),<_(\1)1\[0\]>\[0\]>\)')

    zeroth_index_extracted = (
        transformations.zip_selection_as_argument_to_lower_level_lambda(
            lam, [[0]]))

    self.assertEqual(zeroth_index_extracted.type_signature, lam.type_signature)
    self.assertIsInstance(zeroth_index_extracted, building_blocks.Lambda)
    self.assertIsInstance(zeroth_index_extracted.result, building_blocks.Call)
    self.assertIsInstance(zeroth_index_extracted.result.function,
                          building_blocks.Lambda)
    self.assertRegexMatch(
        zeroth_index_extracted.result.function.compact_representation(),
        [expected_fn_regex])
    self.assertRegexMatch(
        zeroth_index_extracted.result.argument.compact_representation(),
        [expected_arg_regex])

  def test_binding_single_arg_leaves_no_unbound_references(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
    tuple_of_federated_types = computation_types.StructType(
        [fed_at_clients, fed_at_server])
    lam = building_blocks.Lambda(
        'x', tuple_of_federated_types,
        building_blocks.Selection(
            building_blocks.Reference('x', tuple_of_federated_types), index=0))
    zeroth_index_extracted = transformations.zip_selection_as_argument_to_lower_level_lambda(
        lam, [[0]])
    unbound_references = transformation_utils.get_map_of_unbound_references(
        zeroth_index_extracted)[zeroth_index_extracted]
    self.assertEmpty(unbound_references)

  def test_binds_single_arg_deep_in_type_tree_to_lower_lambda(self):
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

    expected_fn_regex = (r'\(_([a-z]{3})2 -> federated_map\(<\(_(\1)3 -> '
                         r'_(\1)3\[0\]\),_(\1)2>\)\)')
    expected_arg_regex = (r'federated_map\(<\(_([a-z]{3})4 -> '
                          r'<_(\1)4>\),<_(\1)1\[0\]\[0\]>\[0\]>\)')

    deep_zeroth_index_extracted = transformations.zip_selection_as_argument_to_lower_level_lambda(
        lam, [[0, 0]])

    self.assertEqual(deep_zeroth_index_extracted.type_signature,
                     lam.type_signature)
    self.assertIsInstance(deep_zeroth_index_extracted, building_blocks.Lambda)
    self.assertIsInstance(deep_zeroth_index_extracted.result,
                          building_blocks.Call)
    self.assertIsInstance(deep_zeroth_index_extracted.result.function,
                          building_blocks.Lambda)
    self.assertRegexMatch(
        deep_zeroth_index_extracted.result.function.compact_representation(),
        [expected_fn_regex])

    self.assertRegexMatch(
        deep_zeroth_index_extracted.result.argument.compact_representation(),
        [expected_arg_regex])

  def test_binds_multiple_args_deep_in_type_tree_to_lower_lambda(self):
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
    expected_fn_regex = (r'\(_([a-z]{3})2 -> <federated_map\(<\(_(\1)3 -> '
                         r'_(\1)3\[0\]\),_(\1)2>\),federated_map\(<\(_(\1)4 -> '
                         r'_(\1)4\[1\]\),_(\1)2>\)>\)')
    expected_arg_regex = r'federated_zip_at_clients\(<_([a-z]{3})1\[0\]\[0\],_(\1)1\[2\]\[0\]>\)'

    deep_zeroth_index_extracted = transformations.zip_selection_as_argument_to_lower_level_lambda(
        lam, [[0, 0], [2, 0]])

    self.assertEqual(deep_zeroth_index_extracted.type_signature,
                     lam.type_signature)
    self.assertIsInstance(deep_zeroth_index_extracted, building_blocks.Lambda)
    self.assertIsInstance(deep_zeroth_index_extracted.result,
                          building_blocks.Call)
    self.assertIsInstance(deep_zeroth_index_extracted.result.function,
                          building_blocks.Lambda)
    self.assertRegexMatch(
        deep_zeroth_index_extracted.result.function.compact_representation(),
        [expected_fn_regex])
    self.assertRegexMatch(
        deep_zeroth_index_extracted.result.argument.compact_representation(),
        [expected_arg_regex])

  def test_binds_multiple_args_deep_in_type_tree_to_lower_lambda_selected_by_name(
      self):
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
    expected_fn_regex = (r'\(_([a-z]{3})2 -> <federated_map\(<\(_(\1)3 -> '
                         r'_(\1)3\[0\]\),_(\1)2>\),federated_map\(<\(_(\1)4 -> '
                         r'_(\1)4\[1\]\),_(\1)2>\)>\)')
    expected_arg_regex = r'federated_zip_at_clients\(<_([a-z]{3})1\[0\]\[0\],_(\1)1\[2\]\[0\]>\)'

    deep_zeroth_index_extracted = transformations.zip_selection_as_argument_to_lower_level_lambda(
        lam, [[0, 0], [2, 0]])

    self.assertEqual(deep_zeroth_index_extracted.type_signature,
                     lam.type_signature)
    self.assertIsInstance(deep_zeroth_index_extracted, building_blocks.Lambda)
    self.assertIsInstance(deep_zeroth_index_extracted.result,
                          building_blocks.Call)
    self.assertIsInstance(deep_zeroth_index_extracted.result.function,
                          building_blocks.Lambda)
    self.assertRegexMatch(
        deep_zeroth_index_extracted.result.function.compact_representation(),
        [expected_fn_regex])
    self.assertRegexMatch(
        deep_zeroth_index_extracted.result.argument.compact_representation(),
        [expected_arg_regex])

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
    deep_zeroth_index_extracted = transformations.zip_selection_as_argument_to_lower_level_lambda(
        lam, [[0, 0], [2, 0]])
    tree_analysis.check_has_unique_names(deep_zeroth_index_extracted)


class SelectFederatedOutputFromLambdaTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      transformations.select_output_from_lambda(None, 0)

  def test_raises_on_non_lambda(self):
    fed_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ref = building_blocks.Reference('x', [fed_type])
    with self.assertRaises(TypeError):
      transformations.select_output_from_lambda(ref, 0)

  def test_raises_on_string_indices(self):
    fed_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ref = building_blocks.Reference('x', [('a', fed_type)])
    lam = building_blocks.Lambda('x', ref.type_signature, ref)
    with self.assertRaises(TypeError):
      transformations.select_output_from_lambda(lam, 'a')

  def test_raises_on_list_of_strings(self):
    fed_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ref = building_blocks.Reference('x', [[('a', fed_type)]])
    lam = building_blocks.Lambda('x', ref.type_signature, ref)
    with self.assertRaises(TypeError):
      transformations.select_output_from_lambda(lam, ['a'])

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

  def test_selects_tuple_of_federated_outputs(self):
    fed_at_clients = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    fed_at_server = computation_types.FederatedType(tf.int32, placements.SERVER)
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


class ConcatenateFunctionOutputsTest(absltest.TestCase):

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


class NormalizedBitTest(absltest.TestCase):

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
  context = execution_context.ExecutionContext(executor_fn=factory)
  set_default_context.set_default_context(context)
  absltest.main()
