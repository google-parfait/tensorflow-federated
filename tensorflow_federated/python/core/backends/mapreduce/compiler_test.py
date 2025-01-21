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
import federated_language
from federated_language.proto import computation_pb2
import numpy as np
import tensorflow as tf
from tensorflow_federated.python.core.backends.mapreduce import compiler
from tensorflow_federated.python.core.backends.mapreduce import form_utils
from tensorflow_federated.python.core.backends.mapreduce import mapreduce_test_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_computation_factory
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.compiler import building_block_test_utils
from tensorflow_federated.python.core.impl.executor_stacks import executor_factory  # pylint: enable=line-too-long

DEFAULT_GRAPPLER_CONFIG = tf.compat.v1.ConfigProto()


def _create_test_context():
  factory = executor_factory.local_cpp_executor_factory()
  return federated_language.framework.SyncExecutionContext(
      executor_fn=factory,
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
  )


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
    self.assertIsInstance(tree, federated_language.framework.Lambda)
    self.assertIsNone(tree.parameter_type)
    self.assertIsInstance(tree.result, federated_language.framework.Block)
    comp_to_return = tree.result.locals[0][1]
    self.assertIsInstance(comp_to_return, federated_language.framework.Call)
    return comp_to_return.function

  def compiled_computation_for_initialize(self, initialize):
    # Create a federated version of initialize.
    @federated_language.federated_computation
    def federated_initialize_computation():
      return federated_language.federated_value(
          initialize(), federated_language.SERVER
      )

    block = federated_initialize_computation.to_building_block()
    return self.get_function_from_first_symbol_binding_in_lambda_result(block)

  def test_raises_on_none_args(self):
    with self.assertRaisesRegex(TypeError, 'None'):
      compiler.check_extraction_result(
          None, federated_language.framework.Reference('x', np.int32)
      )
    with self.assertRaisesRegex(TypeError, 'None'):
      compiler.check_extraction_result(
          federated_language.framework.Reference('x', np.int32), None
      )

  def test_raises_function_and_call(self):
    function = federated_language.framework.Reference(
        'f', federated_language.FunctionType(np.int32, np.int32)
    )
    integer_ref = federated_language.framework.Reference('x', np.int32)
    call = federated_language.framework.Call(function, integer_ref)
    with self.assertRaisesRegex(
        compiler.MapReduceFormCompilationError, 'we have the functional type'
    ):
      compiler.check_extraction_result(function, call)

  def test_raises_non_function_and_compiled_computation(self):
    initialize = (
        mapreduce_test_utils.get_temperature_sensor_example().initialize
    )
    init = form_utils.get_state_initialization_computation(initialize)
    compiled_computation = self.compiled_computation_for_initialize(init)
    integer_ref = federated_language.framework.Reference('x', np.int32)
    with self.assertRaisesRegex(
        compiler.MapReduceFormCompilationError,
        'we have the non-functional type',
    ):
      compiler.check_extraction_result(integer_ref, compiled_computation)

  def test_raises_function_and_compiled_computation_of_different_type(self):
    initialize = (
        mapreduce_test_utils.get_temperature_sensor_example().initialize
    )
    init = form_utils.get_state_initialization_computation(initialize)
    compiled_computation = self.compiled_computation_for_initialize(init)
    function = federated_language.framework.Reference(
        'f', federated_language.FunctionType(np.int32, np.int32)
    )
    with self.assertRaisesRegex(
        compiler.MapReduceFormCompilationError, 'incorrect TFF type'
    ):
      compiler.check_extraction_result(function, compiled_computation)

  def test_raises_tensor_and_call_to_not_compiled_computation(self):
    function = federated_language.framework.Reference(
        'f', federated_language.FunctionType(np.int32, np.int32)
    )
    ref_to_int = federated_language.framework.Reference('x', np.int32)
    called_fn = federated_language.framework.Call(function, ref_to_int)
    with self.assertRaisesRegex(
        compiler.MapReduceFormCompilationError, 'missing'
    ):
      compiler.check_extraction_result(ref_to_int, called_fn)

  def test_passes_function_and_compiled_computation_of_same_type(self):
    initialize = (
        mapreduce_test_utils.get_temperature_sensor_example().initialize
    )
    init = form_utils.get_state_initialization_computation(initialize)
    compiled_computation = self.compiled_computation_for_initialize(init)
    function = federated_language.framework.Reference(
        'f', compiled_computation.type_signature
    )
    compiler.check_extraction_result(function, compiled_computation)


class ConsolidateAndExtractTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      compiler.consolidate_and_extract_local_processing(
          None, DEFAULT_GRAPPLER_CONFIG
      )

  def test_already_reduced_case(self):
    initialize = (
        mapreduce_test_utils.get_temperature_sensor_example().initialize
    )
    init = form_utils.get_state_initialization_computation(initialize)

    comp = init.to_building_block()

    result = compiler.consolidate_and_extract_local_processing(
        comp, DEFAULT_GRAPPLER_CONFIG
    )

    self.assertIsInstance(
        result, federated_language.framework.CompiledComputation
    )
    self.assertIsInstance(result.proto, computation_pb2.Computation)
    self.assertEqual(result.proto.WhichOneof('computation'), 'tensorflow')

  def test_reduces_unplaced_lambda_leaving_type_signature_alone(self):
    lam = federated_language.framework.Lambda(
        'x', np.int32, federated_language.framework.Reference('x', np.int32)
    )
    extracted_tf = compiler.consolidate_and_extract_local_processing(
        lam, DEFAULT_GRAPPLER_CONFIG
    )
    self.assertIsInstance(
        extracted_tf, federated_language.framework.CompiledComputation
    )
    self.assertEqual(extracted_tf.type_signature, lam.type_signature)

  def test_further_concretizes_type_if_possible(self):
    unk_size_int_type = federated_language.TensorType(
        dtype=np.int32, shape=[None]
    )
    known_size_int_type = federated_language.TensorType(
        dtype=np.int32, shape=[1]
    )
    lam_with_unk_size = federated_language.framework.Lambda(
        'x',
        unk_size_int_type,
        federated_language.framework.Reference('x', unk_size_int_type),
    )
    known_size_ref = federated_language.framework.Reference(
        'y', known_size_int_type
    )
    called_identity_knowable_size = federated_language.framework.Call(
        lam_with_unk_size, known_size_ref
    )
    lam_with_knowable_size = federated_language.framework.Lambda(
        known_size_ref.name,
        known_size_ref.type_signature,
        called_identity_knowable_size,
    )

    extracted_tf = compiler.consolidate_and_extract_local_processing(
        lam_with_knowable_size, DEFAULT_GRAPPLER_CONFIG
    )

    self.assertIsInstance(
        extracted_tf, federated_language.framework.CompiledComputation
    )
    # Assert assignability only goes one way in this case--the compiler can
    # concretize the type of the lambda further.
    self.assertTrue(
        lam_with_knowable_size.type_signature.is_assignable_from(
            extracted_tf.type_signature
        )
    )
    self.assertFalse(
        extracted_tf.type_signature.is_assignable_from(
            lam_with_knowable_size.type_signature
        )
    )

  def test_reduces_unplaced_lambda_to_equivalent_tf(self):
    lam = federated_language.framework.Lambda(
        'x', np.int32, federated_language.framework.Reference('x', np.int32)
    )
    extracted_tf = compiler.consolidate_and_extract_local_processing(
        lam, DEFAULT_GRAPPLER_CONFIG
    )
    executable_tf = (
        federated_language.framework.ConcreteComputation.from_building_block(
            extracted_tf
        )
    )
    executable_lam = (
        federated_language.framework.ConcreteComputation.from_building_block(
            lam
        )
    )
    for k in range(10):
      self.assertEqual(executable_tf(k), executable_lam(k))

  def test_reduces_federated_identity_to_member_identity(self):
    fed_int_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    lam = federated_language.framework.Lambda(
        'x',
        fed_int_type,
        federated_language.framework.Reference('x', fed_int_type),
    )
    extracted_tf = compiler.consolidate_and_extract_local_processing(
        lam, DEFAULT_GRAPPLER_CONFIG
    )
    self.assertIsInstance(
        extracted_tf, federated_language.framework.CompiledComputation
    )
    unplaced_function_type = federated_language.FunctionType(
        fed_int_type.member, fed_int_type.member
    )
    self.assertEqual(extracted_tf.type_signature, unplaced_function_type)

  def test_reduces_federated_map_to_equivalent_function(self):
    lam = federated_language.framework.Lambda(
        'x', np.int32, federated_language.framework.Reference('x', np.int32)
    )
    arg_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    arg = federated_language.framework.Reference('arg', arg_type)
    map_block = federated_language.framework.create_federated_map_or_apply(
        lam, arg
    )
    mapping_fn = federated_language.framework.Lambda('arg', arg_type, map_block)
    extracted_tf = compiler.consolidate_and_extract_local_processing(
        mapping_fn, DEFAULT_GRAPPLER_CONFIG
    )
    self.assertIsInstance(
        extracted_tf, federated_language.framework.CompiledComputation
    )
    executable_tf = (
        federated_language.framework.ConcreteComputation.from_building_block(
            extracted_tf
        )
    )
    executable_lam = (
        federated_language.framework.ConcreteComputation.from_building_block(
            lam
        )
    )
    for k in range(10):
      self.assertEqual(executable_tf(k), executable_lam(k))

  def test_reduces_federated_apply_to_equivalent_function(self):
    lam = federated_language.framework.Lambda(
        'x', np.int32, federated_language.framework.Reference('x', np.int32)
    )
    arg_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    arg = federated_language.framework.Reference('arg', arg_type)
    map_block = federated_language.framework.create_federated_map_or_apply(
        lam, arg
    )
    mapping_fn = federated_language.framework.Lambda('arg', arg_type, map_block)
    extracted_tf = compiler.consolidate_and_extract_local_processing(
        mapping_fn, DEFAULT_GRAPPLER_CONFIG
    )
    self.assertIsInstance(
        extracted_tf, federated_language.framework.CompiledComputation
    )
    executable_tf = (
        federated_language.framework.ConcreteComputation.from_building_block(
            extracted_tf
        )
    )
    executable_lam = (
        federated_language.framework.ConcreteComputation.from_building_block(
            lam
        )
    )
    for k in range(10):
      self.assertEqual(executable_tf(k), executable_lam(k))

  def test_reduces_federated_value_at_server_to_equivalent_noarg_function(self):
    zero_proto, zero_type = tensorflow_computation_factory.create_constant(
        0, federated_language.TensorType(np.int32)
    )
    zero_compiled = federated_language.framework.CompiledComputation(
        zero_proto, type_signature=zero_type
    )
    zero = federated_language.framework.Call(zero_compiled, None)
    federated_value = federated_language.framework.create_federated_value(
        zero, federated_language.SERVER
    )
    federated_value_func = federated_language.framework.Lambda(
        None, None, federated_value
    )
    extracted_tf = compiler.consolidate_and_extract_local_processing(
        federated_value_func, DEFAULT_GRAPPLER_CONFIG
    )
    executable_tf = (
        federated_language.framework.ConcreteComputation.from_building_block(
            extracted_tf
        )
    )
    self.assertEqual(executable_tf(), 0)

  def test_reduces_federated_value_at_clients_to_equivalent_noarg_function(
      self,
  ):
    zero_proto, zero_type = tensorflow_computation_factory.create_constant(
        0, federated_language.TensorType(np.int32)
    )
    zero_compiled = federated_language.framework.CompiledComputation(
        zero_proto, type_signature=zero_type
    )
    zero = federated_language.framework.Call(zero_compiled, None)
    federated_value = federated_language.framework.create_federated_value(
        zero, federated_language.CLIENTS
    )
    federated_value_func = federated_language.framework.Lambda(
        None, None, federated_value
    )
    extracted_tf = compiler.consolidate_and_extract_local_processing(
        federated_value_func, DEFAULT_GRAPPLER_CONFIG
    )
    executable_tf = (
        federated_language.framework.ConcreteComputation.from_building_block(
            extracted_tf
        )
    )
    self.assertEqual(executable_tf(), 0)

  def test_reduces_generic_intrinsic_to_equivalent_tf_op(self):
    arg_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    arg = federated_language.framework.Reference('arg', arg_type)
    multiply_intrinsic = federated_language.framework.Intrinsic(
        federated_language.framework.GENERIC_MULTIPLY.uri,
        federated_language.FunctionType([arg_type, arg_type], arg_type),
    )
    multiply_fn = federated_language.framework.Lambda(
        'arg',
        arg_type,
        federated_language.framework.Call(
            multiply_intrinsic, federated_language.framework.Struct([arg, arg])
        ),
    )
    extracted_tf = compiler.consolidate_and_extract_local_processing(
        multiply_fn, DEFAULT_GRAPPLER_CONFIG
    )
    self.assertIsInstance(
        extracted_tf, federated_language.framework.CompiledComputation
    )
    executable_tf = (
        federated_language.framework.ConcreteComputation.from_building_block(
            extracted_tf
        )
    )
    for k in range(10):
      self.assertEqual(executable_tf(k), k * k)

  def test_reduces_lambda_returning_empty_tuple_to_tf(self):
    empty_tuple = federated_language.framework.Struct([])
    lam = federated_language.framework.Lambda('x', np.int32, empty_tuple)
    extracted_tf = compiler.consolidate_and_extract_local_processing(
        lam, DEFAULT_GRAPPLER_CONFIG
    )
    self.assertIsInstance(
        extracted_tf, federated_language.framework.CompiledComputation
    )


class CompileLocalComputationToTensorFlow(absltest.TestCase):

  def assert_compiles_to_tensorflow(
      self, comp: federated_language.framework.ComputationBuildingBlock
  ):
    result = compiler.compile_local_computation_to_tensorflow(comp)
    if isinstance(comp.type_signature, federated_language.FunctionType):
      if not isinstance(
          result, federated_language.framework.CompiledComputation
      ):
        raise ValueError(
            'Expected a `federated_language.framework.CompiledComputation`,'
            f' found {type(result)}.'
        )
    else:
      if not isinstance(result, federated_language.framework.Call):
        raise ValueError(
            'Expected a `federated_language.framework.Call`, found'
            f' {type(result)}.'
        )
      if not isinstance(
          result.function, federated_language.framework.CompiledComputation
      ):
        raise ValueError(
            'Expected a `federated_language.framework.CompiledComputation`,'
            f' found {type(result.function)}.'
        )
    federated_language.framework.assert_types_equivalent(
        comp.type_signature, result.type_signature
    )

  def test_returns_tf_computation_with_functional_type_lambda_no_block(self):
    param = federated_language.framework.Reference(
        'x', [('a', np.int32), ('b', np.float32)]
    )
    sel = federated_language.framework.Selection(source=param, index=0)
    tup = federated_language.framework.Struct([sel, sel, sel])
    lam = federated_language.framework.Lambda(
        param.name, param.type_signature, tup
    )
    self.assert_compiles_to_tensorflow(lam)

  def test_returns_tf_computation_with_functional_type_lambda_with_block(self):
    param = federated_language.framework.Reference(
        'x', [('a', np.int32), ('b', np.float32)]
    )
    block_to_param = federated_language.framework.Block([('x', param)], param)
    lam = federated_language.framework.Lambda(
        param.name, param.type_signature, block_to_param
    )
    self.assert_compiles_to_tensorflow(lam)

  def test_returns_tf_computation_with_functional_type_block_to_lambda_no_block(
      self,
  ):
    concrete_int_type = federated_language.TensorType(np.int32)
    param = federated_language.framework.Reference('x', np.float32)
    lam = federated_language.framework.Lambda(
        param.name, param.type_signature, param
    )
    unused_proto, unused_type = tensorflow_computation_factory.create_constant(
        1, concrete_int_type
    )
    unused_compiled = federated_language.framework.CompiledComputation(
        unused_proto, type_signature=unused_type
    )
    unused_int = federated_language.framework.Call(unused_compiled, None)
    blk_to_lam = federated_language.framework.Block([('y', unused_int)], lam)
    self.assert_compiles_to_tensorflow(blk_to_lam)

  def test_returns_tf_computation_with_functional_type_block_to_lambda_with_block(
      self,
  ):
    concrete_int_type = federated_language.TensorType(np.int32)
    param = federated_language.framework.Reference('x', np.float32)
    block_to_param = federated_language.framework.Block([('x', param)], param)
    lam = federated_language.framework.Lambda(
        param.name, param.type_signature, block_to_param
    )
    unused_proto, unused_type = tensorflow_computation_factory.create_constant(
        1, concrete_int_type
    )
    unused_compiled = federated_language.framework.CompiledComputation(
        unused_proto, type_signature=unused_type
    )
    unused_int = federated_language.framework.Call(unused_compiled, None)
    blk_to_lam = federated_language.framework.Block([('y', unused_int)], lam)
    self.assert_compiles_to_tensorflow(blk_to_lam)

  def test_returns_tf_computation_block_with_compiled_comp(self):
    concrete_int_type = federated_language.TensorType(np.int32)
    proto, type_signature = tensorflow_computation_factory.create_identity(
        concrete_int_type
    )
    tf_identity = federated_language.framework.CompiledComputation(
        proto, type_signature=type_signature
    )
    unused_proto, unused_type = tensorflow_computation_factory.create_constant(
        1, concrete_int_type
    )
    unused_compiled = federated_language.framework.CompiledComputation(
        unused_proto, type_signature=unused_type
    )
    unused_int = federated_language.framework.Call(unused_compiled, None)
    block_to_id = federated_language.framework.Block(
        [('x', unused_int)], tf_identity
    )
    self.assert_compiles_to_tensorflow(block_to_id)

  def test_returns_tf_computation_compiled_comp(self):
    concrete_int_type = federated_language.TensorType(np.int32)
    proto, type_signature = tensorflow_computation_factory.create_identity(
        concrete_int_type
    )
    tf_identity = federated_language.framework.CompiledComputation(
        proto, type_signature=type_signature
    )
    self.assert_compiles_to_tensorflow(tf_identity)

  def test_returns_called_tf_computation_with_truct(self):
    constant_tuple_type = federated_language.StructType([np.int32, np.float32])
    constant_proto, constant_type = (
        tensorflow_computation_factory.create_constant(1, constant_tuple_type)
    )
    constant_compiled = federated_language.framework.CompiledComputation(
        constant_proto, type_signature=constant_type
    )
    constant_tuple = federated_language.framework.Call(constant_compiled, None)
    sel = federated_language.framework.Selection(source=constant_tuple, index=0)
    tup = federated_language.framework.Struct([sel, sel, sel])
    self.assert_compiles_to_tensorflow(tup)

  def test_passes_on_tf(self):
    proto, type_signature = tensorflow_computation_factory.create_identity(
        federated_language.TensorType(np.int32)
    )
    tf_comp = federated_language.framework.CompiledComputation(
        proto, type_signature=type_signature
    )
    transformed = compiler.compile_local_computation_to_tensorflow(tf_comp)
    self.assertEqual(tf_comp, transformed)

  def test_raises_on_xla(self):
    function_type = federated_language.FunctionType(
        federated_language.TensorType(np.int32),
        federated_language.TensorType(np.int32),
    )
    empty_xla_computation_proto = computation_pb2.Computation(
        type=function_type.to_proto(),
        xla=computation_pb2.Xla(),
    )

    compiled_comp = federated_language.framework.CompiledComputation(
        proto=empty_xla_computation_proto
    )

    with self.assertRaises(compiler.XlaToTensorFlowError):
      compiler.compile_local_computation_to_tensorflow(compiled_comp)

  def test_generates_tf_with_lambda(self):
    ref_to_x = federated_language.framework.Reference(
        'x', federated_language.StructType([np.int32, np.float32])
    )
    identity_lambda = federated_language.framework.Lambda(
        ref_to_x.name, ref_to_x.type_signature, ref_to_x
    )
    self.assert_compiles_to_tensorflow(identity_lambda)

  def test_generates_tf_with_block(self):
    ref_to_x = federated_language.framework.Reference(
        'x', federated_language.StructType([np.int32, np.float32])
    )
    identity_lambda = federated_language.framework.Lambda(
        ref_to_x.name, ref_to_x.type_signature, ref_to_x
    )
    zero_proto, zero_type = tensorflow_computation_factory.create_constant(
        0, federated_language.StructType([np.int32, np.float32])
    )
    zero_compiled = federated_language.framework.CompiledComputation(
        zero_proto, type_signature=zero_type
    )
    zero = federated_language.framework.Call(zero_compiled, None)
    ref_to_z = federated_language.framework.Reference(
        'z', [np.int32, np.float32]
    )
    called_lambda_on_z = federated_language.framework.Call(
        identity_lambda, ref_to_z
    )
    blk = federated_language.framework.Block([('z', zero)], called_lambda_on_z)
    self.assert_compiles_to_tensorflow(blk)

  def test_generates_tf_with_sequence_type(self):
    ref_to_x = federated_language.framework.Reference(
        'x', federated_language.SequenceType([np.int32, np.float32])
    )
    identity_lambda = federated_language.framework.Lambda(
        ref_to_x.name, ref_to_x.type_signature, ref_to_x
    )
    self.assert_compiles_to_tensorflow(identity_lambda)

  def test_returns_result_with_literal(self):
    comp = federated_language.framework.Literal(
        1, federated_language.TensorType(np.int32)
    )

    result = compiler.compile_local_computation_to_tensorflow(comp)

    self.assertIsInstance(result, federated_language.framework.Call)
    self.assertIsInstance(
        result.function, federated_language.framework.CompiledComputation
    )
    federated_language.framework.assert_types_equivalent(
        comp.type_signature, result.type_signature
    )


class CompileLocalSubcomputationsToTensorFlowTest(absltest.TestCase):

  def test_leaves_federated_comp_alone(self):
    ref_to_federated_x = federated_language.framework.Reference(
        'x',
        federated_language.FederatedType(np.int32, federated_language.SERVER),
    )
    identity_lambda = federated_language.framework.Lambda(
        ref_to_federated_x.name,
        ref_to_federated_x.type_signature,
        ref_to_federated_x,
    )
    transformed = compiler.compile_local_subcomputations_to_tensorflow(
        identity_lambda
    )
    self.assertEqual(transformed, identity_lambda)

  def test_compiles_lambda_under_federated_comp_to_tf(self):
    ref_to_x = federated_language.framework.Reference(
        'x', federated_language.StructType([np.int32, np.float32])
    )
    identity_lambda = federated_language.framework.Lambda(
        ref_to_x.name, ref_to_x.type_signature, ref_to_x
    )
    any_proto = building_block_test_utils.create_any_proto_from_array(
        np.array(1)
    )
    federated_data = federated_language.framework.Data(
        any_proto,
        federated_language.FederatedType(
            federated_language.StructType([np.int32, np.float32]),
            federated_language.SERVER,
        ),
    )
    applied = federated_language.framework.create_federated_apply(
        identity_lambda, federated_data
    )

    transformed = compiler.compile_local_subcomputations_to_tensorflow(applied)

    self.assertIsInstance(transformed, federated_language.framework.Call)
    self.assertIsInstance(
        transformed.function, federated_language.framework.Intrinsic
    )
    self.assertIsInstance(
        transformed.argument[0],
        federated_language.framework.CompiledComputation,
    )
    self.assertEqual(transformed.argument[1], federated_data)
    self.assertEqual(
        transformed.argument[0].type_signature, identity_lambda.type_signature
    )

  def test_leaves_local_comp_with_unbound_reference_alone(self):
    ref_to_x = federated_language.framework.Reference(
        'x', [np.int32, np.float32]
    )
    ref_to_z = federated_language.framework.Reference(
        'z', [np.int32, np.float32]
    )
    lambda_with_unbound_ref = federated_language.framework.Lambda(
        ref_to_x.name, ref_to_x.type_signature, ref_to_z
    )
    transformed = compiler.compile_local_subcomputations_to_tensorflow(
        lambda_with_unbound_ref
    )

    self.assertEqual(transformed, lambda_with_unbound_ref)


class ConcatenateFunctionOutputsTest(absltest.TestCase):

  def test_raises_on_non_lambda_args(self):
    reference = federated_language.framework.Reference('x', np.int32)
    tff_lambda = federated_language.framework.Lambda('x', np.int32, reference)
    with self.assertRaises(TypeError):
      compiler.concatenate_function_outputs(tff_lambda, reference)
    with self.assertRaises(TypeError):
      compiler.concatenate_function_outputs(reference, tff_lambda)

  def test_raises_on_non_unique_names(self):
    reference = federated_language.framework.Reference('x', np.int32)
    good_lambda = federated_language.framework.Lambda('x', np.int32, reference)
    bad_lambda = federated_language.framework.Lambda('x', np.int32, good_lambda)
    with self.assertRaises(ValueError):
      compiler.concatenate_function_outputs(good_lambda, bad_lambda)
    with self.assertRaises(ValueError):
      compiler.concatenate_function_outputs(bad_lambda, good_lambda)

  def test_raises_on_different_parameter_types(self):
    int_reference = federated_language.framework.Reference('x', np.int32)
    int_lambda = federated_language.framework.Lambda(
        'x', np.int32, int_reference
    )
    float_reference = federated_language.framework.Reference('x', np.float32)
    float_lambda = federated_language.framework.Lambda(
        'x', np.float32, float_reference
    )
    with self.assertRaises(TypeError):
      compiler.concatenate_function_outputs(int_lambda, float_lambda)

  def test_parameters_are_mapped_together(self):
    x_reference = federated_language.framework.Reference('x', np.int32)
    x_lambda = federated_language.framework.Lambda('x', np.int32, x_reference)
    y_reference = federated_language.framework.Reference('y', np.int32)
    y_lambda = federated_language.framework.Lambda('y', np.int32, y_reference)
    concatenated = compiler.concatenate_function_outputs(x_lambda, y_lambda)
    parameter_name = concatenated.parameter_name

    def _raise_on_other_name_reference(comp):
      if (
          isinstance(comp, federated_language.framework.Reference)
          and comp.name != parameter_name
      ):
        raise ValueError
      return comp, True

    federated_language.framework.check_has_unique_names(concatenated)
    federated_language.framework.transform_postorder(
        concatenated, _raise_on_other_name_reference
    )

  def test_concatenates_identities(self):
    x_reference = federated_language.framework.Reference('x', np.int32)
    x_lambda = federated_language.framework.Lambda('x', np.int32, x_reference)
    y_reference = federated_language.framework.Reference('y', np.int32)
    y_lambda = federated_language.framework.Lambda('y', np.int32, y_reference)
    concatenated = compiler.concatenate_function_outputs(x_lambda, y_lambda)
    self.assertEqual(str(concatenated), '(y -> <y,y>)')


if __name__ == '__main__':
  context = _create_test_context()
  federated_language.framework.set_default_context(context)
  absltest.main()
