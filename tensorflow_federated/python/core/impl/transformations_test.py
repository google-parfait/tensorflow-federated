# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import transformation_utils
from tensorflow_federated.python.core.impl import transformations
from tensorflow_federated.python.core.impl import type_utils

RENAME_PREFIX = '_variable'


def _create_called_federated_aggregate(value, zero, accumulate, merge, report):
  r"""Creates a to call a federated aggregate.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp, Comp, Comp, Comp]

  Args:
    value: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the value of the federated aggregate intrinsic.
    zero: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      zero of the federated aggregate intrinsic.
    accumulate: A `computation_building_blocks.ComputationBuildingBlock` to use
      as the accumulate of the federated aggregate intrinsic.
    merge: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the merge of the federated aggregate intrinsic.
    report: A `computation_building_blocks.ComputationBuildingBlock` to use as
      the report of the federated aggregate intrinsic.

  Returns:
    A `computation_building_blocks.Call`.
  """
  py_typecheck.check_type(value,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(zero,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(accumulate,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(merge,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(report,
                          computation_building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(report.type_signature.result,
                                                placements.SERVER, True)
  intrinsic_type = computation_types.FunctionType((
      value.type_signature,
      zero.type_signature,
      accumulate.type_signature,
      merge.type_signature,
      report.type_signature,
  ), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_AGGREGATE.uri, intrinsic_type)
  arg = computation_building_blocks.Tuple((
      value,
      zero,
      accumulate,
      merge,
      report,
  ))
  return computation_building_blocks.Call(intrinsic, arg)


def _create_called_federated_apply(fn, arg):
  r"""Creates a to call a federated apply.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A functional `computation_building_blocks.ComputationBuildingBlock` to
      use as the function.
    arg: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      argument.

  Returns:
    A `computation_building_blocks.Call`.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  if not type_utils.is_assignable_from(fn.parameter_type,
                                       arg.type_signature.member):
    raise TypeError(
        'The parameter of the function is of type {}, and the argument is of '
        'an incompatible type {}.'.format(
            str(fn.parameter_type), str(arg.type_signature.member)))
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                placements.SERVER, True)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_APPLY.uri, intrinsic_type)
  tup = computation_building_blocks.Tuple((fn, arg))
  return computation_building_blocks.Call(intrinsic, tup)


def _create_called_sequence_map(fn, arg):
  r"""Creates a to call a sequence map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A functional `computation_building_blocks.ComputationBuildingBlock` to
      use as the function.
    arg: A `computation_building_blocks.ComputationBuildingBlock` to use as the
      argument.

  Returns:
    A `computation_building_blocks.Call`.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  if not type_utils.is_assignable_from(fn.parameter_type,
                                       arg.type_signature.element):
    raise TypeError(
        'The parameter of the function is of type {}, and the argument is of '
        'an incompatible type {}.'.format(
            str(fn.parameter_type), str(arg.type_signature.element)))
  result_type = computation_types.SequenceType(fn.type_signature.result)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.SEQUENCE_MAP.uri, intrinsic_type)
  tup = computation_building_blocks.Tuple((fn, arg))
  return computation_building_blocks.Call(intrinsic, tup)


def _create_chained_calls(functions, arg):
  r"""Creates a chain of `n` calls.

       Call
      /    \
  Comp      ...
               \
                Call
               /    \
           Comp      Comp

  The first functional computation in `functions` must have a parameter type
  that is assignable from the type of `arg`, each other functional computation
  in `functions` must have a parameter type that is assignable from the previous
  functional computations result type.

  Args:
    functions: A Python list of functional computations.
    arg: A `computation_building_blocks.ComputationBuildingBlock`.

  Returns:
    A `computation_building_blocks.Call`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  for fn in functions:
    py_typecheck.check_type(
        fn, computation_building_blocks.ComputationBuildingBlock)
    if not type_utils.is_assignable_from(fn.parameter_type, arg.type_signature):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature)))
    call = computation_building_blocks.Call(fn, arg)
    arg = call
  return call


def _create_chained_called_federated_map(functions, arg):
  r"""Creates a chain of `n` calls to federated map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]
                            \
                             ...
                                \
                                 Call
                                /    \
                       Intrinsic      Tuple
                                      |
                                      [Comp, Comp]

  The first functional computation in `functions` must have a parameter type
  that is assignable from the type of `arg`, each other functional computation
  in `functions` must have a parameter type that is assignable from the previous
  functional computations result type.

  Args:
    functions: A Python list of functional computations.
    arg: A `computation_building_blocks.ComputationBuildingBlock`.

  Returns:
    A `computation_building_blocks.Call`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  for fn in functions:
    py_typecheck.check_type(
        fn, computation_building_blocks.ComputationBuildingBlock)
    if not type_utils.is_assignable_from(fn.parameter_type,
                                         arg.type_signature.member):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature.member)))
    call = computation_constructing_utils.create_federated_map(fn, arg)
    arg = call
  return call


def _create_lambda_to_identity(parameter_name, parameter_type):
  r"""Creates a lambda to return the argument.

  Lambda(x)
           \
            Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.

  Returns:
    A `computation_building_blocks.Lambda`.
  """
  ref = computation_building_blocks.Reference(parameter_name, parameter_type)
  return computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)


def _create_dummy_block(comp):
  r"""Creates a dummy block.

                Block
               /     \
  local=Data(x)       Comp

  Args:
    comp: A `computation_building_blocks.ComputationBuildingBlock`.

  Returns:
    A dummy `computation_building_blocks.Block`.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  data = computation_building_blocks.Data('x', tf.int32)
  return computation_building_blocks.Block([('local', data)], comp)


def _create_lambda_to_dummy_intrinsic(uri='dummy', type_spec=tf.int32):
  r"""Creates a lambda to call a dummy intrinsic.

  Lambda(x)
           \
            Call
           /    \
  Intrinsic      Ref(x)

  Args:
    uri: The URI of the intrinsic.
    type_spec: The type of the parameter.

  Returns:
    A `computation_building_blocks.Lambda`.
  """
  py_typecheck.check_type(type_spec, tf.dtypes.DType)
  intrinsic_type = computation_types.FunctionType(type_spec, type_spec)
  intrinsic = computation_building_blocks.Intrinsic(uri, intrinsic_type)
  ref = computation_building_blocks.Reference('x', type_spec)
  call = computation_building_blocks.Call(intrinsic, ref)
  return computation_building_blocks.Lambda(ref.name, ref.type_signature, call)


def _create_lambda_to_dummy_cast(parameter_type, result_type):
  r"""Creates a lambda to cast from `parameter_type` to `result_type`.

  Lambda(x)
           \
            Data(y)

  Args:
    parameter_type: The type of the argument.
    result_type: The type to cast the argument to.

  Returns:
    A `computation_building_blocks.Lambda`.
  """
  py_typecheck.check_type(parameter_type, tf.dtypes.DType)
  py_typecheck.check_type(result_type, tf.dtypes.DType)
  arg = computation_building_blocks.Data('y', result_type)
  return computation_building_blocks.Lambda('x', parameter_type, arg)


def _create_dummy_called_federated_aggregate():
  value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
  value = computation_building_blocks.Data('v', value_type)
  zero = computation_building_blocks.Data('z', tf.int32)
  accumulate_type = computation_types.FunctionType((tf.int32, tf.int32),
                                                   tf.int32)
  accumulate_result = computation_building_blocks.Data('a', tf.int32)
  accumulate = computation_building_blocks.Lambda('x', accumulate_type,
                                                  accumulate_result)
  merge_type = computation_types.FunctionType((tf.int32, tf.int32), tf.int32)
  merge_result = computation_building_blocks.Data('m', tf.int32)
  merge = computation_building_blocks.Lambda('x', merge_type, merge_result)
  report_type = computation_types.FederatedType(tf.int32, placements.SERVER,
                                                True)
  report_result = computation_building_blocks.Data('r', tf.int32)
  report = computation_building_blocks.Lambda('x', report_type, report_result)
  return _create_called_federated_aggregate(value, zero, accumulate, merge,
                                            report)


def _create_dummy_called_federated_apply(parameter_name='x',
                                         parameter_type=tf.int32,
                                         argument_name='y'):
  fn = _create_lambda_to_identity(parameter_name, parameter_type)
  arg_type = computation_types.FederatedType(parameter_type, placements.SERVER,
                                             True)
  arg = computation_building_blocks.Data(argument_name, arg_type)
  return _create_called_federated_apply(fn, arg)


def _create_dummy_called_federated_map(parameter_name='x',
                                       parameter_type=tf.int32,
                                       argument_name='y'):
  fn = _create_lambda_to_identity(parameter_name, parameter_type)
  arg_type = computation_types.FederatedType(parameter_type, placements.CLIENTS,
                                             False)
  arg = computation_building_blocks.Data(argument_name, arg_type)
  return computation_constructing_utils.create_federated_map(fn, arg)


def _create_dummy_called_sequence_map(parameter_name='x',
                                      parameter_type=tf.int32,
                                      argument_name='y'):
  fn = _create_lambda_to_identity(parameter_name, parameter_type)
  arg_type = computation_types.SequenceType(parameter_type)
  arg = computation_building_blocks.Data(argument_name, arg_type)
  return _create_called_sequence_map(fn, arg)


def _create_dummy_called_intrinsic(uri='dummy', type_spec=tf.int32):
  py_typecheck.check_type(type_spec, tf.dtypes.DType)
  intrinsic_type = computation_types.FunctionType(type_spec, type_spec)
  intrinsic = computation_building_blocks.Intrinsic(uri, intrinsic_type)
  arg = computation_building_blocks.Data('x', type_spec)
  return computation_building_blocks.Call(intrinsic, arg)


def _has_unique_names(comp):
  """Checks that each variable of `comp` is bound at most once."""
  names = set()
  # TODO(b/129791812): Cleanup Python 2 and 3 compatibility
  unique = [True]

  def _transform(comp):
    if unique[0]:
      if isinstance(comp, computation_building_blocks.Block):
        for name, _ in comp.locals:
          if name in names:
            unique[0] = False
          names.add(name)
      elif isinstance(comp, computation_building_blocks.Lambda):
        if comp.parameter_name in names:
          unique[0] = False
        names.add(comp.parameter_name)
    return comp, False

  transformation_utils.transform_postorder(comp, _transform)

  return unique[0]


class TransformationsTest(parameterized.TestCase):

  def test_replace_compiled_computations_names_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.replace_compiled_computations_names_with_unique_names(
          None)

  def test_replace_compiled_computations_names_replaces_name(self):
    fn = lambda: tf.constant(1)
    tf_comp, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
        fn, None, context_stack_impl.context_stack)
    compiled_comp = computation_building_blocks.CompiledComputation(tf_comp)
    comp = compiled_comp

    transformed_comp, modified = transformations.replace_compiled_computations_names_with_unique_names(
        comp)

    self.assertNotEqual(transformed_comp._name, comp._name)
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_compiled_computations_names_replaces_multiple_names(self):
    elements = []
    for _ in range(10):
      fn = lambda: tf.constant(1)
      tf_comp, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
          fn, None, context_stack_impl.context_stack)
      compiled_comp = computation_building_blocks.CompiledComputation(tf_comp)
      elements.append(compiled_comp)
    compiled_comps = computation_building_blocks.Tuple(elements)
    comp = compiled_comps

    transformed_comp, modified = transformations.replace_compiled_computations_names_with_unique_names(
        comp)

    comp_names = [element._name for element in comp]
    transformed_comp_names = [element._name for element in transformed_comp]
    self.assertNotEqual(transformed_comp_names, comp_names)
    self.assertEqual(
        len(transformed_comp_names), len(set(transformed_comp_names)),
        'The transformed computation names are not unique: {}.'.format(
            transformed_comp_names))
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_compiled_computations_names_does_not_replace_other_name(
      self):
    comp = computation_building_blocks.Reference('name', tf.int32)

    transformed_comp, modified = transformations.replace_compiled_computations_names_with_unique_names(
        comp)

    self.assertEqual(transformed_comp._name, comp._name)
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_replace_intrinsic_raises_type_error_none_comp(self):
    uri = 'dummy'
    body = lambda x: x

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(
          None, uri, body, context_stack_impl.context_stack)

  def test_replace_intrinsic_raises_type_error_none_uri(self):
    comp = _create_lambda_to_dummy_intrinsic()
    body = lambda x: x

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(
          comp, None, body, context_stack_impl.context_stack)

  def test_replace_intrinsic_raises_type_error_none_body(self):
    comp = _create_lambda_to_dummy_intrinsic()
    uri = 'dummy'

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(
          comp, uri, None, context_stack_impl.context_stack)

  def test_replace_intrinsic_raises_type_error_none_context_stack(self):
    comp = _create_lambda_to_dummy_intrinsic()
    uri = 'dummy'
    body = lambda x: x

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(comp, uri, body, None)

  def test_replace_intrinsic_replaces_intrinsic(self):
    comp = _create_lambda_to_dummy_intrinsic()
    uri = 'dummy'
    body = lambda x: x

    transformed_comp, modified = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(comp.tff_repr, '(x -> dummy(x))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(x -> (dummy_arg -> dummy_arg)(x))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_intrinsic_replaces_nested_intrinsic(self):
    fn = _create_lambda_to_dummy_intrinsic()
    block = _create_dummy_block(fn)
    comp = block
    uri = 'dummy'
    body = lambda x: x

    transformed_comp, modified = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(comp.tff_repr, '(let local=x in (x -> dummy(x)))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let local=x in (x -> (dummy_arg -> dummy_arg)(x)))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_intrinsic_replaces_chained_intrinsics(self):
    fn = _create_lambda_to_dummy_intrinsic(type_spec=tf.int32)
    arg = computation_building_blocks.Data('x', tf.int32)
    call = _create_chained_calls([fn, fn], arg)
    comp = call
    uri = 'dummy'
    body = lambda x: x

    transformed_comp, modified = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(comp.tff_repr, '(x -> dummy(x))((x -> dummy(x))(x))')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> (dummy_arg -> dummy_arg)(x))((x -> (dummy_arg -> dummy_arg)(x))(x))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_intrinsic_does_not_replace_other_intrinsic(self):
    comp = _create_lambda_to_dummy_intrinsic()
    uri = 'other'
    body = lambda x: x

    transformed_comp, modified = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(x -> dummy(x))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_replace_called_lambda_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.replace_called_lambda_with_block(None)

  def test_replace_called_lambda_replaces_called_lambda(self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg = computation_building_blocks.Data('y', tf.int32)
    call = computation_building_blocks.Call(fn, arg)
    comp = call

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(comp.tff_repr, '(x -> x)(y)')
    self.assertEqual(transformed_comp.tff_repr, '(let x=y in x)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_called_lambda_replaces_nested_called_lambda(self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg = computation_building_blocks.Data('y', tf.int32)
    call = computation_building_blocks.Call(fn, arg)
    block = _create_dummy_block(call)
    comp = block

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(comp.tff_repr, '(let local=x in (x -> x)(y))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let local=x in (let x=y in x))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_called_lambda_replaces_chained_called_lambdas(self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg = computation_building_blocks.Data('y', tf.int32)
    call = _create_chained_calls([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(comp.tff_repr, '(x -> x)((x -> x)(y))')
    self.assertEqual(transformed_comp.tff_repr, '(let x=(let x=y in x) in x)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_called_lambda_does_not_replace_uncalled_lambda(self):
    fn = _create_lambda_to_identity('x', tf.int32)
    comp = fn

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(x -> x)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_replace_called_lambda_does_not_replace_separated_called_lambda(self):
    fn = _create_lambda_to_identity('x', tf.int32)
    block = _create_dummy_block(fn)
    arg = computation_building_blocks.Data('y', tf.int32)
    call = computation_building_blocks.Call(block, arg)
    comp = call

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(let local=x in (x -> x))(y)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_remove_mapped_or_applied_identity_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.remove_mapped_or_applied_identity(None)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_apply',
       intrinsic_defs.FEDERATED_APPLY.uri,
       _create_dummy_called_federated_apply),
      ('federated_map',
       intrinsic_defs.FEDERATED_MAP.uri,
       _create_dummy_called_federated_map),
      ('sequence_map',
       intrinsic_defs.SEQUENCE_MAP.uri,
       _create_dummy_called_sequence_map))
  # pyformat: enable
  def test_remove_mapped_or_applied_identity_removes_identity(
      self, uri, comp_factory):
    call = comp_factory()
    comp = call

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.tff_repr, '{}(<(x -> x),y>)'.format(uri))
    self.assertEqual(transformed_comp.tff_repr, 'y')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_remove_mapped_or_applied_identity_replaces_federated_maps_with_named_result(
      self):
    parameter_type = [('a', tf.int32), ('b', tf.int32)]
    fn = _create_lambda_to_identity('x', parameter_type)
    arg_type = computation_types.FederatedType(parameter_type,
                                               placements.CLIENTS, False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = computation_constructing_utils.create_federated_map(fn, arg)
    comp = call

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.tff_repr, 'federated_map(<(x -> x),y>)')
    self.assertEqual(transformed_comp.tff_repr, 'y')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_remove_mapped_or_applied_identity_removes_nested_federated_map(self):
    call = _create_dummy_called_federated_map()
    block = _create_dummy_block(call)
    comp = block

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.tff_repr,
                     '(let local=x in federated_map(<(x -> x),y>))')
    self.assertEqual(transformed_comp.tff_repr, '(let local=x in y)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_remove_mapped_or_applied_identity_removes_chained_federated_maps(
      self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = _create_chained_called_federated_map([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.tff_repr,
                     'federated_map(<(x -> x),federated_map(<(x -> x),y>)>)')
    self.assertEqual(transformed_comp.tff_repr, 'y')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_remove_mapped_or_applied_identity_does_not_remove_dummy_intrinsic(
      self):
    comp = _create_dummy_called_intrinsic()

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, 'dummy(x)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_remove_mapped_or_applied_identity_does_not_remove_called_lambda(
      self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg = computation_building_blocks.Data('y', tf.int32)
    call = computation_building_blocks.Call(fn, arg)
    comp = call

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(x -> x)(y)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_replace_chained_federated_maps_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.replace_chained_federated_maps_with_federated_map(None)

  def test_replace_chained_federated_maps_replaces_federated_maps(self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = _create_chained_called_federated_map([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(comp.tff_repr,
                     'federated_map(<(x -> x),federated_map(<(x -> x),y>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(x -> x),(x -> x)> in (arg -> fn[1](fn[0](arg)))),y>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_chained_federated_maps_replaces_federated_maps_with_different_names(
      self):
    fn_1 = _create_lambda_to_identity('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('b', arg_type)
    fn_2 = _create_lambda_to_identity('c', tf.int32)
    call = _create_chained_called_federated_map([fn_1, fn_2], arg)
    comp = call

    transformed_comp, modified = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(comp.tff_repr,
                     'federated_map(<(c -> c),federated_map(<(a -> a),b>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(a -> a),(c -> c)> in (arg -> fn[1](fn[0](arg)))),b>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_chained_federated_maps_replaces_federated_maps_with_different_types(
      self):
    fn_1 = _create_lambda_to_dummy_cast(tf.int32, tf.float32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    fn_2 = _create_lambda_to_identity('x', tf.float32)
    call = _create_chained_called_federated_map([fn_1, fn_2], arg)
    comp = call

    transformed_comp, modified = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(comp.tff_repr,
                     'federated_map(<(x -> x),federated_map(<(x -> y),y>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(x -> y),(x -> x)> in (arg -> fn[1](fn[0](arg)))),y>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_chained_federated_maps_replaces_federated_maps_with_named_result(
      self):
    parameter_type = [('a', tf.int32), ('b', tf.int32)]
    fn = _create_lambda_to_identity('x', parameter_type)
    arg_type = computation_types.FederatedType(parameter_type,
                                               placements.CLIENTS, False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = _create_chained_called_federated_map([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(comp.tff_repr,
                     'federated_map(<(x -> x),federated_map(<(x -> x),y>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(x -> x),(x -> x)> in (arg -> fn[1](fn[0](arg)))),y>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_chained_federated_maps_replaces_federated_maps_with_unbound_references(
      self):
    ref = computation_building_blocks.Reference('arg', tf.int32)
    fn = computation_building_blocks.Lambda('x', tf.int32, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = _create_chained_called_federated_map([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_map(<(x -> arg),federated_map(<(x -> arg),y>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(x -> arg),(x -> arg)> in (arg -> fn[1](fn[0](arg)))),y>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_chained_federated_maps_replaces_nested_federated_maps(self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = _create_chained_called_federated_map([fn, fn], arg)
    block = _create_dummy_block(call)
    comp = block

    transformed_comp, modified = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '(let local=x in federated_map(<(x -> x),federated_map(<(x -> x),y>)>))'
    )
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let local=x in federated_map(<(let fn=<(x -> x),(x -> x)> in (arg -> fn[1](fn[0](arg)))),y>))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_chained_federated_maps_replaces_multiple_chained_federated_maps(
      self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = _create_chained_called_federated_map([fn, fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_map(<(x -> x),federated_map(<(x -> x),federated_map(<(x -> x),y>)>)>)'
    )
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(let fn=<(x -> x),(x -> x)> in (arg -> fn[1](fn[0](arg)))),(x -> x)> in (arg -> fn[1](fn[0](arg)))),y>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_chained_federated_maps_does_not_replace_one_federated_map(
      self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = computation_constructing_utils.create_federated_map(fn, arg)
    comp = call

    transformed_comp, modified = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, 'federated_map(<(x -> x),y>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_replace_chained_federated_maps_does_not_replace_separated_federated_maps(
      self):
    fn = _create_lambda_to_identity('x', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    call_1 = computation_constructing_utils.create_federated_map(fn, arg)
    block = _create_dummy_block(call_1)
    call_2 = computation_constructing_utils.create_federated_map(fn, block)
    comp = call_2

    transformed_comp, modified = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(x -> x),(let local=x in federated_map(<(x -> x),y>))>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_replace_tuple_intrinsics_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.replace_tuple_intrinsics_with_intrinsic(None)

  def test_replace_tuple_intrinsics_replaces_federated_aggregates(self):
    elements = [_create_dummy_called_federated_aggregate() for _ in range(2)]
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_aggregate(<v,z,(x -> a),(x -> m),(x -> r)>),federated_aggregate(<v,z,(x -> a),(x -> m),(x -> r)>)>'
    )
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_aggregate(<<v,v>,<z,z>,(let fn=<(x -> a),(x -> a)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),(let fn=<(x -> m),(x -> m)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),(let fn=<(x -> r),(x -> r)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>))>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_replaces_federated_maps(self):
    elements = [_create_dummy_called_federated_map() for _ in range(2)]
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(x -> x),y>),federated_map(<(x -> x),y>)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(x -> x),(x -> x)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),<y,y>>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_replaces_federated_maps_with_different_names(
      self):
    elements = (
        _create_dummy_called_federated_map(
            parameter_name='a', argument_name='b'),
        _create_dummy_called_federated_map(
            parameter_name='c', argument_name='d'),
    )
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(a -> a),b>),federated_map(<(c -> c),d>)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(a -> a),(c -> c)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),<b,d>>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_replaces_federated_maps_with_different_types(
      self):
    elements = (
        _create_dummy_called_federated_map(parameter_type=tf.int32),
        _create_dummy_called_federated_map(parameter_type=tf.float32),
    )
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(x -> x),y>),federated_map(<(x -> x),y>)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(x -> x),(x -> x)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),<y,y>>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_replaces_federated_maps_with_named_result(
      self):
    parameter_type = [('a', tf.int32), ('b', tf.int32)]
    fn = _create_lambda_to_identity('x', parameter_type)
    arg_type = computation_types.FederatedType(parameter_type,
                                               placements.CLIENTS, False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = computation_constructing_utils.create_federated_map(fn, arg)
    elements = [call, call]
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(x -> x),y>),federated_map(<(x -> x),y>)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(x -> x),(x -> x)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),<y,y>>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_replaces_federated_maps_with_unbound_reference(
      self):
    ref = computation_building_blocks.Reference('arg', tf.int32)
    fn = computation_building_blocks.Lambda('x', tf.int32, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS,
                                               False)
    arg = computation_building_blocks.Data('y', arg_type)
    call = computation_constructing_utils.create_federated_map(fn, arg)
    elements = [call, call]
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(x -> arg),y>),federated_map(<(x -> arg),y>)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(x -> arg),(x -> arg)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),<y,y>>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_replaces_named_federated_maps(self):
    elements = (
        ('a', _create_dummy_called_federated_map()),
        ('b', _create_dummy_called_federated_map()),
    )
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '<a=federated_map(<(x -> x),y>),b=federated_map(<(x -> x),y>)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<a=(x -> x),b=(x -> x)> in (arg -> <a=fn[0](arg[0]),b=fn[1](arg[1])>)),<a=y,b=y>>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_replaces_nested_federated_maps(self):
    elements = [_create_dummy_called_federated_map() for _ in range(2)]
    calls = computation_building_blocks.Tuple(elements)
    block = _create_dummy_block(calls)
    comp = block

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '(let local=x in <federated_map(<(x -> x),y>),federated_map(<(x -> x),y>)>)'
    )
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let local=x in federated_map(<(let fn=<(x -> x),(x -> x)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),<y,y>>))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_replaces_multiple_federated_maps(self):
    comp_elements = []
    for _ in range(2):
      call_elements = [_create_dummy_called_federated_map() for _ in range(2)]
      calls = computation_building_blocks.Tuple(call_elements)
      comp_elements.append(calls)
    comps = computation_building_blocks.Tuple(comp_elements)
    comp = comps

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '<<federated_map(<(x -> x),y>),federated_map(<(x -> x),y>)>,<federated_map(<(x -> x),y>),federated_map(<(x -> x),y>)>>'
    )
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(let fn=<(x -> x),(x -> x)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),(let fn=<(x -> x),(x -> x)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>))> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),<<y,y>,<y,y>>>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_replaces_one_federated_map(self):
    elements = (_create_dummy_called_federated_map(),)
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(comp.tff_repr, '<federated_map(<(x -> x),y>)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(x -> x)> in (arg -> <fn[0](arg[0])>)),<y>>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replace_tuple_intrinsics_does_not_replace_different_intrinsics(self):
    elements = (
        _create_dummy_called_federated_aggregate(),
        _create_dummy_called_federated_map(),
    )
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(
        transformed_comp.tff_repr,
        '<federated_aggregate(<v,z,(x -> a),(x -> m),(x -> r)>),federated_map(<(x -> x),y>)>'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_replace_tuple_intrinsics_does_not_replace_dummy_intrinsics(self):
    elements = [_create_dummy_called_intrinsic() for _ in range(2)]
    calls = computation_building_blocks.Tuple(elements)
    comp = calls

    transformed_comp, modified = transformations.replace_tuple_intrinsics_with_intrinsic(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '<dummy(x),dummy(x)>')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_merge_chained_blocks_fails_on_none(self):
    with self.assertRaises(TypeError):
      transformations.merge_chained_blocks(None)

  def test_merge_chained_blocks_single_level_of_nesting(self):
    input1 = computation_building_blocks.Reference('input1', tf.int32)
    result = computation_building_blocks.Reference('result', tf.int32)
    block1 = computation_building_blocks.Block([('result', input1)], result)
    input2 = computation_building_blocks.Data('input2', tf.int32)
    block2 = computation_building_blocks.Block([('input1', input2)], block1)
    self.assertEqual(block2.tff_repr,
                     '(let input1=input2 in (let result=input1 in result))')
    merged_blocks, modified = transformations.merge_chained_blocks(block2)
    self.assertEqual(merged_blocks.tff_repr,
                     '(let input1=input2,result=input1 in result)')
    self.assertTrue(modified)

  def test_merge_chained_blocks_leaves_names(self):
    input1 = computation_building_blocks.Data('input1', tf.int32)
    result_tuple = computation_building_blocks.Tuple([
        ('a', computation_building_blocks.Data('result_a', tf.int32)),
        ('b', computation_building_blocks.Data('result_b', tf.int32))
    ])
    block1 = computation_building_blocks.Block([('x', input1)], result_tuple)
    result_block = block1
    input2 = computation_building_blocks.Data('input2', tf.int32)
    block2 = computation_building_blocks.Block([('y', input2)], result_block)
    self.assertEqual(
        block2.tff_repr,
        '(let y=input2 in (let x=input1 in <a=result_a,b=result_b>))')
    merged, modified = transformations.merge_chained_blocks(block2)
    self.assertEqual(merged.tff_repr,
                     '(let y=input2,x=input1 in <a=result_a,b=result_b>)')
    self.assertTrue(modified)

  def test_merge_chained_blocks_leaves_separated_chained_blocks_alone(self):
    input1 = computation_building_blocks.Data('input1', tf.int32)
    result = computation_building_blocks.Data('result', tf.int32)
    block1 = computation_building_blocks.Block([('x', input1)], result)
    result_block = block1
    result_tuple = computation_building_blocks.Tuple([result_block])
    input2 = computation_building_blocks.Data('input2', tf.int32)
    block2 = computation_building_blocks.Block([('y', input2)], result_tuple)
    self.assertEqual(block2.tff_repr,
                     '(let y=input2 in <(let x=input1 in result)>)')
    merged, modified = transformations.merge_chained_blocks(block2)
    self.assertEqual(merged.tff_repr,
                     '(let y=input2 in <(let x=input1 in result)>)')
    self.assertFalse(modified)

  def test_merge_chained_blocks_two_levels_of_nesting(self):
    input1 = computation_building_blocks.Reference('input1', tf.int32)
    result = computation_building_blocks.Reference('result', tf.int32)
    block1 = computation_building_blocks.Block([('result', input1)], result)
    input2 = computation_building_blocks.Reference('input2', tf.int32)
    block2 = computation_building_blocks.Block([('input1', input2)], block1)
    input3 = computation_building_blocks.Data('input3', tf.int32)
    block3 = computation_building_blocks.Block([('input2', input3)], block2)
    self.assertEqual(
        block3.tff_repr,
        '(let input2=input3 in (let input1=input2 in (let result=input1 in result)))'
    )
    merged_blocks, modified = transformations.merge_chained_blocks(block3)
    self.assertEqual(
        merged_blocks.tff_repr,
        '(let input2=input3,input1=input2,result=input1 in result)')
    self.assertTrue(modified)

  def test_replace_selection_from_tuple_fails_on_none_comp(self):
    with self.assertRaises(TypeError):
      transformations.replace_selection_from_tuple_with_tuple_element(None)

  def test_replace_selection_from_tuple_leaves_selection_from_ref_by_index_alone(
      self):
    ref_to_tuple = computation_building_blocks.Reference(
        'tup', [('a', tf.int32), ('b', tf.float32)])
    a_selected = computation_building_blocks.Selection(ref_to_tuple, index=0)
    b_selected = computation_building_blocks.Selection(ref_to_tuple, index=1)

    a_returned, a_transformed = transformations.replace_selection_from_tuple_with_tuple_element(
        a_selected)
    b_returned, b_transformed = transformations.replace_selection_from_tuple_with_tuple_element(
        b_selected)

    self.assertFalse(a_transformed)
    self.assertEqual(a_returned.proto, a_selected.proto)
    self.assertFalse(b_transformed)
    self.assertEqual(b_returned.proto, b_selected.proto)

  def test_replace_selection_from_tuple_leaves_selection_from_ref_by_name_alone(
      self):
    ref_to_tuple = computation_building_blocks.Reference(
        'tup', [('a', tf.int32), ('b', tf.float32)])
    a_selected = computation_building_blocks.Selection(ref_to_tuple, name='a')
    b_selected = computation_building_blocks.Selection(ref_to_tuple, name='b')

    a_returned, a_transformed = transformations.replace_selection_from_tuple_with_tuple_element(
        a_selected)
    b_returned, b_transformed = transformations.replace_selection_from_tuple_with_tuple_element(
        b_selected)

    self.assertFalse(a_transformed)
    self.assertEqual(a_returned.proto, a_selected.proto)
    self.assertFalse(b_transformed)
    self.assertEqual(b_returned.proto, b_selected.proto)

  def test_replace_selection_from_tuple_by_index_grabs_correct_element(self):
    x_data = computation_building_blocks.Data('x', tf.int32)
    y_data = computation_building_blocks.Data('y', [('a', tf.float32)])
    tup = computation_building_blocks.Tuple([x_data, y_data])
    x_selected = computation_building_blocks.Selection(tup, index=0)
    y_selected = computation_building_blocks.Selection(tup, index=1)

    collapsed_selection_x, x_transformed = transformations.replace_selection_from_tuple_with_tuple_element(
        x_selected)
    collapsed_selection_y, y_transformed = transformations.replace_selection_from_tuple_with_tuple_element(
        y_selected)

    self.assertTrue(x_transformed)
    self.assertTrue(y_transformed)
    self.assertEqual(collapsed_selection_x.proto, x_data.proto)
    self.assertEqual(collapsed_selection_y.proto, y_data.proto)

  def test_replace_selection_from_tuple_by_name_grabs_correct_element(self):
    x_data = computation_building_blocks.Data('x', tf.int32)
    y_data = computation_building_blocks.Data('y', [('a', tf.float32)])
    tup = computation_building_blocks.Tuple([('a', x_data), ('b', y_data)])
    x_selected = computation_building_blocks.Selection(tup, name='a')
    y_selected = computation_building_blocks.Selection(tup, name='b')

    collapsed_selection_x, x_transformed = transformations.replace_selection_from_tuple_with_tuple_element(
        x_selected)
    collapsed_selection_y, y_transformed = transformations.replace_selection_from_tuple_with_tuple_element(
        y_selected)

    self.assertTrue(x_transformed)
    self.assertTrue(y_transformed)
    self.assertEqual(collapsed_selection_x.proto, x_data.proto)
    self.assertEqual(collapsed_selection_y.proto, y_data.proto)


class UniquifyReferencesTest(absltest.TestCase):

  def test_uniquify_references_single_level_block(self):
    x_ref = computation_building_blocks.Reference('x', tf.int32)
    data = computation_building_blocks.Data('data', tf.int32)
    block = computation_building_blocks.Block([('x', data), ('x', x_ref),
                                               ('x', x_ref)], x_ref)
    self.assertEqual(block.tff_repr, '(let x=data,x=x,x=x in x)')
    renamed = transformations.uniquify_references(block)
    self.assertEqual(
        renamed.tff_repr,
        '(let {0}1=data,{0}2={0}1,{0}3={0}2 in {0}3)'.format(RENAME_PREFIX))

  def test_uniquify_references_nested_blocks(self):
    x_ref = computation_building_blocks.Reference('x', tf.int32)
    input1 = computation_building_blocks.Data('input1', tf.int32)
    block1 = computation_building_blocks.Block([('x', input1), ('x', x_ref)],
                                               x_ref)
    input2 = computation_building_blocks.Data('input2', tf.int32)
    block2 = computation_building_blocks.Block([('x', input2), ('x', x_ref)],
                                               block1)
    self.assertEqual(
        str(block2), '(let x=input2,x=x in (let x=input1,x=x in x))')
    renamed = transformations.uniquify_references(block2)
    self.assertTrue(_has_unique_names(renamed))
    self.assertEqual(
        renamed.tff_repr,
        '(let {0}1=input2,{0}2={0}1 in (let {0}3=input1,{0}4={0}3 in {0}4))'
        .format(RENAME_PREFIX))

  def test_uniquify_references_nested_lambdas(self):
    comp = computation_building_blocks.Data('test', tf.int32)
    input1 = computation_building_blocks.Reference('input1',
                                                   comp.type_signature)
    first_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input1', input1.type_signature,
                                           input1), comp)
    input2 = computation_building_blocks.Reference(
        'input2', first_level_call.type_signature)
    second_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input2', input2.type_signature,
                                           input2), first_level_call)
    renamed = transformations.uniquify_references(second_level_call)
    self.assertTrue(_has_unique_names(renamed))
    self.assertEqual(
        renamed.tff_repr,
        '({0}1 -> {0}1)(({0}2 -> {0}2)(test))'.format(RENAME_PREFIX))

  def test_uniquify_references_block_lambda_block_lambda(self):
    x_ref = computation_building_blocks.Reference('x', tf.int32)
    inner_lambda = computation_building_blocks.Lambda('x', tf.int32, x_ref)
    called_lambda = computation_building_blocks.Call(inner_lambda, x_ref)
    lower_block = computation_building_blocks.Block([('x', x_ref),
                                                     ('x', x_ref)],
                                                    called_lambda)
    second_lambda = computation_building_blocks.Lambda('x', tf.int32,
                                                       lower_block)
    second_call = computation_building_blocks.Call(second_lambda, x_ref)
    final_input = computation_building_blocks.Data('test_data', tf.int32)
    last_block = computation_building_blocks.Block([('x', final_input),
                                                    ('x', x_ref)], second_call)
    renamed = transformations.uniquify_references(last_block)
    self.assertEqual(
        last_block.tff_repr,
        '(let x=test_data,x=x in (x -> (let x=x,x=x in (x -> x)(x)))(x))')
    self.assertTrue(_has_unique_names(renamed))
    self.assertEqual(
        renamed.tff_repr,
        '(let {0}1=test_data,{0}2={0}1 in ({0}3 -> (let {0}4={0}3,{0}5={0}4 in ({0}6 -> {0}6)({0}5)))({0}2))'
        .format(RENAME_PREFIX))

  def test_uniquify_references_blocks_nested_inside_of_locals(self):
    x_data = computation_building_blocks.Data('x', tf.int32)
    data = computation_building_blocks.Data('data', tf.int32)
    lower_block = computation_building_blocks.Block([('y', data)], x_data)
    middle_block = computation_building_blocks.Block([('y', lower_block)],
                                                     x_data)
    higher_block = computation_building_blocks.Block([('y', middle_block)],
                                                     x_data)

    y_ref = computation_building_blocks.Reference('y', tf.int32)
    lower_block_with_y_ref = computation_building_blocks.Block([('y', y_ref)],
                                                               x_data)
    middle_block_with_y_ref = computation_building_blocks.Block(
        [('y', lower_block_with_y_ref)], x_data)
    higher_block_with_y_ref = computation_building_blocks.Block(
        [('y', middle_block_with_y_ref)], x_data)

    multiple_bindings_highest_block = computation_building_blocks.Block(
        [('y', higher_block),
         ('y', higher_block_with_y_ref)], higher_block_with_y_ref)
    renamed = transformations.uniquify_references(
        multiple_bindings_highest_block)
    self.assertEqual(higher_block.tff_repr,
                     '(let y=(let y=(let y=data in x) in x) in x)')
    self.assertEqual(higher_block_with_y_ref.tff_repr,
                     '(let y=(let y=(let y=y in x) in x) in x)')
    self.assertEqual(renamed.locals[0][0], '{}4'.format(RENAME_PREFIX))
    self.assertEqual(
        renamed.locals[0][1].tff_repr,
        '(let {0}3=(let {0}2=(let {0}1=data in x) in x) in x)'.format(
            RENAME_PREFIX))
    self.assertEqual(renamed.locals[1][0], '{}8'.format(RENAME_PREFIX))
    self.assertEqual(
        renamed.locals[1][1].tff_repr,
        '(let {0}7=(let {0}6=(let {0}5={0}4 in x) in x) in x)'.format(
            RENAME_PREFIX))
    self.assertEqual(
        renamed.result.tff_repr,
        '(let {0}11=(let {0}10=(let {0}9={0}8 in x) in x) in x)'.format(
            RENAME_PREFIX))
    self.assertTrue(_has_unique_names(renamed))


if __name__ == '__main__':
  absltest.main()
