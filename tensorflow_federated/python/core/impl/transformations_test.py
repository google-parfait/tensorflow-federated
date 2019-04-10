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
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import transformation_utils
from tensorflow_federated.python.core.impl import transformations


def _to_building_block(comp):
  """Converts a computation into a computation building block.

  Args:
    comp: An instance of `computation_impl.ComputationImpl`.

  Returns:
    A instance of `computation_building_blocks.ComputationBuildingBlock`
    representing the `computation_impl.ComputationImpl`.
  """
  py_typecheck.check_type(comp, computation_impl.ComputationImpl)
  proto = computation_impl.ComputationImpl.get_proto(comp)
  return computation_building_blocks.ComputationBuildingBlock.from_proto(proto)


def _to_comp(fn):
  """Converts a computation building block into a computation.

  Args:
    fn: An instance of `computation_building_blocks.ComputationBuildingBlock`.

  Returns:
    An instance of `computation_impl.ComputationImpl` representing the
    `computation_building_blocks.ComputationBuildingBlock`.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  return computation_impl.ComputationImpl(fn.proto,
                                          context_stack_impl.context_stack)


def _create_call_to_py_fn(fn):
  r"""Creates a computation to call a Python function.

           Call
          /
  Compiled
  Computation

  Args:
    fn: The Python function to wrap.

  Returns:
    An instance of `computation_building_blocks.Call` wrapping the Python
    function.
  """
  tf_comp = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      fn, None, context_stack_impl.context_stack)
  compiled_comp = computation_building_blocks.CompiledComputation(tf_comp)
  return computation_building_blocks.Call(compiled_comp)


def _create_call_to_federated_map(fn, arg):
  r"""Creates a computation to call a federated map.

            Call
           /    \
  Intrinsic      Tuple
                /     \
     Computation       Computation

  Args:
    fn: An instance of a functional
      `computation_building_blocks.ComputationBuildingBlock` to use as the map
      function.
    arg: An instance of `computation_building_blocks.ComputationBuildingBlock`
      to use as the map argument.

  Returns:
    An instance of `computation_building_blocks.Call` wrapping the federated map
    computation.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  federated_type = computation_types.FederatedType(fn.type_signature.result,
                                                   placements.CLIENTS)
  function_type = computation_types.FunctionType(
      [fn.type_signature, arg.type_signature], federated_type)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP.uri, function_type)
  tup = computation_building_blocks.Tuple((fn, arg))
  return computation_building_blocks.Call(intrinsic, tup)


def _create_lambda_to_chained_call(fn, arg, n):
  r"""Creates a lambda to chain `n` calls.

       Lambda 1
             \
              Call 2
             /    \
  Computation 3    ... 4
                  /   \
               ... 5   Call 6
                      /    \
           Computation 7    Computation 8

  The `n` calls are represented by (2, 4, and 6), for each call the function is
  the functional computation `fn` (3, 5, and 7) and the argument is either the
  result of the previous call or, if the call is the first call, the argument
  `arg` (8).

  Args:
    fn: An instance of a functional
      `computation_building_blocks.ComputationBuildingBlock` to use as the call
      function.
    arg: An instance of `computation_building_blocks.Reference` to use as the
      call argument.
    n: The number of calls to chain.

  Returns:
    An instance of `computation_building_blocks.Lambda` pointing to `n` chained
    calls.
  """
  py_typecheck.check_type(fn,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, computation_building_blocks.Reference)
  chained_arg = arg
  for _ in range(n):
    chained_call = computation_building_blocks.Call(fn, chained_arg)
    chained_arg = chained_call
  return computation_building_blocks.Lambda(arg.name, arg.type_signature,
                                            chained_call)


def _create_lambda_to_add_one(dtype):
  r"""Creates a computation to add `1` to an argument.

  Lambda
        \
         Call
        /    \
  Intrinsic   Tuple
             /     \
    Reference       Computation

  Args:
    dtype: The type of the argument.

  Returns:
    An instance of `computation_building_blocks.Lambda` wrapping a function that
    adds 1 to an argument.
  """
  function_type = computation_types.FunctionType([dtype, dtype], dtype)
  intrinsic = computation_building_blocks.Intrinsic(
      intrinsic_defs.GENERIC_PLUS.uri, function_type)
  arg = computation_building_blocks.Reference('arg', dtype)
  constant = _create_call_to_py_fn(lambda: tf.constant(1))
  tup = computation_building_blocks.Tuple([arg, constant])
  call = computation_building_blocks.Call(intrinsic, tup)
  return computation_building_blocks.Lambda(arg.name, arg.type_signature, call)


def _create_lambda_to_identity(dtype):
  r"""Creates a lambda to return the argument.

  Lambda(x)
        \
         Reference(x)

  Args:
    dtype: The type of the argument.

  Returns:
    An instance of `computation_building_blocks.Lambda`.
  """
  arg = computation_building_blocks.Reference('arg', dtype)
  return computation_building_blocks.Lambda(arg.name, arg.type_signature, arg)


def _get_number_of_nodes(comp, predicate=None):
  """Returns the number of nodes in `comp` matching `predicate`."""
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  count = [0]  # TODO(b/129791812): Cleanup Python 2 and 3 compatibility.

  def fn(comp):
    if predicate is None or predicate(comp):
      count[0] += 1
    return comp

  transformation_utils.transform_postorder(comp, fn)
  return count[0]


def _get_number_of_computations(comp, comp_type=None):
  """Returns the number of computations in `comp` with the type `comp_type`."""
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def fn(node):
    # Using explicit type comparisons here to prevent a subclass from passing.
    return type(node) is comp_type  # pylint: disable=unidiomatic-typecheck

  return _get_number_of_nodes(comp, fn)


def _get_number_of_intrinsics(comp, uri=None):
  """Returns the number of intrinsics in `comp` with the uri `comp_type`."""
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def fn(node):
    return (isinstance(node, computation_building_blocks.Intrinsic) and
            node.uri == uri)

  return _get_number_of_nodes(comp, fn)


class TransformationsTest(parameterized.TestCase):

  def test_replace_compiled_computations_names_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.replace_compiled_computations_names_with_unique_names(
          None)

  def test_replace_compiled_computations_names_replaces_name(self):
    fn = lambda: tf.constant(1)
    tf_comp = tensorflow_serialization.serialize_py_fn_as_tf_computation(
        fn, None, context_stack_impl.context_stack)
    compiled_comp = computation_building_blocks.CompiledComputation(tf_comp)
    comp = compiled_comp

    transformed_comp = transformations.replace_compiled_computations_names_with_unique_names(
        comp)

    self.assertNotEqual(transformed_comp._name, comp._name)

  def test_replace_compiled_computations_names_replaces_multiple_names(self):
    comps = []
    for _ in range(10):
      fn = lambda: tf.constant(1)
      tf_comp = tensorflow_serialization.serialize_py_fn_as_tf_computation(
          fn, None, context_stack_impl.context_stack)
      compiled_comp = computation_building_blocks.CompiledComputation(tf_comp)
      comps.append(compiled_comp)
    tup = computation_building_blocks.Tuple(comps)
    comp = tup

    transformed_comp = transformations.replace_compiled_computations_names_with_unique_names(
        comp)

    comp_names = [element._name for element in comp]
    transformed_comp_names = [element._name for element in transformed_comp]
    self.assertNotEqual(transformed_comp_names, comp_names)
    self.assertEqual(
        len(transformed_comp_names), len(set(transformed_comp_names)),
        'The transformed computation names are not unique: {}.'.format(
            transformed_comp_names))

  def test_replace_compiled_computations_names_does_not_replace_other_name(
      self):
    comp = computation_building_blocks.Reference('name', tf.int32)

    transformed_comp = transformations.replace_compiled_computations_names_with_unique_names(
        comp)

    self.assertEqual(transformed_comp._name, comp._name)

  def test_replace_intrinsic_raises_type_error_none_comp(self):
    uri = intrinsic_defs.GENERIC_PLUS.uri
    body = lambda x: 100

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(
          None, uri, body, context_stack_impl.context_stack)

  def test_replace_intrinsic_raises_type_error_none_uri(self):
    comp = _create_lambda_to_add_one(tf.int32)
    body = lambda x: 100

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(
          comp, None, body, context_stack_impl.context_stack)

  def test_replace_intrinsic_raises_type_error_none_body(self):
    comp = _create_lambda_to_add_one(tf.int32)
    uri = intrinsic_defs.GENERIC_PLUS.uri

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(
          comp, uri, None, context_stack_impl.context_stack)

  def test_replace_intrinsic_raises_type_error_none_context_stack(self):
    comp = _create_lambda_to_add_one(tf.int32)
    uri = intrinsic_defs.GENERIC_PLUS.uri
    body = lambda x: 100

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(comp, uri, body, None)

  def test_replace_intrinsic_replaces_intrinsic(self):
    comp = _create_lambda_to_add_one(tf.int32)
    uri = intrinsic_defs.GENERIC_PLUS.uri
    body = lambda x: 100

    self.assertEqual(_get_number_of_intrinsics(comp, uri), 1)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl(1), 2)

    transformed_comp = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(_get_number_of_intrinsics(transformed_comp, uri), 0)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl(1), 100)

  def test_replace_intrinsic_replaces_multiple_intrinsics(self):
    calling_arg = computation_building_blocks.Reference('arg', tf.int32)
    arg_type = calling_arg.type_signature
    arg = calling_arg
    for _ in range(10):
      lam = _create_lambda_to_add_one(arg_type)
      call = computation_building_blocks.Call(lam, arg)
      arg_type = call.function.type_signature.result
      arg = call
    calling_lambda = computation_building_blocks.Lambda(
        calling_arg.name, calling_arg.type_signature, call)
    comp = calling_lambda
    uri = intrinsic_defs.GENERIC_PLUS.uri
    body = lambda x: 100

    self.assertEqual(_get_number_of_intrinsics(comp, uri), 10)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl(1), 11)

    transformed_comp = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(_get_number_of_intrinsics(transformed_comp, uri), 0)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl(1), 100)

  def test_replace_intrinsic_does_not_replace_other_intrinsic(self):
    comp = _create_lambda_to_add_one(tf.int32)
    uri = intrinsic_defs.GENERIC_PLUS.uri
    different_uri = intrinsic_defs.FEDERATED_SUM.uri
    body = lambda x: 100

    self.assertEqual(_get_number_of_intrinsics(comp, uri), 1)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl(1), 2)

    transformed_comp = transformations.replace_intrinsic_with_callable(
        comp, different_uri, body, context_stack_impl.context_stack)

    self.assertEqual(_get_number_of_intrinsics(transformed_comp, uri), 1)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl(1), 2)

  def test_replace_called_lambda_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.replace_called_lambda_with_block(None)

  def test_replace_called_lambda_replaces_called_lambda(self):
    arg = computation_building_blocks.Reference('arg', tf.int32)
    lam = _create_lambda_to_add_one(arg.type_signature)
    call = computation_building_blocks.Call(lam, arg)
    calling_lambda = computation_building_blocks.Lambda(arg.name,
                                                        arg.type_signature,
                                                        call)
    comp = calling_lambda

    self.assertEqual(
        _get_number_of_computations(comp, computation_building_blocks.Block), 0)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl(1), 2)

    transformed_comp = transformations.replace_called_lambda_with_block(comp)

    self.assertEqual(
        _get_number_of_computations(transformed_comp,
                                    computation_building_blocks.Call),
        _get_number_of_computations(comp, computation_building_blocks.Call) - 1)
    self.assertEqual(
        _get_number_of_computations(transformed_comp,
                                    computation_building_blocks.Lambda),
        _get_number_of_computations(comp, computation_building_blocks.Lambda) -
        1)
    self.assertEqual(
        _get_number_of_computations(transformed_comp,
                                    computation_building_blocks.Block), 1)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl(1), 2)

  def test_replace_called_lambda_replaces_multiple_called_lambdas(self):
    arg = computation_building_blocks.Reference('arg', tf.int32)
    lam = _create_lambda_to_add_one(arg.type_signature)
    calling_lambda = _create_lambda_to_chained_call(lam, arg, 10)
    comp = calling_lambda

    self.assertEqual(
        _get_number_of_computations(comp, computation_building_blocks.Block), 0)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl(1), 11)

    transformed_comp = transformations.replace_called_lambda_with_block(comp)

    self.assertEqual(
        _get_number_of_computations(transformed_comp,
                                    computation_building_blocks.Call),
        _get_number_of_computations(comp, computation_building_blocks.Call) -
        10)
    self.assertEqual(
        _get_number_of_computations(transformed_comp,
                                    computation_building_blocks.Lambda),
        _get_number_of_computations(comp, computation_building_blocks.Lambda) -
        10)
    self.assertEqual(
        _get_number_of_computations(transformed_comp,
                                    computation_building_blocks.Block), 10)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl(1), 11)

  def test_replace_called_lambda_does_not_replace_uncalled_lambda(self):
    comp = _create_lambda_to_add_one(tf.int32)

    self.assertEqual(
        _get_number_of_computations(comp, computation_building_blocks.Block), 0)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl(1), 2)

    transformed_comp = transformations.replace_called_lambda_with_block(comp)

    self.assertEqual(
        _get_number_of_computations(transformed_comp,
                                    computation_building_blocks.Call),
        _get_number_of_computations(comp, computation_building_blocks.Call))
    self.assertEqual(
        _get_number_of_computations(transformed_comp,
                                    computation_building_blocks.Lambda),
        _get_number_of_computations(comp, computation_building_blocks.Lambda))
    self.assertEqual(
        _get_number_of_computations(transformed_comp,
                                    computation_building_blocks.Block), 0)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl(1), 2)

  def test_replace_called_lambda_does_not_replace_separated_called_lambda(self):
    arg = computation_building_blocks.Reference('arg', tf.int32)
    lam = _create_lambda_to_identity(arg.type_signature)
    block = computation_building_blocks.Block([], lam)
    call = computation_building_blocks.Call(block, arg)
    comp = call

    transformed_comp = transformations.replace_called_lambda_with_block(comp)

    self.assertEqual(str(transformed_comp), str(comp))
    self.assertEqual(str(transformed_comp), '(let  in (arg -> arg))(arg)')

  def test_remove_mapped_or_applied_identity_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.remove_mapped_or_applied_identity(None)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_map',
       intrinsic_defs.FEDERATED_MAP.uri,
       computation_types.FederatedType(tf.float32, placements.CLIENTS)),
      ('federated_apply',
       intrinsic_defs.FEDERATED_APPLY.uri,
       computation_types.FederatedType(tf.float32, placements.SERVER)),
      ('sequence_map',
       intrinsic_defs.SEQUENCE_MAP.uri,
       computation_types.SequenceType(tf.float32)))
  # pyformat: enable
  def test_remove_mapped_or_applied_identity_removes_identity(
      self, uri, data_type):
    data = computation_building_blocks.Data('x', data_type)
    identity_arg = computation_building_blocks.Reference('arg', tf.float32)
    identity_lam = computation_building_blocks.Lambda('arg', tf.float32,
                                                      identity_arg)
    arg_tuple = computation_building_blocks.Tuple([identity_lam, data])
    function_type = computation_types.FunctionType(
        [arg_tuple.type_signature[0], arg_tuple.type_signature[1]],
        arg_tuple.type_signature[1])
    intrinsic = computation_building_blocks.Intrinsic(uri, function_type)
    call = computation_building_blocks.Call(intrinsic, arg_tuple)
    self.assertEqual(str(call), '{}(<(arg -> arg),x>)'.format(uri))
    reduced = transformations.remove_mapped_or_applied_identity(call)
    self.assertEqual(str(reduced), 'x')

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_map',
       intrinsic_defs.FEDERATED_MAP.uri,
       computation_types.FederatedType(tf.float32, placements.CLIENTS)),
      ('federated_apply',
       intrinsic_defs.FEDERATED_APPLY.uri,
       computation_types.FederatedType(tf.float32, placements.SERVER)),
      ('sequence_map',
       intrinsic_defs.SEQUENCE_MAP.uri,
       computation_types.SequenceType(tf.float32)))
  # pyformat: enable
  def test_remove_mapped_or_applied_identity_removes_nested_identity(
      self, uri, data_type):
    data = computation_building_blocks.Data('x', data_type)
    identity_arg = computation_building_blocks.Reference('arg', tf.float32)
    identity_lam = computation_building_blocks.Lambda('arg', tf.float32,
                                                      identity_arg)
    arg_tuple = computation_building_blocks.Tuple([identity_lam, data])
    function_type = computation_types.FunctionType(
        [arg_tuple.type_signature[0], arg_tuple.type_signature[1]],
        arg_tuple.type_signature[1])
    intrinsic = computation_building_blocks.Intrinsic(uri, function_type)
    call = computation_building_blocks.Call(intrinsic, arg_tuple)
    tuple_wrapped_call = computation_building_blocks.Tuple([call])
    lambda_wrapped_tuple = computation_building_blocks.Lambda(
        'y', tf.int32, tuple_wrapped_call)
    self.assertEqual(
        str(lambda_wrapped_tuple), '(y -> <{}(<(arg -> arg),x>)>)'.format(uri))
    reduced = transformations.remove_mapped_or_applied_identity(
        lambda_wrapped_tuple)
    self.assertEqual(str(reduced), '(y -> <x>)')

  def test_remove_mapped_or_applied_identity_removes_multiple_identities(self):
    calling_arg_type = computation_types.FederatedType(tf.int32,
                                                       placements.CLIENTS)
    calling_arg = computation_building_blocks.Data('x', calling_arg_type)
    arg_type = calling_arg.type_signature.member
    arg = calling_arg
    for _ in range(2):
      lam = _create_lambda_to_identity(arg_type)
      call = _create_call_to_federated_map(lam, arg)
      arg_type = call.function.type_signature.result.member
      arg = call
    comp = call
    uri = intrinsic_defs.FEDERATED_MAP.uri

    transformed_comp = transformations.remove_mapped_or_applied_identity(comp)

    self.assertEqual(
        str(comp),
        '{uri}(<(arg -> arg),{uri}(<(arg -> arg),x>)>)'.format(uri=uri))
    self.assertEqual(str(transformed_comp), 'x')

  def test_remove_mapped_or_applied_identity_does_not_remove_other_intrinsic(
      self):
    data_type = tf.int32
    uri = 'dummy'
    data = computation_building_blocks.Data('x', data_type)
    identity_arg = computation_building_blocks.Reference('arg', tf.float32)
    identity_lam = computation_building_blocks.Lambda('arg', tf.float32,
                                                      identity_arg)
    arg_tuple = computation_building_blocks.Tuple([identity_lam, data])
    function_type = computation_types.FunctionType(
        [arg_tuple.type_signature[0], arg_tuple.type_signature[1]],
        arg_tuple.type_signature[1])
    intrinsic = computation_building_blocks.Intrinsic(uri, function_type)
    call = computation_building_blocks.Call(intrinsic, arg_tuple)
    comp = call

    transformed_comp = transformations.remove_mapped_or_applied_identity(comp)

    self.assertEqual(str(comp), '{}(<(arg -> arg),x>)'.format(uri))
    self.assertEqual(str(transformed_comp), '{}(<(arg -> arg),x>)'.format(uri))

  def test_remove_mapped_or_applied_identity_does_not_remove_called_lambda(
      self):
    x = computation_building_blocks.Reference('x', tf.int32)
    dummy_lambda = computation_building_blocks.Lambda('x', tf.int32, x)
    test_arg = computation_building_blocks.Data('test', tf.int32)
    called = computation_building_blocks.Call(dummy_lambda, test_arg)
    self.assertEqual(str(called), '(x -> x)(test)')
    self.assertEqual(
        str(transformations.remove_mapped_or_applied_identity(called)),
        '(x -> x)(test)')

  def test_replace_chained_federated_maps_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.replace_chained_federated_maps_with_federated_map(None)

  def test_replace_chained_federated_maps_replaces_federated_maps(self):
    map_arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    map_arg = computation_building_blocks.Reference('arg', map_arg_type)
    inner_lambda = _create_lambda_to_add_one(map_arg.type_signature.member)
    inner_call = _create_call_to_federated_map(inner_lambda, map_arg)
    outer_lambda = _create_lambda_to_add_one(
        inner_call.function.type_signature.result.member)
    outer_call = _create_call_to_federated_map(outer_lambda, inner_call)
    map_lambda = computation_building_blocks.Lambda(map_arg.name,
                                                    map_arg.type_signature,
                                                    outer_call)
    comp = map_lambda
    uri = intrinsic_defs.FEDERATED_MAP.uri

    self.assertEqual(_get_number_of_intrinsics(comp, uri), 2)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl([(1)]), [3])

    transformed_comp = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(_get_number_of_intrinsics(transformed_comp, uri), 1)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl([(1)]), [3])

  def test_replace_chained_federated_maps_replaces_federated_maps_different_types(
      self):
    # TODO(b/130043404): Write a minimal test to check that merging federated
    # maps places correct type signatures on the constructed intrinsic.
    pass

  def test_replace_chained_federated_maps_replaces_multiple_federated_maps(
      self):
    calling_arg_type = computation_types.FederatedType(tf.int32,
                                                       placements.CLIENTS)
    calling_arg = computation_building_blocks.Reference('arg', calling_arg_type)
    arg_type = calling_arg.type_signature.member
    arg = calling_arg
    for _ in range(10):
      lam = _create_lambda_to_add_one(arg_type)
      call = _create_call_to_federated_map(lam, arg)
      arg_type = call.function.type_signature.result.member
      arg = call
    calling_lambda = computation_building_blocks.Lambda(
        calling_arg.name, calling_arg.type_signature, call)
    comp = calling_lambda
    uri = intrinsic_defs.FEDERATED_MAP.uri

    self.assertEqual(_get_number_of_intrinsics(comp, uri), 10)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl([(1)]), [11])

    transformed_comp = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(_get_number_of_intrinsics(transformed_comp, uri), 1)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl([(1)]), [11])

  def test_replace_chained_federated_maps_does_not_replace_one_federated_maps(
      self):
    map_arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    map_arg = computation_building_blocks.Reference('arg', map_arg_type)
    inner_lambda = _create_lambda_to_add_one(map_arg.type_signature.member)
    inner_call = _create_call_to_federated_map(inner_lambda, map_arg)
    map_lambda = computation_building_blocks.Lambda(map_arg.name,
                                                    map_arg.type_signature,
                                                    inner_call)
    comp = map_lambda
    uri = intrinsic_defs.FEDERATED_MAP.uri

    self.assertEqual(_get_number_of_intrinsics(comp, uri), 1)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl([(1)]), [2])

    transformed_comp = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(_get_number_of_intrinsics(transformed_comp, uri), 1)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl([(1)]), [2])

  def test_replace_chained_federated_maps_does_not_replace_unchained_federated_maps(
      self):
    map_arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    map_arg = computation_building_blocks.Reference('arg', map_arg_type)
    inner_lambda = _create_lambda_to_add_one(map_arg.type_signature.member)
    inner_call = _create_call_to_federated_map(inner_lambda, map_arg)
    dummy_tuple = computation_building_blocks.Tuple([inner_call])
    dummy_selection = computation_building_blocks.Selection(
        dummy_tuple, index=0)
    outer_lambda = _create_lambda_to_add_one(
        inner_call.function.type_signature.result.member)
    outer_call = _create_call_to_federated_map(outer_lambda, dummy_selection)
    map_lambda = computation_building_blocks.Lambda(map_arg.name,
                                                    map_arg.type_signature,
                                                    outer_call)
    comp = map_lambda
    uri = intrinsic_defs.FEDERATED_MAP.uri

    self.assertEqual(_get_number_of_intrinsics(comp, uri), 2)
    comp_impl = _to_comp(comp)
    self.assertEqual(comp_impl([(1)]), [3])

    transformed_comp = transformations.replace_chained_federated_maps_with_federated_map(
        comp)

    self.assertEqual(_get_number_of_intrinsics(transformed_comp, uri), 2)
    transformed_comp_impl = _to_comp(transformed_comp)
    self.assertEqual(transformed_comp_impl([(1)]), [3])

  def test_inline_conflicting_lambdas(self):
    comp = computation_building_blocks.Tuple(
        [computation_building_blocks.Data('test', tf.int32)])
    input1 = computation_building_blocks.Reference('input2',
                                                   comp.type_signature)
    first_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input2', input1.type_signature,
                                           input1), comp)
    input2 = computation_building_blocks.Reference(
        'input2', first_level_call.type_signature)
    second_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input2', input2.type_signature,
                                           input2), first_level_call)
    self.assertEqual(
        str(second_level_call),
        '(input2 -> input2)((input2 -> input2)(<test>))')
    lambda_reduced_comp = transformations.replace_called_lambda_with_block(
        second_level_call)
    self.assertEqual(
        str(lambda_reduced_comp),
        '(let input2=(let input2=<test> in input2) in input2)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        lambda_reduced_comp)
    self.assertEqual(str(inlined), '(let  in (let  in <test>))')

  def test_inline_conflicting_locals(self):
    arg_comp = computation_building_blocks.Reference('arg',
                                                     [tf.int32, tf.int32])
    selected = computation_building_blocks.Selection(arg_comp, index=0)
    internal_arg = computation_building_blocks.Reference('arg', tf.int32)
    block = computation_building_blocks.Block([('arg', selected)], internal_arg)
    lam = computation_building_blocks.Lambda('arg', arg_comp.type_signature,
                                             block)
    self.assertEqual(str(lam), '(arg -> (let arg=arg[0] in arg))')
    inlined = transformations.inline_blocks_with_n_referenced_locals(lam)
    self.assertEqual(str(inlined), '(arg -> (let  in arg[0]))')

  def test_simple_block_inlining(self):
    test_arg = computation_building_blocks.Data('test_data', tf.int32)
    result = computation_building_blocks.Reference('test_x',
                                                   test_arg.type_signature)
    simple_block = computation_building_blocks.Block([('test_x', test_arg)],
                                                     result)
    self.assertEqual(str(simple_block), '(let test_x=test_data in test_x)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        simple_block)
    self.assertEqual(str(inlined), '(let  in test_data)')

  def test_no_inlining_if_referenced_twice(self):
    test_arg = computation_building_blocks.Data('test_data', tf.int32)
    ref1 = computation_building_blocks.Reference('test_x',
                                                 test_arg.type_signature)
    ref2 = computation_building_blocks.Reference('test_x',
                                                 test_arg.type_signature)
    result = computation_building_blocks.Tuple([ref1, ref2])
    simple_block = computation_building_blocks.Block([('test_x', test_arg)],
                                                     result)
    self.assertEqual(
        str(simple_block), '(let test_x=test_data in <test_x,test_x>)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        simple_block)
    self.assertEqual(str(inlined), str(simple_block))

  def test_inlining_n_2(self):
    test_arg = computation_building_blocks.Data('test_data', tf.int32)
    ref1 = computation_building_blocks.Reference('test_x',
                                                 test_arg.type_signature)
    ref2 = computation_building_blocks.Reference('test_x',
                                                 test_arg.type_signature)
    result = computation_building_blocks.Tuple([ref1, ref2])
    simple_block = computation_building_blocks.Block([('test_x', test_arg)],
                                                     result)
    self.assertEqual(
        str(simple_block), '(let test_x=test_data in <test_x,test_x>)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        simple_block, 2)
    self.assertEqual(str(inlined), '(let  in <test_data,test_data>)')

  def test_conflicting_name_resolved_inlining(self):
    red_herring_arg = computation_building_blocks.Reference(
        'redherring', tf.int32)
    used_arg = computation_building_blocks.Reference('used', tf.int32)
    ref = computation_building_blocks.Reference('x', used_arg.type_signature)
    lower_block = computation_building_blocks.Block([('x', used_arg)], ref)
    higher_block = computation_building_blocks.Block([('x', red_herring_arg)],
                                                     lower_block)
    self.assertEqual(
        str(higher_block), '(let x=redherring in (let x=used in x))')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        higher_block)
    self.assertEqual(str(inlined), '(let  in (let  in used))')

  def test_multiple_inline_for_nested_block(self):
    used1 = computation_building_blocks.Reference('used1', tf.int32)
    used2 = computation_building_blocks.Reference('used2', tf.int32)
    ref = computation_building_blocks.Reference('x', used1.type_signature)
    lower_block = computation_building_blocks.Block([('x', used1)], ref)
    higher_block = computation_building_blocks.Block([('used1', used2)],
                                                     lower_block)
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        higher_block)
    self.assertEqual(
        str(higher_block), '(let used1=used2 in (let x=used1 in x))')
    self.assertEqual(str(inlined), '(let  in (let  in used2))')
    user_inlined_lower_block = computation_building_blocks.Block([('x', used1)],
                                                                 used1)
    user_inlined_higher_block = computation_building_blocks.Block(
        [('used1', used2)], user_inlined_lower_block)
    self.assertEqual(
        str(user_inlined_higher_block),
        '(let used1=used2 in (let x=used1 in used1))')
    inlined_noop = transformations.inline_blocks_with_n_referenced_locals(
        user_inlined_higher_block)
    self.assertEqual(str(inlined_noop), '(let used1=used2 in (let  in used1))')

  def test_conflicting_nested_name_inlining(self):
    innermost = computation_building_blocks.Reference('x', tf.int32)
    intermediate_arg = computation_building_blocks.Reference('y', tf.int32)
    item2 = computation_building_blocks.Block([('x', intermediate_arg)],
                                              innermost)
    item1 = computation_building_blocks.Reference('x', tf.int32)
    mediate_tuple = computation_building_blocks.Tuple([item1, item2])
    used = computation_building_blocks.Reference('used', tf.int32)
    used1 = computation_building_blocks.Reference('used1', tf.int32)
    outer_block = computation_building_blocks.Block([('x', used), ('y', used1)],
                                                    mediate_tuple)
    self.assertEqual(
        str(outer_block), '(let x=used,y=used1 in <x,(let x=y in x)>)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        outer_block)
    self.assertEqual(str(inlined), '(let  in <used,(let  in used1)>)')


if __name__ == '__main__':
  absltest.main()
