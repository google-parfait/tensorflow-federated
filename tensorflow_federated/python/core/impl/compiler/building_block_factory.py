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
"""Library implementing reusable `computation_building_blocks` structures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import string

import six
from six.moves import range
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def unique_name_generator(comp, prefix='_var'):
  """Yields a new unique name that does not exist in `comp`.

  Args:
    comp: The compuation building block to use as a reference.
    prefix: The prefix to use when generating unique names. If `prefix` is
      `None` or if `comp` contains any name with this prefix, then a unique
      prefix will be generated from random lowercase ascii characters.
  """
  if comp is not None:
    names = transformation_utils.get_unique_names(comp)
  else:
    names = set()
  while prefix is None or any(n.startswith(prefix) for n in names):
    characters = string.ascii_lowercase
    prefix = '_{}'.format(''.join(random.choice(characters) for _ in range(3)))
  index = 1
  while True:
    yield '{}{}'.format(prefix, index)
    index += 1


def create_compiled_empty_tuple():
  """Returns called graph representing the empty tuple.

  Returns:
    An instance of `building_blocks.Call`, calling a noarg function
    which returns an empty tuple. This function is an instance of
    `building_blocks.CompiledComputation`.
  """
  with tf.Graph().as_default() as graph:
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        [], graph)

  function_type = computation_types.FunctionType(None, result_type)
  serialized_function_type = type_serialization.serialize_type(function_type)

  proto = pb.Computation(
      type=serialized_function_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=None,
          result=result_binding))

  return building_blocks.Call(building_blocks.CompiledComputation(proto), None)


def create_compiled_identity(type_signature, name=None):
  """Creates CompiledComputation representing identity function.

  Args:
    type_signature: Argument convertible to instance of `computation_types.Type`
      via `computation_types.to_type`.
    name: An optional string name to use as the name of the computation.

  Returns:
    An instance of `building_blocks.CompiledComputation`
    representing the identity function taking an argument of type
    `type_signature` and returning the same value.

  Raises:
    TypeError: If `type_signature` contains any types which cannot appear in
      TensorFlow bindings.
  """
  type_spec = computation_types.to_type(type_signature)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if not type_utils.is_tensorflow_compatible_type(type_spec):
    raise TypeError(
        'Can only construct a TF block with types which only contain tensor, '
        'sequence or tuple types; you have tried to construct a TF block with '
        'parameter of type {}'.format(type_spec))
  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_spec, graph)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        parameter_value, graph)
  function_type = computation_types.FunctionType(type_spec, result_type)
  serialized_function_type = type_serialization.serialize_type(function_type)
  proto = pb.Computation(
      type=serialized_function_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=parameter_binding,
          result=result_binding))
  return building_blocks.CompiledComputation(proto, name)


def create_tensorflow_constant(type_spec, scalar_value):
  """Creates called graph returning constant `scalar_value` of type `type_spec`.

  `scalar_value` must be a scalar, and cannot be a float if any of the tensor
  leaves of `type_spec` contain an integer data type. `type_spec` must contain
  only named tuples and tensor types, but these can be arbitrarily nested.

  Args:
    type_spec: Value convertible to `computation_types.Type` via
      `computation_types.to_type`, and whose resulting type tree can only
      contain named tuples and tensors.
    scalar_value: Scalar value to place in all the tensor leaves of `type_spec`.

  Returns:
    An instance of `building_blocks.Call`, whose argument is `None`
    and whose function is a noarg
    `building_blocks.CompiledComputation` which returns the
    specified `scalar_value` packed into a TFF structure of type `type_spec.

  Raises:
    TypeError: If the type assumptions above are violated.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if not type_utils.is_generic_op_compatible_type(type_spec):
    raise TypeError('Type spec {} cannot be constructed as a TensorFlow '
                    'constant in TFF; only nested tuples and tensors are '
                    'permitted.'.format(type_spec))
  inferred_scalar_value_type = type_utils.infer_type(scalar_value)
  if (not isinstance(inferred_scalar_value_type, computation_types.TensorType)
      or inferred_scalar_value_type.shape != tf.TensorShape(())):
    raise TypeError('Must pass a scalar value to '
                    '`create_tensorflow_constant`; encountered a value '
                    '{}'.format(scalar_value))
  tensor_dtypes_in_type_spec = []

  def _pack_dtypes(type_signature):
    """Appends dtype of `type_signature` to nonlocal variable."""
    if isinstance(type_signature, computation_types.TensorType):
      tensor_dtypes_in_type_spec.append(type_signature.dtype)
    return type_signature, False

  type_utils.transform_type_postorder(type_spec, _pack_dtypes)

  if (any(x.is_integer for x in tensor_dtypes_in_type_spec) and
      not inferred_scalar_value_type.dtype.is_integer):
    raise TypeError('Only integers can be used as scalar values if our desired '
                    'constant type spec contains any integer tensors; passed '
                    'scalar {} of dtype {} for type spec {}.'.format(
                        scalar_value, inferred_scalar_value_type.dtype,
                        type_spec))

  def _create_result_tensor(type_spec, scalar_value):
    """Packs `scalar_value` into `type_spec` recursively."""
    if isinstance(type_spec, computation_types.TensorType):
      type_spec.shape.assert_is_fully_defined()
      result = tf.constant(
          scalar_value, dtype=type_spec.dtype, shape=type_spec.shape)
    else:
      elements = []
      for _, type_element in anonymous_tuple.to_elements(type_spec):
        elements.append(_create_result_tensor(type_element, scalar_value))
      result = elements
    return result

  with tf.Graph().as_default() as graph:
    result = _create_result_tensor(type_spec, scalar_value)
  _, result_binding = tensorflow_utils.capture_result_from_graph(result, graph)

  function_type = computation_types.FunctionType(None, type_spec)
  serialized_function_type = type_serialization.serialize_type(function_type)

  proto = pb.Computation(
      type=serialized_function_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=None,
          result=result_binding))

  noarg_constant_fn = building_blocks.CompiledComputation(proto)
  return building_blocks.Call(noarg_constant_fn, None)


def create_compiled_input_replication(type_signature, n_replicas):
  """Creates a compiled computation which replicates its argument.

  Args:
    type_signature: Value convertible to `computation_types.Type` via
      `computation_types.to_type`. The type of the parameter of the constructed
      computation.
    n_replicas: Integer, the number of times the argument is intended to be
      replicated.

  Returns:
    An instance of `building_blocks.CompiledComputation` encoding
    a function taking a single argument fo type `type_signature` and returning
    `n_replicas` identical copies of this argument.

  Raises:
    TypeError: If `type_signature` contains any types which cannot appear in
      TensorFlow bindings, or if `n_replicas` is not an integer.
  """
  type_spec = computation_types.to_type(type_signature)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if not type_utils.is_tensorflow_compatible_type(type_spec):
    raise TypeError(
        'Can only construct a TF block with types which only contain tensor, '
        'sequence or tuple types; you have tried to construct a TF block with '
        'parameter of type {}'.format(type_spec))
  py_typecheck.check_type(n_replicas, int)
  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_spec, graph)
    result = [parameter_value] * n_replicas
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  function_type = computation_types.FunctionType(type_spec, result_type)
  serialized_function_type = type_serialization.serialize_type(function_type)

  proto = pb.Computation(
      type=serialized_function_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=parameter_binding,
          result=result_binding))

  return building_blocks.CompiledComputation(proto)


def create_tensorflow_to_broadcast_scalar(scalar_type, new_shape):
  """Creates TF function broadcasting scalar to shape `new_shape`.

  Args:
    scalar_type: Instance of `tf.DType`, the type of the scalar we are looking
      to broadcast.
    new_shape: Instance of `tf.TensorShape`, the shape we wish to broadcast to.
      Must be fully defined.

  Returns:
    Instance of `building_blocks.CompiledComputation` representing
    a function declaring a scalar parameter of dtype `scalar_type`, and
    returning a tensor of this same dtype and shape `new_shape`, with the same
    value in each entry as its scalar argument.

  Raises:
    TypeError: If the types of the arguments do not match the declared arg
    types.
    ValueError: If `new_shape` is not fully defined.
  """
  # TODO(b/136119348): There are enough of these little TF helper functions,
  # and they are suffificiently conceptually similar, to potentially warrant
  # factoring out into their own file. At the same time, a possible clearer
  # pattern than immediately dropping into the graph context manager would be
  # to declare parameter ad result bindings outside of the context manager
  # (after constructing the graph of course) and only dropping in for the body.
  # If these functions get moved, perhaps that would be a natural time to
  # revisit the pattern.
  py_typecheck.check_type(scalar_type, tf.DType)
  py_typecheck.check_type(new_shape, tf.TensorShape)
  new_shape.assert_is_fully_defined()
  tensor_spec = computation_types.TensorType(scalar_type, shape=())

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', tensor_spec, graph)
    result = tf.broadcast_to(parameter_value, new_shape)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  function_type = computation_types.FunctionType(
      computation_types.TensorType(dtype=scalar_type, shape=()), result_type)
  serialized_function_type = type_serialization.serialize_type(function_type)

  proto = pb.Computation(
      type=serialized_function_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=parameter_binding,
          result=result_binding))

  return building_blocks.CompiledComputation(proto)


def create_tensorflow_binary_operator(operand_type, operator):
  """Creates a TensorFlow computation for the binary `operator`.

  For `T` the `operand_type`, the type signature of the constructed operator
  will be `(<T,T> -> U)`, where `U` is the result of applying `operator` to
  a tuple of type `<T,T>`.

  Notice that we have quite serious restrictions on `operand_type` here; not
  only must it be compatible with stamping into a TensorFlow graph, but
  additionally cannot contain a `computation_types.SequenceType`, as checked by
  `type_utils.is_generic_op_compatible_type`.

  Notice also that if `operand_type` is a `computation_types.NamedTupleType`,
  `operator` will be applied pointwise. This places the burden on callers of
  this function to construct the correct values to pass into the returned
  function. For example, to divide `[2, 2]` by `2`, first the `int 2` must
  be packed into the data structure `[x, x]`, before the division operator of
  the appropriate type is called.

  Args:
    operand_type: The type of argument to the constructed binary operator. Must
      be convertible to `computation_types.Type`.
    operator: Callable taking two arguments specifying the operation to encode.
      For example, `tf.add`, `tf.multiply`, `tf.divide`, ...

  Returns:
    Instance of `building_blocks.CompiledComputation` encoding
    this binary operator.

  Raises:
    TypeError: If the type tree of `operand_type` contains any type which is
    incompatible with the TFF generic operators, as checked by
    `type_utils.is_generic_op_compatible_type`, or `operator` is not callable.
  """
  operand_type = computation_types.to_type(operand_type)
  py_typecheck.check_type(operand_type, computation_types.Type)
  py_typecheck.check_callable(operator)
  if not type_utils.is_generic_op_compatible_type(operand_type):
    raise TypeError('The type {} contains a type other than '
                    '`computation_types.TensorType` and '
                    '`computation_types.NamedTupleType`; this is disallowed '
                    'in the generic operators.'.format(operand_type))
  with tf.Graph().as_default() as graph:
    operand_1_value, operand_1_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', operand_type, graph)
    operand_2_value, operand_2_binding = tensorflow_utils.stamp_parameter_in_graph(
        'y', operand_type, graph)

    if isinstance(operand_type, computation_types.TensorType):
      result_value = operator(operand_1_value, operand_2_value)
    elif isinstance(operand_type, computation_types.NamedTupleType):
      result_value = anonymous_tuple.map_structure(operator, operand_1_value,
                                                   operand_2_value)
    else:
      raise TypeError('Operand type {} cannot be used in generic operations. '
                      'The whitelist in '
                      '`type_utils.is_generic_op_compatible_type` has allowed '
                      'it to pass, and should be updated.'.format(operand_type))
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph)

  function_type = computation_types.FunctionType(
      computation_types.NamedTupleType([operand_type, operand_type]),
      result_type)
  serialized_function_type = type_serialization.serialize_type(function_type)

  parameter_binding = pb.TensorFlow.Binding(
      tuple=pb.TensorFlow.NamedTupleBinding(
          element=[operand_1_binding, operand_2_binding]))

  proto = pb.Computation(
      type=serialized_function_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=parameter_binding,
          result=result_binding))

  return building_blocks.CompiledComputation(proto)


def create_federated_getitem_call(arg, idx):
  """Creates computation building block passing getitem to federated value.

  Args:
    arg: Instance of `building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `idx`.
    idx: Index, instance of `int` or `slice` used to address the
      `computation_types.NamedTupleType` underlying `arg`.

  Returns:
    Returns a `building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`, the result
    of applying or mapping the appropriate `__getitem__` function, as defined
    by `idx`.
  """
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(idx, (int, slice))
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getitem_comp = create_federated_getitem_comp(arg, idx)
  return create_federated_map_or_apply(getitem_comp, arg)


def create_federated_getattr_call(arg, name):
  """Creates computation building block passing getattr to federated value.

  Args:
    arg: Instance of `building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.NamedTupleType` from which we wish to pick out item
      `name`.
    name: String name to address the `computation_types.NamedTupleType`
      underlying `arg`.

  Returns:
    Returns a `building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`,
    the result of applying or mapping the appropriate `__getattr__` function,
    as defined by `name`.
  """
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.NamedTupleType)
  getattr_comp = create_federated_getattr_comp(arg, name)
  return create_federated_map_or_apply(getattr_comp, arg)


def create_federated_setattr_call(federated_comp, name, value_comp):
  """Returns building block for `setattr(name, value_comp)` on `federated_comp`.

  Creates an appropriate communication intrinsic (either `federated_map` or
  `federated_apply`) as well as a `building_blocks.Lambda`
  representing setting the `name` attribute of `federated_comp`'s `member` to
  `value_comp`, and stitches these together in a call.

  Notice that `federated_comp`'s `member` must actually define a `name`
  attribute; this is enforced to avoid the need to worry about theplacement of a
  previously undefined name.

  Args:
    federated_comp: Instance of `building_blocks.ComputationBuildingBlock` of
      type `computation_types.FederatedType`, with member of type
      `computation_types.NamedTupleType` whose attribute `name` we wish to set
      to `value_comp`.
    name: String name of the attribute we wish to overwrite in `federated_comp`.
    value_comp: Instance of `building_blocks.ComputationBuildingBlock`, the
      value to assign to `federated_comp`'s `member`'s `name` attribute.

  Returns:
    Instance of `building_blocks.ComputationBuildingBlock`
    representing `federated_comp` with its `member`'s `name` attribute set to
    `value`.
  """
  py_typecheck.check_type(federated_comp,
                          building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(value_comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(federated_comp.type_signature,
                          computation_types.FederatedType)
  py_typecheck.check_type(federated_comp.type_signature.member,
                          computation_types.NamedTupleType)
  named_tuple_type_signature = federated_comp.type_signature.member
  setattr_lambda = create_named_tuple_setattr_lambda(named_tuple_type_signature,
                                                     name, value_comp)
  return create_federated_map_or_apply(setattr_lambda, federated_comp)


def create_named_tuple_setattr_lambda(named_tuple_signature, name, value_comp):
  """Creates a building block for replacing one attribute in a named tuple.

  Returns an instance of `building_blocks.Lambda` which takes an
  argument of type `computation_types.NamedTupleType` and returns a
  `building_blocks.Tuple` which contains all the same elements as
  the argument, except the attribute `name` now has value `value_comp`. The
  Lambda constructed is the analogue of Python's `setattr` for the concrete
  type `named_tuple_signature`.

  Args:
    named_tuple_signature: Instance of `computation_types.NamedTupleType`, the
      type of the argument to the constructed `building_blocks.Lambda`.
    name: String name of the attribute in the `named_tuple_signature` to replace
      with `value_comp`. Must be present as a name in `named_tuple_signature;
      otherwise we will raise an `AttributeError`.
    value_comp: Instance of `building_blocks.ComputationBuildingBlock`, the
      value to place as attribute `name` in the argument of the returned
      function.

  Returns:
    An instance of `building_blocks.Block` of functional type
    representing setting attribute `name` to value `value_comp` in its argument
    of type `named_tuple_signature`.

  Raises:
    TypeError: If the types of the arguments don't match the assumptions above.
    AttributeError: If `name` is not present as a named element in
      `named_tuple_signature`
  """
  py_typecheck.check_type(named_tuple_signature,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(name, six.string_types)
  py_typecheck.check_type(value_comp, building_blocks.ComputationBuildingBlock)
  value_comp_placeholder = building_blocks.Reference('value_comp_placeholder',
                                                     value_comp.type_signature)
  lambda_arg = building_blocks.Reference('lambda_arg', named_tuple_signature)
  if name not in dir(named_tuple_signature):
    raise AttributeError(
        'There is no such attribute as \'{}\' in this federated tuple. '
        'TFF does not allow for assigning to a nonexistent attribute. '
        'If you want to assign to \'{}\', you must create a new named tuple '
        'containing this attribute.'.format(name, name))
  elements = []
  for idx, (key, element_type) in enumerate(
      anonymous_tuple.to_elements(named_tuple_signature)):
    if key == name:
      if not type_utils.is_assignable_from(element_type,
                                           value_comp.type_signature):
        raise TypeError(
            '`setattr` has attempted to set element {} of type {} with incompatible type {}'
            .format(key, element_type, value_comp.type_signature))
      elements.append((key, value_comp_placeholder))
    else:
      elements.append((key, building_blocks.Selection(lambda_arg, index=idx)))
  return_tuple = building_blocks.Tuple(elements)
  lambda_to_return = building_blocks.Lambda(lambda_arg.name,
                                            named_tuple_signature, return_tuple)
  symbols = ((value_comp_placeholder.name, value_comp),)
  return building_blocks.Block(symbols, lambda_to_return)


def create_federated_getattr_comp(comp, name):
  """Function to construct computation for `federated_apply` of `__getattr__`.

  Creates a `building_blocks.ComputationBuildingBlock`
  which selects `name` from its argument, of type `comp.type_signature.member`,
  an instance of `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` with type
      signature `computation_types.FederatedType` whose `member` attribute is of
      type `computation_types.NamedTupleType`.
    name: String name of attribute to grab.

  Returns:
    Instance of `building_blocks.Lambda` which grabs attribute
      according to `name` of its argument.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(name, six.string_types)
  element_names = [
      x for x, _ in anonymous_tuple.to_elements(comp.type_signature.member)
  ]
  if name not in element_names:
    raise ValueError('The federated value {} has no element of name {}'.format(
        comp, name))
  apply_input = building_blocks.Reference('x', comp.type_signature.member)
  selected = building_blocks.Selection(apply_input, name=name)
  apply_lambda = building_blocks.Lambda('x', apply_input.type_signature,
                                        selected)
  return apply_lambda


def create_federated_getitem_comp(comp, key):
  """Function to construct computation for `federated_apply` of `__getitem__`.

  Creates a `building_blocks.ComputationBuildingBlock`
  which selects `key` from its argument, of type `comp.type_signature.member`,
  of type `computation_types.NamedTupleType`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` with type
      signature `computation_types.FederatedType` whose `member` attribute is of
      type `computation_types.NamedTupleType`.
    key: Instance of `int` or `slice`, key used to grab elements from the member
      of `comp`. implementation of slicing for `ValueImpl` objects with
      `type_signature` `computation_types.NamedTupleType`.

  Returns:
    Instance of `building_blocks.Lambda` which grabs slice
      according to `key` of its argument.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.NamedTupleType)
  py_typecheck.check_type(key, (int, slice))
  apply_input = building_blocks.Reference('x', comp.type_signature.member)
  if isinstance(key, int):
    selected = building_blocks.Selection(apply_input, index=key)
  else:
    elems = anonymous_tuple.to_elements(comp.type_signature.member)
    index_range = range(*key.indices(len(elems)))
    elem_list = []
    for k in index_range:
      elem_list.append(
          (elems[k][0], building_blocks.Selection(apply_input, index=k)))
    selected = building_blocks.Tuple(elem_list)
  apply_lambda = building_blocks.Lambda('x', apply_input.type_signature,
                                        selected)
  return apply_lambda


def create_computation_appending(comp1, comp2):
  r"""Returns a block appending `comp2` to `comp1`.

                Block
               /     \
  [comps=Tuple]       Tuple
         |            |
    [Comp, Comp]      [Sel(0), ...,  Sel(0),   Sel(1)]
                             \             \         \
                              Sel(0)        Sel(n)    Ref(comps)
                                    \             \
                                     Ref(comps)    Ref(comps)

  Args:
    comp1: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_type.NamedTupleType`.
    comp2: A `building_blocks.ComputationBuildingBlock` or a named computation
      (a tuple pair of name, computation) representing a single element of an
      `anonymous_tuple.AnonymousTuple`.

  Returns:
    A `building_blocks.Block`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(comp1, building_blocks.ComputationBuildingBlock)
  if isinstance(comp2, building_blocks.ComputationBuildingBlock):
    name2 = None
  elif py_typecheck.is_name_value_pair(
      comp2,
      name_required=False,
      value_type=building_blocks.ComputationBuildingBlock):
    name2, comp2 = comp2
  else:
    raise TypeError('Unexpected tuple element: {}.'.format(comp2))
  comps = building_blocks.Tuple((comp1, comp2))
  ref = building_blocks.Reference('comps', comps.type_signature)
  sel_0 = building_blocks.Selection(ref, index=0)
  elements = []
  named_type_signatures = anonymous_tuple.to_elements(comp1.type_signature)
  for index, (name, _) in enumerate(named_type_signatures):
    sel = building_blocks.Selection(sel_0, index=index)
    elements.append((name, sel))
  sel_1 = building_blocks.Selection(ref, index=1)
  elements.append((name2, sel_1))
  result = building_blocks.Tuple(elements)
  symbols = ((ref.name, comps),)
  return building_blocks.Block(symbols, result)


def create_federated_aggregate(value, zero, accumulate, merge, report):
  r"""Creates a called federated aggregate.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp, Comp, Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    zero: A `building_blocks.ComputationBuildingBlock` to use as the initial
      value.
    accumulate: A `building_blocks.ComputationBuildingBlock` to use as the
      accumulate function.
    merge: A `building_blocks.ComputationBuildingBlock` to use as the merge
      function.
    report: A `building_blocks.ComputationBuildingBlock` to use as the report
      function.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(zero, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(accumulate, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(merge, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(report, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(report.type_signature.result,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType((
      type_utils.to_non_all_equal(value.type_signature),
      zero.type_signature,
      accumulate.type_signature,
      merge.type_signature,
      report.type_signature,
  ), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_AGGREGATE.uri,
                                        intrinsic_type)
  values = building_blocks.Tuple((value, zero, accumulate, merge, report))
  return building_blocks.Call(intrinsic, values)


def create_federated_apply(fn, arg):
  r"""Creates a called federated apply.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_APPLY.uri,
                                        intrinsic_type)
  values = building_blocks.Tuple((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_federated_broadcast(value):
  r"""Creates a called federated broadcast.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(
      value.type_signature.member, placement_literals.CLIENTS, all_equal=True)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_BROADCAST.uri,
                                        intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def create_federated_collect(value):
  r"""Creates a called federated collect.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  type_signature = computation_types.SequenceType(value.type_signature.member)
  result_type = computation_types.FederatedType(type_signature,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType(
      type_utils.to_non_all_equal(value.type_signature), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_COLLECT.uri,
                                        intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def create_federated_map(fn, arg):
  r"""Creates a called federated map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  parameter_type = computation_types.FederatedType(arg.type_signature.member,
                                                   placement_literals.CLIENTS)
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                placement_literals.CLIENTS)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, parameter_type), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_MAP.uri,
                                        intrinsic_type)
  values = building_blocks.Tuple((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_federated_map_all_equal(fn, arg):
  r"""Creates a called federated map of equal values.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  NOTE: The `fn` is required to be deterministic and therefore should contain no
  `building_blocks.CompiledComputations`.

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  parameter_type = computation_types.FederatedType(
      arg.type_signature.member, placement_literals.CLIENTS, all_equal=True)
  result_type = computation_types.FederatedType(
      fn.type_signature.result, placement_literals.CLIENTS, all_equal=True)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, parameter_type), result_type)
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri, intrinsic_type)
  values = building_blocks.Tuple((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_federated_map_or_apply(fn, arg):
  r"""Creates a called federated map or apply depending on `arg`s placement.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  if arg.type_signature.placement is placement_literals.CLIENTS:
    if arg.type_signature.all_equal:
      return create_federated_map_all_equal(fn, arg)
    else:
      return create_federated_map(fn, arg)
  elif arg.type_signature.placement is placement_literals.SERVER:
    return create_federated_apply(fn, arg)
  else:
    raise TypeError('Unsupported placement {}.'.format(
        arg.type_signature.placement))


def create_federated_mean(value, weight):
  r"""Creates a called federated mean.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    weight: A `building_blocks.ComputationBuildingBlock` to use as the weight or
      `None`.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  if weight is not None:
    py_typecheck.check_type(weight, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(value.type_signature.member,
                                                placement_literals.SERVER)
  if weight is not None:
    intrinsic_type = computation_types.FunctionType(
        (type_utils.to_non_all_equal(value.type_signature),
         type_utils.to_non_all_equal(weight.type_signature)), result_type)
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri, intrinsic_type)
    values = building_blocks.Tuple((value, weight))
    return building_blocks.Call(intrinsic, values)
  else:
    intrinsic_type = computation_types.FunctionType(
        type_utils.to_non_all_equal(value.type_signature), result_type)
    intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_MEAN.uri,
                                          intrinsic_type)
    return building_blocks.Call(intrinsic, value)


def create_federated_reduce(value, zero, op):
  r"""Creates a called federated reduce.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    zero: A `building_blocks.ComputationBuildingBlock` to use as the initial
      value.
    op: A `building_blocks.ComputationBuildingBlock` to use as the op function.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(zero, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(op, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(op.type_signature.result,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType((
      type_utils.to_non_all_equal(value.type_signature),
      zero.type_signature,
      op.type_signature,
  ), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_REDUCE.uri,
                                        intrinsic_type)
  values = building_blocks.Tuple((value, zero, op))
  return building_blocks.Call(intrinsic, values)


def create_federated_sum(value):
  r"""Creates a called federated sum.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(value.type_signature.member,
                                                placement_literals.SERVER)
  intrinsic_type = computation_types.FunctionType(
      type_utils.to_non_all_equal(value.type_signature), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_SUM.uri,
                                        intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def create_federated_unzip(value):
  r"""Creates a tuple of called federated maps or applies.

                Block
               /     \
  [value=Comp]        Tuple
                      |
                      [Call,                        Call, ...]
                      /    \                       /    \
             Intrinsic      Tuple         Intrinsic      Tuple
                            |                            |
                [Lambda(arg), Ref(value)]    [Lambda(arg), Ref(value)]
                            \                            \
                             Sel(0)                       Sel(1)
                                   \                            \
                                    Ref(arg)                     Ref(arg)

  This function returns a tuple of federated values given a `value` with a
  federated tuple type signature.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.NamedTupleType` containing at least one
      element.

  Returns:
    A `building_blocks.Block`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain any elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(
      value.type_signature.member)
  length = len(named_type_signatures)
  if length == 0:
    raise ValueError('federated_zip is only supported on non-empty tuples.')
  value_ref = building_blocks.Reference('value', value.type_signature)
  elements = []
  fn_ref = building_blocks.Reference('arg', named_type_signatures)
  for index, (name, _) in enumerate(named_type_signatures):
    sel = building_blocks.Selection(fn_ref, index=index)
    fn = building_blocks.Lambda(fn_ref.name, fn_ref.type_signature, sel)
    intrinsic = create_federated_map_or_apply(fn, value_ref)
    elements.append((name, intrinsic))
  result = building_blocks.Tuple(elements)
  symbols = ((value_ref.name, value),)
  return building_blocks.Block(symbols, result)


def create_federated_value(value, placement):
  r"""Creates a called federated value.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    placement: A `placement_literals.PlacementLiteral` to use as the placement.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  if placement is placement_literals.CLIENTS:
    uri = intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri
  elif placement is placement_literals.SERVER:
    uri = intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))
  result_type = computation_types.FederatedType(
      value.type_signature, placement, all_equal=True)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  result_type)
  intrinsic = building_blocks.Intrinsic(uri, intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def create_federated_zip(value):
  r"""Creates a called federated zip.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  This function returns a federated tuple given a `value` with a tuple of
  federated values type signature.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.NamedTupleType` containing at least one
      element.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain any elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(value.type_signature)
  names_to_add = [name for name, _ in named_type_signatures]
  length = len(named_type_signatures)
  if length == 0:
    raise ValueError('federated_zip is only supported on non-empty tuples.')
  first_name, first_type_signature = named_type_signatures[0]
  if first_type_signature.placement == placement_literals.CLIENTS:
    map_fn = create_federated_map
  elif first_type_signature.placement == placement_literals.SERVER:
    map_fn = create_federated_apply
  else:
    raise TypeError('Unsupported placement {}.'.format(
        first_type_signature.placement))
  if length == 1:
    ref = building_blocks.Reference('arg', first_type_signature.member)
    values = building_blocks.Tuple(((first_name, ref),))
    fn = building_blocks.Lambda(ref.name, ref.type_signature, values)
    sel = building_blocks.Selection(value, index=0)
    return map_fn(fn, sel)
  else:
    zipped_args = _create_chain_zipped_values(value)
    append_fn = _create_fn_to_append_chain_zipped_values(value)
    unnamed_zip = map_fn(append_fn, zipped_args)
    return create_named_federated_tuple(unnamed_zip, names_to_add)


def create_generic_constant(type_spec, scalar_value):
  """Creates constant for a combination of federated, tuple and tensor types.

  Args:
    type_spec: Instance of `computation_types.Type` containing only federated,
      tuple or tensor types for which we wish to construct a generic constant.
      May also be something convertible to a `computation_types.Type` via
      `computation_types.to_type`.
    scalar_value: The scalar value we wish this constant to have.

  Returns:
    Instance of `building_blocks.ComputationBuildingBlock`
    representing `scalar_value` packed into `type_spec`.

  Raises:
    TypeError: If types don't match their specification in the args section.
      Notice validation of consistency of `type_spec` with `scalar_value` is not
      the rsponsibility of this function.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  inferred_scalar_value_type = type_utils.infer_type(scalar_value)
  if (not isinstance(inferred_scalar_value_type, computation_types.TensorType)
      or inferred_scalar_value_type.shape != tf.TensorShape(())):
    raise TypeError(
        'Must pass a scalar value to `create_generic_constant`; encountered a '
        'value {}'.format(scalar_value))
  if not type_utils.type_tree_contains_only(type_spec, (
      computation_types.FederatedType,
      computation_types.NamedTupleType,
      computation_types.TensorType,
  )):
    raise TypeError
  if type_utils.type_tree_contains_only(type_spec, (
      computation_types.NamedTupleType,
      computation_types.TensorType,
  )):
    return create_tensorflow_constant(type_spec, scalar_value)
  elif isinstance(type_spec, computation_types.FederatedType):
    unplaced_zero = create_tensorflow_constant(type_spec.member, scalar_value)
    if type_spec.placement == placement_literals.CLIENTS:
      placement_federated_type = computation_types.FederatedType(
          type_spec.member, type_spec.placement, all_equal=True)
      placement_fn_type = computation_types.FunctionType(
          type_spec.member, placement_federated_type)
      placement_function = building_blocks.Intrinsic(
          intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri, placement_fn_type)
    elif type_spec.placement == placement_literals.SERVER:
      placement_federated_type = computation_types.FederatedType(
          type_spec.member, type_spec.placement, all_equal=True)
      placement_fn_type = computation_types.FunctionType(
          type_spec.member, placement_federated_type)
      placement_function = building_blocks.Intrinsic(
          intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri, placement_fn_type)
    return building_blocks.Call(placement_function, unplaced_zero)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    elements = []
    for k in range(len(type_spec)):
      elements.append(create_generic_constant(type_spec[k], scalar_value))
    names = [name for name, _ in anonymous_tuple.to_elements(type_spec)]
    packed_elements = building_blocks.Tuple(elements)
    named_tuple = create_named_tuple(packed_elements, names)
    return named_tuple
  else:
    raise ValueError(
        'The type_spec {} has slipped through all our '
        'generic constant cases, and failed to raise.'.format(type_spec))


def _create_chain_zipped_values(value):
  r"""Creates a chain of called federated zip with two values.

                Block--------
               /             \
  [value=Tuple]               Call
         |                   /    \
         [Comp1,    Intrinsic      Tuple
          Comp2,                   |
          ...]                     [Call,  Sel(n)]
                                   /    \        \
                          Intrinsic      Tuple    Ref(value)
                                         |
                                         [Sel(0),       Sel(1)]
                                                \             \
                                                 Ref(value)    Ref(value)

  NOTE: This function is intended to be used in conjunction with
  `_create_fn_to_append_chain_zipped_values` and will drop the tuple names. The
  names will be added back to the resulting computation when the zipped values
  are mapped to a function that flattens the chain. This nested zip -> flatten
  structure must be used since length of a named tuple type in the TFF type
  system is an element of the type proper. That is, a named tuple type of
  length 2 is a different type than a named tuple type of length 3, they are
  not simply items with the same type and different values, as would be the
  case if you were thinking of these as Python `list`s. It may be better to
  think of named tuple types in TFF as more like `struct`s.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.NamedTupleType` containing at least two
      elements.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain at least two elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(value.type_signature)
  length = len(named_type_signatures)
  if length < 2:
    raise ValueError(
        'Expected a value with at least two elements, received {} elements.'
        .format(named_type_signatures))
  ref = building_blocks.Reference('value', value.type_signature)
  symbols = ((ref.name, value),)
  sel_0 = building_blocks.Selection(ref, index=0)
  result = sel_0
  for i in range(1, length):
    sel = building_blocks.Selection(ref, index=i)
    values = building_blocks.Tuple((result, sel))
    result = create_zip_two_values(values)
  return building_blocks.Block(symbols, result)


def create_zip_two_values(value):
  r"""Creates a called federated zip with two values.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp1, Comp2]

  Notice that this function will drop any names associated to the two-tuple it
  is processing. This is necessary due to the type signature of the
  underlying federated zip intrinsic, `<T@P,U@P>-><T,U>@P`. Keeping names
  here would violate this type signature. The names are cached at a higher
  level than this function, and appended to the resulting tuple in a single
  call to `federated_map` or `federated_apply` before the resulting structure
  is sent back to the caller.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.NamedTupleType` containing exactly two
      elements.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain exactly two elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(value.type_signature)
  length = len(named_type_signatures)
  if length != 2:
    raise ValueError(
        'Expected a value with exactly two elements, received {} elements.'
        .format(named_type_signatures))
  placement = value[0].type_signature.placement
  if placement is placement_literals.CLIENTS:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri
    all_equal = False
  elif placement is placement_literals.SERVER:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri
    all_equal = True
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))
  elements = []
  for _, type_signature in named_type_signatures:
    federated_type = computation_types.FederatedType(type_signature.member,
                                                     placement, all_equal)
    elements.append((None, federated_type))
  parameter_type = computation_types.NamedTupleType(elements)
  result_type = computation_types.FederatedType(
      [(None, e.member) for _, e in named_type_signatures], placement,
      all_equal)
  intrinsic_type = computation_types.FunctionType(parameter_type, result_type)
  intrinsic = building_blocks.Intrinsic(uri, intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def _create_fn_to_append_chain_zipped_values(value):
  r"""Creates a function to append a chain of zipped values.

  Lambda(arg3)
            \
             append([Call,    Sel(1)])
                    /    \            \
        Lambda(arg2)      Sel(0)       Ref(arg3)
                  \             \
                   \             Ref(arg3)
                    \
                     append([Call,    Sel(1)])
                            /    \            \
                Lambda(arg1)      Sel(0)       Ref(arg2)
                            \           \
                             \           Ref(arg2)
                              \
                               Ref(arg1)

  Note that this function will not respect any names it is passed; names
  for tuples will be cached at a higher level than this function and added back
  in a single call to federated map or federated apply.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.NamedTupleType` containing at least two
      elements.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  named_type_signatures = anonymous_tuple.to_elements(value.type_signature)
  length = len(named_type_signatures)
  if length < 2:
    raise ValueError(
        'Expected a value with at least two elements, received {} elements.'
        .format(named_type_signatures))
  _, first_type_signature = named_type_signatures[0]
  _, second_type_signature = named_type_signatures[1]
  ref_type = computation_types.NamedTupleType((
      first_type_signature.member,
      second_type_signature.member,
  ))
  ref = building_blocks.Reference('arg', ref_type)
  fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
  for _, type_signature in named_type_signatures[2:]:
    ref_type = computation_types.NamedTupleType((
        fn.type_signature.parameter,
        type_signature.member,
    ))
    ref = building_blocks.Reference('arg', ref_type)
    sel_0 = building_blocks.Selection(ref, index=0)
    call = building_blocks.Call(fn, sel_0)
    sel_1 = building_blocks.Selection(ref, index=1)
    result = create_computation_appending(call, sel_1)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, result)
  return fn


def create_sequence_map(fn, arg):
  r"""Creates a called sequence map.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    arg: A `building_blocks.ComputationBuildingBlock` to use as the argument.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.SequenceType(fn.type_signature.result)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_MAP.uri,
                                        intrinsic_type)
  values = building_blocks.Tuple((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_sequence_reduce(value, zero, op):
  r"""Creates a called sequence reduce.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    zero: A `building_blocks.ComputationBuildingBlock` to use as the initial
      value.
    op: A `building_blocks.ComputationBuildingBlock` to use as the op function.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(zero, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(op, building_blocks.ComputationBuildingBlock)
  intrinsic_type = computation_types.FunctionType((
      value.type_signature,
      zero.type_signature,
      op.type_signature,
  ), op.type_signature.result)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_REDUCE.uri,
                                        intrinsic_type)
  values = building_blocks.Tuple((value, zero, op))
  return building_blocks.Call(intrinsic, values)


def create_sequence_sum(value):
  r"""Creates a called sequence sum.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  value.type_signature.element)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_SUM.uri,
                                        intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def _create_naming_function(tuple_type_to_name, names_to_add):
  """Private function to construct lambda naming a given tuple type.

  Args:
    tuple_type_to_name: Instance of `computation_types.NamedTupleType`, the type
      of the argument which we wish to name.
    names_to_add: Python `list` or `tuple`, the names we wish to give to
      `tuple_type_to_name`.

  Returns:
    An instance of `building_blocks.Lambda` representing a function
    which will take an argument of type `tuple_type_to_name` and return a tuple
    with the same elements, but with names in `names_to_add` attached.

  Raises:
    ValueError: If `tuple_type_to_name` and `names_to_add` have different
    lengths.
  """
  py_typecheck.check_type(tuple_type_to_name, computation_types.NamedTupleType)
  if len(names_to_add) != len(tuple_type_to_name):
    raise ValueError(
        'Number of elements in `names_to_add` must match number of element in '
        'the named tuple type `tuple_type_to_name`; here, `names_to_add` has '
        '{} elements and `tuple_type_to_name` has {}.'.format(
            len(names_to_add), len(tuple_type_to_name)))
  naming_lambda_arg = building_blocks.Reference('x', tuple_type_to_name)

  def _create_tuple_element(i):
    return (names_to_add[i],
            building_blocks.Selection(naming_lambda_arg, index=i))

  named_result = building_blocks.Tuple(
      [_create_tuple_element(k) for k in range(len(names_to_add))])
  return building_blocks.Lambda('x', naming_lambda_arg.type_signature,
                                named_result)


def create_named_federated_tuple(tuple_to_name, names_to_add):
  """Name tuple elements with names in `names_to_add`.

  Certain intrinsics, e.g. `federated_zip`, only accept unnamed tuples as
  arguments, and can only produce unnamed tuples as their outputs. This is not
  necessarily desirable behavior, as it necessitates dropping any names that
  exist before the zip. This function is intended to provide a general remedy
  for this shortcoming, so that a tuple can be renamed after it is passed
  through any function which drops its names.

  Args:
    tuple_to_name: Instance of `building_blocks.ComputationBuildingBlock` of
      type `computation_types.FederatedType` with
      `computation_types.NamedTupleType` member, to populate with names from
      `names_to_add`.
    names_to_add: Python `tuple` or `list` containing instances of type `str` or
      `None`, the names to give to `tuple_to_name`.

  Returns:
    An instance of `building_blocks.ComputationBuildingBlock`
    representing a federated tuple with the same elements as `tuple_to_name`
    but with the names from `names_to_add` attached to the type
    signature. Notice that if these names are already present in
    `tuple_to_name`, `create_naming_function` represents the identity.

  Raises:
    TypeError: If the types do not match the description above.
  """
  py_typecheck.check_type(names_to_add, (list, tuple))
  element_types_to_accept = six.string_types + (type(None),)
  if not all(isinstance(x, element_types_to_accept) for x in names_to_add):
    raise TypeError('`names_to_add` must contain only instances of `str` or '
                    'NoneType; you have passed in {}'.format(names_to_add))
  py_typecheck.check_type(tuple_to_name,
                          building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(tuple_to_name.type_signature,
                          computation_types.FederatedType)

  naming_fn = _create_naming_function(tuple_to_name.type_signature.member,
                                      names_to_add)
  return create_federated_map_or_apply(naming_fn, tuple_to_name)


def create_named_tuple(comp, names):
  """Creates a computation that applies `names` to `comp`.

  Args:
    comp: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.NamedTupleType`.
    names: Python `tuple` or `list` containing instances of type `str` or
      `None`, the names to apply to `comp`.

  Returns:
    A `building_blocks.ComputationBuildingBlock` representing a
    tuple with the elements from `comp` and the names from `names` attached to
    the `type_signature` of those elements.

  Raises:
    TypeError: If the types do not match.
  """
  py_typecheck.check_type(names, (list, tuple))
  if not all(isinstance(x, (six.string_types, type(None))) for x in names):
    raise TypeError('Expected `names` containing only instances of `str` or '
                    '`None`, found {}'.format(names))
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.NamedTupleType)
  fn = _create_naming_function(comp.type_signature, names)
  return building_blocks.Call(fn, comp)


def create_zip(comp):
  r"""Returns a computation which zips `comp`.

  Returns the following computation where `x` is `comp` unless `comp` is a
  Reference, in which case the Reference is inlined and the Tuple is returned.

           Block
          /     \
  [comp=x]       Tuple
                 |
                 [Tuple,                    Tuple]
                  |                         |
                  [Sel(0),      Sel(0)]     [Sel(1),      Sel(1)]
                   |            |            |            |
                   Sel(0)       Sel(1)       Sel(0)       Sel(1)
                   |            |            |            |
                   Ref(comp)    Ref(comp)    Ref(comp)    Ref(comp)

  The returned computation intentionally drops names from the tuples, otherwise
  it would be possible for the resulting type signature to contain a Tuple where
  two elements have the same name and this is not allowed. It is left up to the
  caller to descide if and where to add the names back.

  Args:
    comp: The computation building block in which to perform the merges.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.NamedTupleType)
  named_type_signatures = anonymous_tuple.to_elements(comp.type_signature)
  _, first_type_signature = named_type_signatures[0]
  py_typecheck.check_type(first_type_signature,
                          computation_types.NamedTupleType)
  length = len(first_type_signature)
  for _, type_signature in named_type_signatures:
    py_typecheck.check_type(type_signature, computation_types.NamedTupleType)
    if len(type_signature) != length:
      raise TypeError(
          'Expected a NamedTupleType containing NamedTupleTypes with the same '
          'length, found: {}'.format(comp.type_signature))
  if not isinstance(comp, building_blocks.Reference):
    name_generator = unique_name_generator(comp)
    name = six.next(name_generator)
    ref = building_blocks.Reference(name, comp.type_signature)
  else:
    ref = comp
  rows = []
  for column in range(len(first_type_signature)):
    columns = []
    for row in range(len(named_type_signatures)):
      sel_row = building_blocks.Selection(ref, index=row)
      sel_column = building_blocks.Selection(sel_row, index=column)
      columns.append(sel_column)
    tup = building_blocks.Tuple(columns)
    rows.append(tup)
  tup = building_blocks.Tuple(rows)
  if not isinstance(comp, building_blocks.Reference):
    return building_blocks.Block(((ref.name, comp),), tup)
  else:
    return tup


def _check_generic_operator_type(type_spec):
  """Checks that `type_spec` can be the signature of args to a generic op."""
  if not type_utils.type_tree_contains_only(type_spec, (
      computation_types.FederatedType,
      computation_types.NamedTupleType,
      computation_types.TensorType,
  )):
    raise TypeError(
        'Generic operators are only implemented for arguments both containing '
        'only federated, tuple and tensor types; you have passed an argument '
        'of type {} '.format(type_spec))
  if not (isinstance(type_spec, computation_types.NamedTupleType) and
          len(type_spec) == 2):
    raise TypeError(
        'We are trying to construct a generic operator declaring argument that '
        'is not a two-tuple, the type {}.'.format(type_spec))
  if not type_utils.is_binary_op_with_upcast_compatible_pair(
      type_spec[0], type_spec[1]):
    raise TypeError(
        'The two-tuple you have passed in is incompatible with upcasted '
        'binary operators. You have passed the tuple type {}, which fails the '
        'check that the two members of the tuple are either the same type, or '
        'the second is a scalar with the same dtype as the leaves of the '
        'first. See `type_utils.is_binary_op_with_upcast_compatible_pair` for '
        'more details.'.format(type_spec))


def create_binary_operator_with_upcast(type_signature, operator):
  """Creates lambda upcasting its argument and applying `operator`.

  The concept of upcasting is explained further in the docstring for
  `apply_binary_operator_with_upcast`.

  Notice that since we are constructing a function here, e.g. for the body
  of an intrinsic, the function we are constructing must be reducible to
  TensorFlow. Therefore `type_signature` can only have named tuple or tensor
  type elements; that is, we cannot handle federated types here in a generic
  way.

  Args:
    type_signature: Value convertible to `computation_types.NamedTupleType`,
      with two elements, both of the same type or the second able to be upcast
      to the first, as explained in `apply_binary_operator_with_upcast`, and
      both containing only tuples and tensors in their type tree.
    operator: Callable defining the operator.

  Returns:
    A `building_blocks.Lambda` encapsulating a function which
    upcasts the second element of its argument and applies the binary
    operator.
  """
  py_typecheck.check_callable(operator)
  type_signature = computation_types.to_type(type_signature)
  _check_generic_operator_type(type_signature)
  ref_to_arg = building_blocks.Reference('binary_operator_arg', type_signature)

  def _pack_into_type(to_pack, type_spec):
    """Pack Tensor value `to_pack` into the nested structure `type_spec`."""
    if isinstance(type_spec, computation_types.NamedTupleType):
      elems = anonymous_tuple.to_elements(type_spec)
      packed_elems = [(elem_name, _pack_into_type(to_pack, elem_type))
                      for elem_name, elem_type in elems]
      return building_blocks.Tuple(packed_elems)
    elif isinstance(type_spec, computation_types.TensorType):
      expand_fn = create_tensorflow_to_broadcast_scalar(
          to_pack.type_signature.dtype, type_spec.shape)
      return building_blocks.Call(expand_fn, to_pack)

  y_ref = building_blocks.Selection(ref_to_arg, index=1)
  first_arg = building_blocks.Selection(ref_to_arg, index=0)

  if type_utils.are_equivalent_types(first_arg.type_signature,
                                     y_ref.type_signature):
    second_arg = y_ref
  else:
    second_arg = _pack_into_type(y_ref, first_arg.type_signature)

  fn = create_tensorflow_binary_operator(first_arg.type_signature, operator)
  packed = building_blocks.Tuple([first_arg, second_arg])
  operated = building_blocks.Call(fn, packed)
  lambda_encapsulating_op = building_blocks.Lambda(ref_to_arg.name,
                                                   ref_to_arg.type_signature,
                                                   operated)
  return lambda_encapsulating_op


def apply_binary_operator_with_upcast(arg, operator):
  """Constructs result of applying `operator` to `arg` upcasting if appropriate.

  Notice `arg` here must be of federated type, with a named tuple member of
  length 2, or a named tuple type of length 2. If the named tuple type of `arg`
  satisfies certain conditions (that is, there is only a single tensor dtype in
  the first element of `arg`, and the second element represents a scalar of
  this dtype), the second element will be upcast to match the first. Here this
  means it will be pushed into a nested structure matching the structure of the
  first element of `arg`. For example, it makes perfect sense to divide a model
  of type `<a=float32[784],b=float32[10]>` by a scalar of type `float32`, but
  the binary operator constructors we have implemented only take arguments of
  type `<T, T>`. Therefore in this case we would broadcast the `float` argument
  to the `tuple` type, before constructing a biary operator which divides
  pointwise.

  Args:
    arg: `building_blocks.ComputationBuildingBlock` of federated type whose
      `member` attribute is a named tuple type of length 2, or named tuple type
      of length 2.
    operator: Callable representing binary operator to apply to the 2-tuple
      represented by the federated `arg`.

  Returns:
    Instance of `building_blocks.ComputationBuildingBlock`
    encapsulating the result of formally applying `operator` to
    `arg[0], `arg[1]`, upcasting `arg[1]` in the condition described above.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_callable(operator)
  if isinstance(arg.type_signature, computation_types.FederatedType):
    py_typecheck.check_type(arg.type_signature.member,
                            computation_types.NamedTupleType)
    tuple_type = arg.type_signature.member
  elif isinstance(arg.type_signature, computation_types.NamedTupleType):
    tuple_type = arg.type_signature
  else:
    raise TypeError(
        'Generic binary operators are only implemented for federated tuple and '
        'unplaced tuples; you have passed {}.'.format(arg.type_signature))

  lambda_encapsulating_op = create_binary_operator_with_upcast(
      tuple_type, operator)

  if isinstance(arg.type_signature, computation_types.FederatedType):
    called = create_federated_map_or_apply(lambda_encapsulating_op, arg)
  else:
    called = building_blocks.Call(lambda_encapsulating_op, arg)

  return called
