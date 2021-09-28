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
"""A library of contruction functions for building block structures."""

import functools
import random
import string
from typing import AbstractSet, Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


Index = Union[str, int]
Path = Union[Index, Tuple[Index, ...]]


def select_output_from_lambda(
    comp: building_blocks.Lambda,
    paths: Union[Path, List[Path]]) -> building_blocks.Lambda:
  """Constructs a new function with result of selecting `paths` from `comp`.

  Args:
    comp: Lambda computation with result type `tff.StructType` from which we
      wish to select the sub-results at `paths`.
    paths: Either a `Path` or list of `Path`s specifying the indices we wish to
      select from the result of `comp`. Each path must be a `tuple` of `str` or
      `int` indices from which to select an output. If `paths` is a list, the
      returned computation will have a `tff.StructType` result holding each of
      the specified selections.

  Returns:
    A version of `comp` with result value the selection from the result of
    `comp` specified by `paths`.
  """
  comp.check_lambda()
  comp.type_signature.result.check_struct()

  def _select_path(result, path: Path):
    if not isinstance(path, tuple):
      path = (path,)
    for index in path:
      if result.is_struct():
        result = result[index]
      elif isinstance(index, str):
        result = building_blocks.Selection(result, name=index)
      elif isinstance(index, int):
        result = building_blocks.Selection(result, index=index)
      else:
        raise TypeError('Invalid selection type: expected `str` or `int`, '
                        f'found value `{index}` of type `{type(index)}`.')
    return result

  if isinstance(paths, list):
    elements = [_select_path(comp.result, path) for path in paths]
    result = building_blocks.Struct(elements)
  else:
    result = _select_path(comp.result, paths)
  return building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                result)


def unique_name_generator(comp: building_blocks.ComputationBuildingBlock,
                          prefix: str = '_var') -> Iterator[str]:
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


@functools.lru_cache()
def create_compiled_no_arg_empty_tuple_computation(
) -> building_blocks.CompiledComputation:
  """Returns graph representing a function that returns an empty tuple.

  Returns:
    An instance of `building_blocks.CompiledComputation`, a noarg function
    which returns an empty tuple.
  """
  proto, type_signature = tensorflow_computation_factory.create_empty_tuple()
  return building_blocks.CompiledComputation(
      proto, type_signature=type_signature)


@functools.lru_cache()
def create_compiled_empty_tuple() -> building_blocks.Call:
  """Returns called graph representing the empty tuple.

  Returns:
    An instance of `building_blocks.Call`, calling a noarg function
    which returns an empty tuple. This function is an instance of
    `building_blocks.CompiledComputation`.
  """
  compiled = create_compiled_no_arg_empty_tuple_computation()
  return building_blocks.Call(compiled, None)


@functools.lru_cache()
def create_compiled_identity(
    type_signature: computation_types.Type,
    name: Optional[str] = None) -> building_blocks.CompiledComputation:
  """Creates CompiledComputation representing identity function.

  Args:
    type_signature: A `computation_types.Type`.
    name: An optional string name to use as the name of the computation.

  Returns:
    An instance of `building_blocks.CompiledComputation`
    representing the identity function taking an argument of type
    `type_signature` and returning the same value.

  Raises:
    TypeError: If `type_signature` contains any types which cannot appear in
      TensorFlow bindings.
  """
  proto, function_type = tensorflow_computation_factory.create_identity(
      type_signature)
  return building_blocks.CompiledComputation(
      proto, name, type_signature=function_type)


class SelectionSpec(object):
  """Data class representing map from input tuple to selection of result.

  Attributes:
    tuple_index: The index of the source of the selection sequence in the
      desired result of the generated TensorFlow. If this `SelectionSpec`
      appears at index i of a list of `SelectionSpec`s, index j is the source
      for the result of the generated function at index i.
    selection_sequence: A list or tuple representing the selections to make from
      `tuple_index`, so that the list `[0]` for example would represent the
      output is the 0th element of `tuple_index`, while `[0, 0]` would represent
      that the output is the 0th element of the 0th element of `tuple_index`.
  """

  def __init__(self, tuple_index: int, selection_sequence: Sequence[int]):
    self._tuple_index = tuple_index
    self._selection_sequence = selection_sequence

  @property
  def tuple_index(self):
    return self._tuple_index

  @property
  def selection_sequence(self):
    return self._selection_sequence

  def __str__(self):
    return 'SelectionSequence(tuple_index={},selection_sequence={}'.format(
        self._tuple_index, self._selection_sequence)

  def __repr__(self):
    return str(self)


def _extract_selections(parameter_value, output_spec):
  results = []
  for selection_spec in output_spec:
    result_element = parameter_value[selection_spec.tuple_index]
    for selection in selection_spec.selection_sequence:
      py_typecheck.check_type(selection, int)
      result_element = result_element[selection]
    results.append(result_element)
  return results


@functools.lru_cache()
def construct_tensorflow_selecting_and_packing_outputs(
    parameter_type: computation_types.StructType,
    output_structure: structure.Struct) -> building_blocks.CompiledComputation:
  """Constructs TensorFlow selecting and packing elements from its input.

  The result of this function can be called on a deduplicated
  `building_blocks.Struct` containing called graphs, thus preventing us from
  embedding the same TensorFlow computation in the generated graphs, and
  reducing the amount of work duplicated in the process of generating
  TensorFlow.

  The TensorFlow which results here will be a function which takes an argument
  of type `arg_type`, returning a result specified by `output_structure`. Each
  `SelectionSpec` nested inside of `output_structure` will represent a selection
  from one of the arguments of the tuple `arg_type`, with the empty selection
  being a possibility. The nested structure of `output_structure` will determine
  how these selections are packed back into a result, IE, the result of the
  function will be a nested tuple with the same structure as `output_structure`,
  where the leaves of this structure (the `SelectionSpecs` of
  `output_structure`) will be selections from the argument.

  Args:
    parameter_type: A `computation_types.StructType` of the argument on which
      the constructed function will be called.
    output_structure: `structure.Struct` with `SelectionSpec` or
      `anonymous_tupl.Struct` elements, mapping from elements of the nested
      argument tuple to the desired result of the generated computation.
      `output_structure` must contain all the names desired on the output of the
      computation.

  Returns:
    A `building_blocks.CompiledComputation` representing the specification
    above.

  Raises:
    TypeError: If `arg_type` is not a `computation_types.StructType`, or
      represents a type which cannot act as an input or output to a TensorFlow
      computation in TFF, IE does not contain exclusively
      `computation_types.SequenceType`, `computation_types.StructType` or
      `computation_types.TensorType`.
  """
  py_typecheck.check_type(parameter_type, computation_types.StructType)
  py_typecheck.check_type(output_structure, structure.Struct)

  def _check_output_structure(elem):
    if isinstance(elem, structure.Struct):
      for x in elem:
        _check_output_structure(x)
    elif not isinstance(elem, SelectionSpec):
      raise TypeError('output_structure can only contain nested anonymous '
                      'tuples and `SelectionSpecs`; encountered the value {} '
                      'of type {}.'.format(elem, type(elem)))

  _check_output_structure(output_structure)
  output_spec = structure.flatten(output_structure)
  type_analysis.check_tensorflow_compatible_type(parameter_type)
  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', parameter_type, graph)
  results = _extract_selections(parameter_value, output_spec)

  repacked_result = structure.pack_sequence_as(output_structure, results)
  result_type, result_binding = tensorflow_utils.capture_result_from_graph(
      repacked_result, graph)

  function_type = computation_types.FunctionType(parameter_type, result_type)
  serialized_function_type = type_serialization.serialize_type(function_type)
  proto = pb.Computation(
      type=serialized_function_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=parameter_binding,
          result=result_binding))
  return building_blocks.CompiledComputation(
      proto, type_signature=function_type)


@functools.lru_cache()
def create_tensorflow_constant(type_spec: computation_types.Type,
                               scalar_value: Union[int, float, str],
                               name=None) -> building_blocks.Call:
  """Creates called graph returning constant `scalar_value` of type `type_spec`.

  `scalar_value` must be a scalar, and cannot be a float if any of the tensor
  leaves of `type_spec` contain an integer data type. `type_spec` must contain
  only named tuples and tensor types, but these can be arbitrarily nested.

  Args:
    type_spec: A `computation_types.Type` whose resulting type tree can only
      contain named tuples and tensors.
    scalar_value: Scalar value to place in all the tensor leaves of `type_spec`.
    name: An optional string name to use as the name of the computation.

  Returns:
    An instance of `building_blocks.Call`, whose argument is `None`
    and whose function is a noarg
    `building_blocks.CompiledComputation` which returns the
    specified `scalar_value` packed into a TFF structure of type `type_spec.

  Raises:
    TypeError: If the type assumptions above are violated.
  """
  proto, function_type = tensorflow_computation_factory.create_constant(
      scalar_value, type_spec)
  compiled = building_blocks.CompiledComputation(
      proto, name, type_signature=function_type)
  return building_blocks.Call(compiled, None)


@functools.lru_cache()
def create_compiled_input_replication(
    type_signature: computation_types.Type,
    n_replicas: int) -> building_blocks.CompiledComputation:
  """Creates a compiled computation which replicates its argument.

  Args:
    type_signature: A `computation_types.Type`, the type of the parameter of the
      constructed computation.
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
  proto, comp_type = tensorflow_computation_factory.create_replicate_input(
      type_signature, n_replicas)
  return building_blocks.CompiledComputation(proto, type_signature=comp_type)


def create_tensorflow_unary_operator(
    operator: Callable[[Any], Any], operand_type: computation_types.Type
) -> building_blocks.CompiledComputation:
  """Creates a TensorFlow computation for the unary `operator`.

  For `T` the `operand_type`, the type signature of the constructed operator
  will be `(T -> U)`, where `U` is the result of applying `operator` to
  a value of type `T`.

  Notice that we have quite serious restrictions on `operand_type` here; not
  only must it be compatible with stamping into a TensorFlow graph, but
  additionally cannot contain a `computation_types.SequenceType`, as checked by
  `type_analysis.is_generic_op_compatible_type`.

  Args:
    operator: Callable taking one argument specifying the operation to encode.
      For example, `tf.math.abs`, `tf.math.reduce_sum`, ...
    operand_type: The type of argument to the constructed unary operator. Must
      be convertible to `computation_types.Type`.

  Returns:
    Instance of `building_blocks.CompiledComputation` encoding this unary
    operator.

  Raises:
    TypeError: If the type tree of `operand_type` contains any type which is
    incompatible with the TFF generic operators, as checked by
    `type_analysis.is_generic_op_compatible_type`, or `operator` is not
    callable.
  """
  proto, type_signature = tensorflow_computation_factory.create_unary_operator(
      operator, operand_type)
  return building_blocks.CompiledComputation(
      proto, type_signature=type_signature)


def create_tensorflow_binary_operator(
    operator: Callable[[Any, Any], Any],
    operand_type: computation_types.Type,
    second_operand_type: Optional[computation_types.Type] = None
) -> building_blocks.CompiledComputation:
  """Creates a TensorFlow computation for the binary `operator`.

  For `T` the `operand_type`, the type signature of the constructed operator
  will be `(<T,T> -> U)`, where `U` is the result of applying `operator` to
  a tuple of type `<T,T>`.

  Notice that we have quite serious restrictions on `operand_type` here; not
  only must it be compatible with stamping into a TensorFlow graph, but
  additionally cannot contain a `computation_types.SequenceType`, as checked by
  `type_analysis.is_generic_op_compatible_type`.

  Notice also that if `operand_type` is a `computation_types.StructType` and
  `second_operand_type` is not `None`, `operator` will be applied pointwise.
  This places the burden on callers of this function to construct the correct
  values to pass into the returned function. For example, to divide `[2, 2]` by
  `2`, first the `int 2` must be packed into the data structure `[x, x]`, before
  the division operator of the appropriate type is called.

  Args:
    operator: Callable taking two arguments specifying the operation to encode.
      For example, `tf.add`, `tf.multiply`, `tf.divide`, ...
    operand_type: The type of argument to the constructed binary operator. Must
      be convertible to `computation_types.Type`.
    second_operand_type: An optional type for the second argument to the
      constructed binary operator. Must be convertible to
      `computation_types.Type`. If `None`, uses `operand_type` for the second
      argument's type.

  Returns:
    Instance of `building_blocks.CompiledComputation` encoding
    this binary operator.

  Raises:
    TypeError: If the type tree of `operand_type` contains any type which is
    incompatible with the TFF generic operators, as checked by
    `type_analysis.is_generic_op_compatible_type`, or `operator` is not
    callable.
  """
  proto, type_signature = tensorflow_computation_factory.create_binary_operator(
      operator, operand_type, second_operand_type)
  return building_blocks.CompiledComputation(
      proto, type_signature=type_signature)


def create_federated_getitem_call(
    arg: building_blocks.ComputationBuildingBlock,
    idx: Union[int, slice]) -> building_blocks.Call:
  """Creates computation building block passing getitem to federated value.

  Args:
    arg: Instance of `building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.StructType` from which we wish to pick out item `idx`.
    idx: Index, instance of `int` or `slice` used to address the
      `computation_types.StructType` underlying `arg`.

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
                          computation_types.StructType)
  getitem_comp = create_federated_getitem_comp(arg, idx)
  return create_federated_map_or_apply(getitem_comp, arg)


def create_federated_getattr_call(arg: building_blocks.ComputationBuildingBlock,
                                  name: str) -> building_blocks.Call:
  """Creates computation building block passing getattr to federated value.

  Args:
    arg: Instance of `building_blocks.ComputationBuildingBlock` of
      `computation_types.FederatedType` with member of type
      `computation_types.StructType` from which we wish to pick out item `name`.
    name: String name to address the `computation_types.StructType` underlying
      `arg`.

  Returns:
    Returns a `building_blocks.Call` with type signature
    `computation_types.FederatedType` of same placement as `arg`,
    the result of applying or mapping the appropriate `__getattr__` function,
    as defined by `name`.
  """
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(name, str)
  py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(arg.type_signature.member,
                          computation_types.StructType)
  getattr_comp = create_federated_getattr_comp(arg, name)
  return create_federated_map_or_apply(getattr_comp, arg)


def create_federated_getattr_comp(
    comp: building_blocks.ComputationBuildingBlock,
    name: str) -> building_blocks.Lambda:
  """Function to construct computation for `federated_apply` of `__getattr__`.

  Creates a `building_blocks.ComputationBuildingBlock`
  which selects `name` from its argument, of type `comp.type_signature.member`,
  an instance of `computation_types.StructType`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` with type
      signature `computation_types.FederatedType` whose `member` attribute is of
      type `computation_types.StructType`.
    name: String name of attribute to grab.

  Returns:
    Instance of `building_blocks.Lambda` which grabs attribute
      according to `name` of its argument.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.StructType)
  py_typecheck.check_type(name, str)
  element_names = [
      x for x, _ in structure.iter_elements(comp.type_signature.member)
  ]
  if name not in element_names:
    raise ValueError(
        'The federated value has no element of name `{}`. Value: {}'.format(
            name, comp.formatted_representation()))
  apply_input = building_blocks.Reference('x', comp.type_signature.member)
  selected = building_blocks.Selection(apply_input, name=name)
  apply_lambda = building_blocks.Lambda('x', apply_input.type_signature,
                                        selected)
  return apply_lambda


def create_federated_getitem_comp(
    comp: building_blocks.ComputationBuildingBlock,
    key: Union[int, slice]) -> building_blocks.Lambda:
  """Function to construct computation for `federated_apply` of `__getitem__`.

  Creates a `building_blocks.ComputationBuildingBlock`
  which selects `key` from its argument, of type `comp.type_signature.member`,
  of type `computation_types.StructType`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` with type
      signature `computation_types.FederatedType` whose `member` attribute is of
      type `computation_types.StructType`.
    key: Instance of `int` or `slice`, key used to grab elements from the member
      of `comp`. implementation of slicing for `ValueImpl` objects with
      `type_signature` `computation_types.StructType`.

  Returns:
    Instance of `building_blocks.Lambda` which grabs slice
      according to `key` of its argument.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  py_typecheck.check_type(comp.type_signature.member,
                          computation_types.StructType)
  py_typecheck.check_type(key, (int, slice))
  apply_input = building_blocks.Reference('x', comp.type_signature.member)
  if isinstance(key, int):
    selected = building_blocks.Selection(apply_input, index=key)
  else:
    elems = structure.to_elements(comp.type_signature.member)
    index_range = range(*key.indices(len(elems)))
    elem_list = []
    for k in index_range:
      elem_list.append(
          (elems[k][0], building_blocks.Selection(apply_input, index=k)))
    selected = building_blocks.Struct(elem_list)
  apply_lambda = building_blocks.Lambda('x', apply_input.type_signature,
                                        selected)
  return apply_lambda


def create_computation_appending(
    comp1: building_blocks.ComputationBuildingBlock,
    comp2: building_blocks.ComputationBuildingBlock):
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
      of type `computation_type.StructType`.
    comp2: A `building_blocks.ComputationBuildingBlock` or a named computation
      (a tuple pair of name, computation) representing a single element of an
      `structure.Struct`.

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
  comps = building_blocks.Struct((comp1, comp2))
  ref = building_blocks.Reference('comps', comps.type_signature)
  sel_0 = building_blocks.Selection(ref, index=0)
  elements = []
  named_type_signatures = structure.to_elements(comp1.type_signature)
  for index, (name, _) in enumerate(named_type_signatures):
    sel = building_blocks.Selection(sel_0, index=index)
    elements.append((name, sel))
  sel_1 = building_blocks.Selection(ref, index=1)
  elements.append((name2, sel_1))
  result = building_blocks.Struct(elements)
  symbols = ((ref.name, comps),)
  return building_blocks.Block(symbols, result)


def _unname_fn_parameter(fn, unnamed_parameter_type):
  """Coerces `fn` to a comp whose parameter type is `unnamed_parameter_type`."""
  if structure.name_list(fn.type_signature.parameter):
    return building_blocks.Lambda(
        'a', unnamed_parameter_type,
        building_blocks.Call(
            fn,
            building_blocks.Reference('a', unnamed_parameter_type),
        ))
  else:
    return fn


def create_null_federated_aggregate() -> building_blocks.Call:
  unit = building_blocks.Struct([])
  unit_type = unit.type_signature
  value = create_federated_value(unit, placements.CLIENTS)
  zero = unit
  accumulate = create_tensorflow_binary_operator(lambda a, b: a, unit_type)
  merge = accumulate
  report = create_compiled_identity(computation_types.StructType([]))
  return create_federated_aggregate(value, zero, accumulate, merge, report)


def create_federated_aggregate(
    value: building_blocks.ComputationBuildingBlock,
    zero: building_blocks.ComputationBuildingBlock,
    accumulate: building_blocks.ComputationBuildingBlock,
    merge: building_blocks.ComputationBuildingBlock,
    report: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
  # Its okay if the first argument of accumulate is assignable from the zero,
  # without being the exact type. This occurs when accumulate has a type like
  # (<int32[?], int32> -> int32[?]) but zero is int32[0].
  zero_arg_type = accumulate.type_signature.parameter[0]
  zero_arg_type.check_assignable_from(zero.type_signature)
  result_type = computation_types.FederatedType(report.type_signature.result,
                                                placements.SERVER)

  accumulate_parameter_type = computation_types.StructType(
      [zero_arg_type, value.type_signature.member])
  accumulate = _unname_fn_parameter(accumulate, accumulate_parameter_type)
  merge_parameter_type = computation_types.StructType(
      [zero_arg_type, zero_arg_type])
  merge = _unname_fn_parameter(merge, merge_parameter_type)

  intrinsic_type = computation_types.FunctionType((
      type_conversions.type_to_non_all_equal(value.type_signature),
      zero_arg_type,
      accumulate.type_signature,
      merge.type_signature,
      report.type_signature,
  ), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_AGGREGATE.uri,
                                        intrinsic_type)
  values = building_blocks.Struct((value, zero, accumulate, merge, report))
  return building_blocks.Call(intrinsic, values)


def create_federated_apply(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
                                                placements.SERVER)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, arg.type_signature), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_APPLY.uri,
                                        intrinsic_type)
  values = building_blocks.Struct((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_null_federated_broadcast():
  return create_federated_broadcast(
      create_federated_value(building_blocks.Struct([]), placements.SERVER))


def create_federated_broadcast(
    value: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
      value.type_signature.member, placements.CLIENTS, all_equal=True)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_BROADCAST.uri,
                                        intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def create_federated_collect(
    value: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
                                                placements.SERVER)
  intrinsic_type = computation_types.FunctionType(
      type_conversions.type_to_non_all_equal(value.type_signature), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_COLLECT.uri,
                                        intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def create_federated_eval(
    fn: building_blocks.ComputationBuildingBlock,
    placement: placements.PlacementLiteral,
) -> building_blocks.Call:
  r"""Creates a called federated eval.

            Call
           /    \
  Intrinsic      Comp

  Args:
    fn: A `building_blocks.ComputationBuildingBlock` to use as the function.
    placement: A `placements.PlacementLiteral` to use as the placement.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  if placement is placements.CLIENTS:
    uri = intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS.uri
    all_equal = False
  elif placement is placements.SERVER:
    uri = intrinsic_defs.FEDERATED_EVAL_AT_SERVER.uri
    all_equal = True
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))
  result_type = computation_types.FederatedType(
      fn.type_signature.result, placement, all_equal=all_equal)
  intrinsic_type = computation_types.FunctionType(fn.type_signature,
                                                  result_type)
  intrinsic = building_blocks.Intrinsic(uri, intrinsic_type)
  return building_blocks.Call(intrinsic, fn)


def create_null_federated_map() -> building_blocks.Call:
  return create_federated_map(
      create_compiled_identity(computation_types.StructType([])),
      create_federated_value(building_blocks.Struct([]), placements.CLIENTS))


def create_federated_map(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
                                                   placements.CLIENTS)
  result_type = computation_types.FederatedType(fn.type_signature.result,
                                                placements.CLIENTS)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, parameter_type), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_MAP.uri,
                                        intrinsic_type)
  values = building_blocks.Struct((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_federated_map_all_equal(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
  r"""Creates a called federated map of equal values.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  Note: The `fn` is required to be deterministic and therefore should contain no
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
      arg.type_signature.member, placements.CLIENTS, all_equal=True)
  result_type = computation_types.FederatedType(
      fn.type_signature.result, placements.CLIENTS, all_equal=True)
  intrinsic_type = computation_types.FunctionType(
      (fn.type_signature, parameter_type), result_type)
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri, intrinsic_type)
  values = building_blocks.Struct((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_federated_map_or_apply(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
  if arg.type_signature.placement is placements.CLIENTS:
    if arg.type_signature.all_equal:
      return create_federated_map_all_equal(fn, arg)
    else:
      return create_federated_map(fn, arg)
  elif arg.type_signature.placement is placements.SERVER:
    return create_federated_apply(fn, arg)
  else:
    raise TypeError('Unsupported placement {}.'.format(
        arg.type_signature.placement))


def create_federated_mean(
    value: building_blocks.ComputationBuildingBlock,
    weight: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
                                                placements.SERVER)
  if weight is not None:
    intrinsic_type = computation_types.FunctionType(
        (type_conversions.type_to_non_all_equal(value.type_signature),
         type_conversions.type_to_non_all_equal(weight.type_signature)),
        result_type)
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri, intrinsic_type)
    values = building_blocks.Struct((value, weight))
    return building_blocks.Call(intrinsic, values)
  else:
    intrinsic_type = computation_types.FunctionType(
        type_conversions.type_to_non_all_equal(value.type_signature),
        result_type)
    intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_MEAN.uri,
                                          intrinsic_type)
    return building_blocks.Call(intrinsic, value)


def create_null_federated_secure_modular_sum():
  return create_federated_secure_modular_sum(
      create_federated_value(building_blocks.Struct([]), placements.CLIENTS),
      building_blocks.Struct([]))


def create_federated_secure_modular_sum(
    value: building_blocks.ComputationBuildingBlock,
    modulus: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
  r"""Creates a called secure modular sum.

            Call
           /    \
  Intrinsic      [Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    modulus: A `building_blocks.ComputationBuildingBlock` to use as the
      `modulus` value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(modulus, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(value.type_signature.member,
                                                placements.SERVER)
  intrinsic_type = computation_types.FunctionType([
      type_conversions.type_to_non_all_equal(value.type_signature),
      modulus.type_signature,
  ], result_type)
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_SECURE_MODULAR_SUM.uri, intrinsic_type)
  values = building_blocks.Struct([value, modulus])
  return building_blocks.Call(intrinsic, values)


def create_null_federated_secure_sum():
  return create_federated_secure_sum(
      create_federated_value(building_blocks.Struct([]), placements.CLIENTS),
      building_blocks.Struct([]))


def create_federated_secure_sum(
    value: building_blocks.ComputationBuildingBlock,
    max_input: building_blocks.ComputationBuildingBlock
) -> building_blocks.Call:
  r"""Creates a called secure sum.

            Call
           /    \
  Intrinsic      [Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    max_input: A `building_blocks.ComputationBuildingBlock` to use as the
      `max_input` value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(max_input, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(value.type_signature.member,
                                                placements.SERVER)
  intrinsic_type = computation_types.FunctionType([
      type_conversions.type_to_non_all_equal(value.type_signature),
      max_input.type_signature,
  ], result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_SECURE_SUM.uri,
                                        intrinsic_type)
  values = building_blocks.Struct([value, max_input])
  return building_blocks.Call(intrinsic, values)


def create_null_federated_secure_sum_bitwidth():
  return create_federated_secure_sum_bitwidth(
      create_federated_value(building_blocks.Struct([]), placements.CLIENTS),
      building_blocks.Struct([]))


def create_federated_secure_sum_bitwidth(
    value: building_blocks.ComputationBuildingBlock,
    bitwidth: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
  r"""Creates a called secure sum using bitwidth.

            Call
           /    \
  Intrinsic      [Comp, Comp]

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    bitwidth: A `building_blocks.ComputationBuildingBlock` to use as the
      bitwidth value.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(bitwidth, building_blocks.ComputationBuildingBlock)
  result_type = computation_types.FederatedType(value.type_signature.member,
                                                placements.SERVER)
  intrinsic_type = computation_types.FunctionType([
      type_conversions.type_to_non_all_equal(value.type_signature),
      bitwidth.type_signature,
  ], result_type)
  intrinsic = building_blocks.Intrinsic(
      intrinsic_defs.FEDERATED_SECURE_SUM_BITWIDTH.uri, intrinsic_type)
  values = building_blocks.Struct([value, bitwidth])
  return building_blocks.Call(intrinsic, values)


def create_federated_select(
    client_keys,
    max_key,
    server_val,
    select_fn,
    secure: bool,
) -> building_blocks.Call:
  """Creates a called `federated_select` or `federated_secure_select`."""
  py_typecheck.check_type(client_keys, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(max_key, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(server_val, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(select_fn, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(secure, bool)
  single_key_type = max_key.type_signature.member
  select_fn_unnamed_param_type = computation_types.StructType([
      (None, server_val.type_signature.member),
      (None, single_key_type),
  ])
  select_fn = _unname_fn_parameter(select_fn, select_fn_unnamed_param_type)
  result_type = computation_types.at_clients(
      computation_types.SequenceType(select_fn.type_signature.result))
  intrinsic_type = computation_types.FunctionType([
      type_conversions.type_to_non_all_equal(
          client_keys.type_signature), max_key.type_signature,
      server_val.type_signature, select_fn.type_signature
  ], result_type)
  intrinsic_def = intrinsic_defs.FEDERATED_SECURE_SELECT if secure else intrinsic_defs.FEDERATED_SELECT
  intrinsic = building_blocks.Intrinsic(intrinsic_def.uri, intrinsic_type)
  values = building_blocks.Struct([client_keys, max_key, server_val, select_fn])
  return building_blocks.Call(intrinsic, values)


def create_federated_sum(
    value: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
                                                placements.SERVER)
  intrinsic_type = computation_types.FunctionType(
      type_conversions.type_to_non_all_equal(value.type_signature), result_type)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_SUM.uri,
                                        intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def create_federated_unzip(
    value: building_blocks.ComputationBuildingBlock) -> building_blocks.Block:
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
      of type `computation_types.StructType` containing at least one element.

  Returns:
    A `building_blocks.Block`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain any elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  named_type_signatures = structure.to_elements(value.type_signature.member)
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
  result = building_blocks.Struct(elements,
                                  value.type_signature.member.python_container)
  symbols = ((value_ref.name, value),)
  return building_blocks.Block(symbols, result)


def create_federated_value(
    value: building_blocks.ComputationBuildingBlock,
    placement: placements.PlacementLiteral) -> building_blocks.Call:
  r"""Creates a called federated value.

            Call
           /    \
  Intrinsic      Comp

  Args:
    value: A `building_blocks.ComputationBuildingBlock` to use as the value.
    placement: A `placements.PlacementLiteral` to use as the placement.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  if placement is placements.CLIENTS:
    uri = intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri
  elif placement is placements.SERVER:
    uri = intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))
  result_type = computation_types.FederatedType(
      value.type_signature, placement, all_equal=True)
  intrinsic_type = computation_types.FunctionType(value.type_signature,
                                                  result_type)
  intrinsic = building_blocks.Intrinsic(uri, intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def _create_flat_federated_zip(value):
  r"""Private function to create a called federated zip for a non-nested type.

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp, Comp]

  This function returns a federated tuple given a `value` with a tuple of
  federated values type signature.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.StructType` containing at least one element.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain any elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  named_type_signatures = structure.to_elements(value.type_signature)
  container_type = value.type_signature.python_container
  names_to_add = [name for name, _ in named_type_signatures]
  length = len(named_type_signatures)
  if length == 0:
    raise ValueError('federated_zip is only supported on non-empty tuples.')
  first_name, first_type_signature = named_type_signatures[0]
  if first_type_signature.placement == placements.CLIENTS:
    map_fn = create_federated_map
  elif first_type_signature.placement == placements.SERVER:
    map_fn = create_federated_apply
  else:
    raise TypeError('Unsupported placement {}.'.format(
        first_type_signature.placement))
  if length == 1:
    ref = building_blocks.Reference('arg', first_type_signature.member)
    values = building_blocks.Struct(((first_name, ref),), container_type)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, values)
    sel = building_blocks.Selection(value, index=0)
    return map_fn(fn, sel)
  elif length == 2:
    # No point building and tearing down a tree if we can just federated_zip
    # Note: this branch is purely an optimization and is not necessary.
    if any((name is not None for name in names_to_add)):
      # Remove names if necessary
      named_ref = building_blocks.Reference('named', value.type_signature)
      value = building_blocks.Block(
          [(named_ref.name, value)],
          building_blocks.Struct((
              building_blocks.Selection(named_ref, index=0),
              building_blocks.Selection(named_ref, index=1),
          )))
    unnamed_zip = create_zip_two_values(value)
  else:
    # Build a binary tree of federated zips
    args = building_blocks.Reference('value', value.type_signature)
    zipped, paths = _build_tree_of_zips_and_paths_to_elements(
        args, 0,
        len(value.type_signature) - 1)
    zipped_block = building_blocks.Block([(args.name, value)], zipped)
    # Select the values out of the tree back into a flat tuple
    zipped_tree_ref = building_blocks.Reference('zipped_tree',
                                                zipped.type_signature.member)
    flattened_tree = building_blocks.Struct(
        [_selection_from_path(zipped_tree_ref, path) for path in paths])
    flatten_fn = building_blocks.Lambda(zipped_tree_ref.name,
                                        zipped_tree_ref.type_signature,
                                        flattened_tree)
    unnamed_zip = map_fn(flatten_fn, zipped_block)
  return create_named_federated_tuple(unnamed_zip, names_to_add, container_type)


def _prepend_to_paths(paths: List[List[int]], element: int):
  for path in paths:
    path.insert(0, element)


def _build_tree_of_zips_and_paths_to_elements(
    args: building_blocks.Reference,
    start_index: int,
    end_index: int,
) -> Tuple[building_blocks.ComputationBuildingBlock, List[List[int]]]:
  """Builds a binary tree of federated_zips and a list of paths to each element.

  Args:
    args: A reference to the values to be zipped.
    start_index: The index of the first element of `args` to zip.
    end_index: The index of the last element of `args` to zip.

  Returns:
    A tuple containing the tree of zips as well as a list of paths to the
    element at each index. A single path is a list of indices that can be used
    with `_selection_from_path` to select an element out of the result.
  """
  py_typecheck.check_type(args, building_blocks.Reference)
  py_typecheck.check_type(args.type_signature, computation_types.StructType)
  if start_index == end_index:
    # Base case for one element
    tree = building_blocks.Selection(args, index=start_index)
    paths = [[]]
  elif start_index + 1 == end_index:
    # Base case for two elements
    first = building_blocks.Selection(args, index=start_index)
    second = building_blocks.Selection(args, index=end_index)
    values = building_blocks.Struct((first, second))
    tree = create_zip_two_values(values)
    paths = [[0], [1]]
  else:
    # Recursive case for three or more elements
    split_point = int((start_index + end_index) / 2)
    left_tree, left_paths = _build_tree_of_zips_and_paths_to_elements(
        args, start_index, split_point)
    right_tree, right_paths = _build_tree_of_zips_and_paths_to_elements(
        args, split_point + 1, end_index)
    values = building_blocks.Struct((left_tree, right_tree))
    tree = create_zip_two_values(values)
    _prepend_to_paths(left_paths, 0)
    _prepend_to_paths(right_paths, 1)
    paths = left_paths + right_paths
  py_typecheck.check_type(tree, building_blocks.ComputationBuildingBlock)
  assert len(paths) == (end_index - start_index + 1)
  return (tree, paths)


def _selection_from_path(
    selected: building_blocks.ComputationBuildingBlock,
    path: List[int],
) -> building_blocks.ComputationBuildingBlock:
  for path_element in path:
    selected = building_blocks.Selection(selected, index=path_element)
  return selected


def _check_placements(
    placement_values: AbstractSet[placements.PlacementLiteral]):
  """Checks if the placements of the values being zipped are compatible."""
  if not placement_values:
    raise TypeError('federated_zip is only supported on nested structures '
                    'containing at least one FederatedType, but none were '
                    'found.')
  elif len(placement_values) > 1:
    placement_list = ', '.join(placement.name for placement in placement_values)
    raise TypeError('federated_zip requires all nested FederatedTypes to '
                    'have the same placement, but values placed at '
                    f'{placement_list} were found.')


def create_federated_zip(
    value: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
  r"""Creates a called federated zip.

  This function accepts a value whose type signature is a (potentially) nested
  tuple structure of federated values all with the same placement, and uses
  one of the federated_zip intrinsics (at client or at server) to promote the
  placement to the highest level. E.g., A value of type '<A@S, <<B@S>, C@S>>'
  would be mapped to a value of type '<A, <<B>, C>>@S'.

  Args:
    value: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.StructType` that may contain other nested
      `computation_types.StructTypes` bottoming out in at least one element of
      type `computation_Types.FederatedType`. These federated types must be at
      the same placement.

  Returns:
    A `building_blocks.Call` whose type signature is now a federated
      `computation_types.StructType`, placed at the same placement as the
      leaves of `value`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain any elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(value.type_signature, computation_types.StructType)

  # If the type signature is flat, just call _create_flat_federated_zip.
  elements = structure.to_elements(value.type_signature)
  if all(type_sig.is_federated() for (_, type_sig) in elements):
    _check_placements(set(type_sig.placement for (_, type_sig) in elements))
    return _create_flat_federated_zip(value)

  all_placements = set()
  nested_selections = []

  def _make_nested_selections(nested):
    """Generates list of selections from nested representation."""
    if nested.type_signature.is_federated():
      all_placements.add(nested.type_signature.placement)
      nested_selections.append(nested)
    elif nested.type_signature.is_struct():
      for i in range(len(nested.type_signature)):
        inner_selection = building_blocks.Selection(nested, index=i)
        _make_nested_selections(inner_selection)
    else:
      raise TypeError(
          'Expected type signatures consisting of structures of StructType '
          'bottoming out in FederatedType, found: \n{}'.format(
              nested.type_signature))

  _make_nested_selections(value)
  _check_placements(all_placements)

  placement = all_placements.pop()

  flat = building_blocks.Struct(nested_selections)
  flat_zipped = _create_flat_federated_zip(flat)

  # Every building block under the lambda is being constructed below, so it is
  # safe to have a fixed static name for the reference-- we don't need to worry
  # about namespace issues as usual.
  ref = building_blocks.Reference('x', flat_zipped.type_signature.member)

  def _make_flat_selections(type_signature, index):
    """Generates nested struct of selections from flattened representation."""
    if type_signature.is_federated():
      return building_blocks.Selection(ref, index=index), index + 1
    elif type_signature.is_struct():
      elements = structure.to_elements(type_signature)
      return_tuple = []
      for name, element_type in elements:
        selection, index = _make_flat_selections(element_type, index)
        return_tuple.append((name, selection))
      return building_blocks.Struct(return_tuple,
                                    type_signature.python_container), index
    else:
      # This shouldn't be possible since the structure was already traversed
      # above.
      raise TypeError('Only type signatures consisting of structures of '
                      'StructType bottoming out in FederatedType can be '
                      'used in federated_zip.')

  repacked, _ = _make_flat_selections(value.type_signature, 0)
  lam = building_blocks.Lambda('x', ref.type_signature, repacked)

  if placement == placements.CLIENTS:
    return create_federated_map(lam, flat_zipped)
  elif placement == placements.SERVER:
    return create_federated_apply(lam, flat_zipped)
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))


@functools.lru_cache()
def create_generic_constant(
    type_spec: Optional[computation_types.Type],
    scalar_value: Union[int,
                        float]) -> building_blocks.ComputationBuildingBlock:
  """Creates constant for a combination of federated, tuple and tensor types.

  Args:
    type_spec: A `computation_types.Type` containing only federated, tuple or
      tensor types, or `None` to use to construct a generic constant.
    scalar_value: The scalar value we wish this constant to have.

  Returns:
    Instance of `building_blocks.ComputationBuildingBlock`
    representing `scalar_value` packed into `type_spec`.

  Raises:
    TypeError: If types don't match their specification in the args section.
      Notice validation of consistency of `type_spec` with `scalar_value` is not
      the rsponsibility of this function.
  """
  if type_spec is None:
    return create_tensorflow_constant(type_spec, scalar_value)
  py_typecheck.check_type(type_spec, computation_types.Type)
  inferred_scalar_value_type = type_conversions.infer_type(scalar_value)
  if (not inferred_scalar_value_type.is_tensor() or
      inferred_scalar_value_type.shape != tf.TensorShape(())):
    raise TypeError(
        'Must pass a scalar value to `create_generic_constant`; encountered a '
        'value {}'.format(scalar_value))
  if not type_analysis.contains_only(
      type_spec, lambda t: t.is_federated() or t.is_struct() or t.is_tensor()):
    raise TypeError
  if type_analysis.contains_only(type_spec,
                                 lambda t: t.is_struct() or t.is_tensor()):
    return create_tensorflow_constant(type_spec, scalar_value)
  elif type_spec.is_federated():
    unplaced_zero = create_tensorflow_constant(type_spec.member, scalar_value)
    if type_spec.placement == placements.CLIENTS:
      placement_federated_type = computation_types.FederatedType(
          type_spec.member, type_spec.placement, all_equal=True)
      placement_fn_type = computation_types.FunctionType(
          type_spec.member, placement_federated_type)
      placement_function = building_blocks.Intrinsic(
          intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri, placement_fn_type)
    elif type_spec.placement == placements.SERVER:
      placement_federated_type = computation_types.FederatedType(
          type_spec.member, type_spec.placement, all_equal=True)
      placement_fn_type = computation_types.FunctionType(
          type_spec.member, placement_federated_type)
      placement_function = building_blocks.Intrinsic(
          intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri, placement_fn_type)
    return building_blocks.Call(placement_function, unplaced_zero)
  elif type_spec.is_struct():
    elements = []
    for k in range(len(type_spec)):
      elements.append(create_generic_constant(type_spec[k], scalar_value))
    names = [name for name, _ in structure.iter_elements(type_spec)]
    packed_elements = building_blocks.Struct(elements)
    named_tuple = create_named_tuple(packed_elements, names,
                                     type_spec.python_container)
    return named_tuple
  else:
    raise ValueError(
        'The type_spec {} has slipped through all our '
        'generic constant cases, and failed to raise.'.format(type_spec))


def create_zip_two_values(
    value: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
      of type `computation_types.StructType` containing exactly two elements.

  Returns:
    A `building_blocks.Call`.

  Raises:
    TypeError: If any of the types do not match.
    ValueError: If `value` does not contain exactly two elements.
  """
  py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
  named_type_signatures = structure.to_elements(value.type_signature)
  length = len(named_type_signatures)
  if length != 2:
    raise ValueError(
        'Expected a value with exactly two elements, received {} elements.'
        .format(named_type_signatures))
  placement = value.type_signature[0].placement
  if placement is placements.CLIENTS:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri
    all_equal = False
  elif placement is placements.SERVER:
    uri = intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri
    all_equal = True
  else:
    raise TypeError('Unsupported placement {}.'.format(placement))
  elements = []
  for _, type_signature in named_type_signatures:
    federated_type = computation_types.FederatedType(type_signature.member,
                                                     placement, all_equal)
    elements.append((None, federated_type))
  parameter_type = computation_types.StructType(elements)
  result_type = computation_types.FederatedType(
      [(None, e.member) for _, e in named_type_signatures], placement,
      all_equal)
  intrinsic_type = computation_types.FunctionType(parameter_type, result_type)
  intrinsic = building_blocks.Intrinsic(uri, intrinsic_type)
  return building_blocks.Call(intrinsic, value)


def create_sequence_map(
    fn: building_blocks.ComputationBuildingBlock,
    arg: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
  values = building_blocks.Struct((fn, arg))
  return building_blocks.Call(intrinsic, values)


def create_sequence_reduce(
    value: building_blocks.ComputationBuildingBlock,
    zero: building_blocks.ComputationBuildingBlock,
    op: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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
  op_parameter_type = computation_types.StructType(
      [zero.type_signature, value.type_signature.element])
  op = _unname_fn_parameter(op, op_parameter_type)
  intrinsic_type = computation_types.FunctionType((
      value.type_signature,
      zero.type_signature,
      op.type_signature,
  ), op.type_signature.result)
  intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_REDUCE.uri,
                                        intrinsic_type)
  values = building_blocks.Struct((value, zero, op))
  return building_blocks.Call(intrinsic, values)


def create_sequence_sum(
    value: building_blocks.ComputationBuildingBlock) -> building_blocks.Call:
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


def _create_naming_function(tuple_type_to_name, names_to_add, container_type):
  """Private function to construct lambda naming a given tuple type.

  Args:
    tuple_type_to_name: Instance of `computation_types.StructType`, the type of
      the argument which we wish to name.
    names_to_add: Python `list` or `tuple`, the names we wish to give to
      `tuple_type_to_name`.
    container_type: Optional Python container type to associate with the
      resulting tuple.

  Returns:
    An instance of `building_blocks.Lambda` representing a function
    which will take an argument of type `tuple_type_to_name` and return a tuple
    with the same elements, but with names in `names_to_add` attached.

  Raises:
    ValueError: If `tuple_type_to_name` and `names_to_add` have different
    lengths.
  """
  py_typecheck.check_type(tuple_type_to_name, computation_types.StructType)
  if len(names_to_add) != len(tuple_type_to_name):
    raise ValueError(
        'Number of elements in `names_to_add` must match number of element in '
        'the named tuple type `tuple_type_to_name`; here, `names_to_add` has '
        '{} elements and `tuple_type_to_name` has {}.'.format(
            len(names_to_add), len(tuple_type_to_name)))
  naming_lambda_arg = building_blocks.Reference('x', tuple_type_to_name)

  def _create_struct_element(i):
    return (names_to_add[i],
            building_blocks.Selection(naming_lambda_arg, index=i))

  named_result = building_blocks.Struct(
      [_create_struct_element(k) for k in range(len(names_to_add))],
      container_type)
  return building_blocks.Lambda('x', naming_lambda_arg.type_signature,
                                named_result)


def create_named_federated_tuple(
    tuple_to_name: building_blocks.ComputationBuildingBlock,
    names_to_add: Sequence[str],
    container_type=None,
) -> building_blocks.ComputationBuildingBlock:
  """Name tuple elements with names in `names_to_add`.

  Certain intrinsics, e.g. `federated_zip`, only accept unnamed tuples as
  arguments, and can only produce unnamed tuples as their outputs. This is not
  necessarily desirable behavior, as it necessitates dropping any names that
  exist before the zip. This function is intended to provide a general remedy
  for this shortcoming, so that a tuple can be renamed after it is passed
  through any function which drops its names.

  Args:
    tuple_to_name: Instance of `building_blocks.ComputationBuildingBlock` of
      type `computation_types.FederatedType` with `computation_types.StructType`
      member, to populate with names from `names_to_add`.
    names_to_add: Python `tuple` or `list` containing instances of type `str` or
      `None`, the names to give to `tuple_to_name`.
    container_type: An optional Python type to associate with the resulting
      tuple.

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
  if not all((x is None or isinstance(x, str)) for x in names_to_add):
    raise TypeError('`names_to_add` must contain only instances of `str` or '
                    'NoneType; you have passed in {}'.format(names_to_add))
  py_typecheck.check_type(tuple_to_name,
                          building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(tuple_to_name.type_signature,
                          computation_types.FederatedType)
  existing_names = (
      name
      for name, _ in structure.to_elements(tuple_to_name.type_signature.member))
  if (all(
      (existing_name == name_to_add
       for existing_name, name_to_add in zip(existing_names, names_to_add))) and
      container_type is None):
    # The names are already correct, so no work is necessary
    return tuple_to_name

  naming_fn = _create_naming_function(tuple_to_name.type_signature.member,
                                      names_to_add, container_type)
  return create_federated_map_or_apply(naming_fn, tuple_to_name)


def create_named_tuple(
    comp: building_blocks.ComputationBuildingBlock,
    names: Sequence[str],
    container_type=None,
) -> building_blocks.ComputationBuildingBlock:
  """Creates a computation that applies `names` to `comp`.

  Args:
    comp: A `building_blocks.ComputationBuildingBlock` with a `type_signature`
      of type `computation_types.StructType`.
    names: Python `tuple` or `list` containing instances of type `str` or
      `None`, the names to apply to `comp`.
    container_type: Optional Python container type to associated with the
      resulting tuple.

  Returns:
    A `building_blocks.ComputationBuildingBlock` representing a
    tuple with the elements from `comp` and the names from `names` attached to
    the `type_signature` of those elements.

  Raises:
    TypeError: If the types do not match.
  """
  py_typecheck.check_type(names, (list, tuple))
  if not all(isinstance(x, (str, type(None))) for x in names):
    raise TypeError('Expected `names` containing only instances of `str` or '
                    '`None`, found {}'.format(names))
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.StructType)
  fn = _create_naming_function(comp.type_signature, names, container_type)
  return building_blocks.Call(fn, comp)


def create_zip(
    comp: building_blocks.ComputationBuildingBlock) -> building_blocks.Block:
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
  py_typecheck.check_type(comp.type_signature, computation_types.StructType)
  named_type_signatures = structure.to_elements(comp.type_signature)
  _, first_type_signature = named_type_signatures[0]
  py_typecheck.check_type(first_type_signature, computation_types.StructType)
  length = len(first_type_signature)
  for _, type_signature in named_type_signatures:
    py_typecheck.check_type(type_signature, computation_types.StructType)
    if len(type_signature) != length:
      raise TypeError(
          'Expected a StructType containing StructTypes with the same '
          'length, found: {}'.format(comp.type_signature))
  if not comp.is_reference():
    name_generator = unique_name_generator(comp)
    name = next(name_generator)
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
    tup = building_blocks.Struct(columns)
    rows.append(tup)
  tup = building_blocks.Struct(rows)
  if not comp.is_reference():
    return building_blocks.Block(((ref.name, comp),), tup)
  else:
    return tup


def _check_generic_operator_type(type_spec):
  """Checks that `type_spec` can be the signature of args to a generic op."""
  if not type_analysis.contains_only(
      type_spec, lambda t: t.is_federated() or t.is_struct() or t.is_tensor()):
    raise TypeError(
        'Generic operators are only implemented for arguments both containing '
        'only federated, tuple and tensor types; you have passed an argument '
        'of type {} '.format(type_spec))
  if not (type_spec.is_struct() and len(type_spec) == 2):
    raise TypeError(
        'We are trying to construct a generic operator declaring argument that '
        'is not a two-tuple, the type {}.'.format(type_spec))
  if not type_analysis.is_binary_op_with_upcast_compatible_pair(
      type_spec[0], type_spec[1]):
    raise TypeError(
        'The two-tuple you have passed in is incompatible with upcasted '
        'binary operators. You have passed the tuple type {}, which fails the '
        'check that the two members of the tuple are either the same type, or '
        'the second is a scalar with the same dtype as the leaves of the '
        'first. See `type_analysis.is_binary_op_with_upcast_compatible_pair` for '
        'more details.'.format(type_spec))


@functools.lru_cache()
def create_tensorflow_binary_operator_with_upcast(
    operator: Callable[[Any, Any], Any], type_signature: computation_types.Type
) -> building_blocks.CompiledComputation:
  """Creates TF computation upcasting its argument and applying `operator`.

  The concept of upcasting is explained further in the docstring for
  `apply_binary_operator_with_upcast`.

  Args:
    operator: Callable defining the operator.
    type_signature: Value convertible to `computation_types.StructType`, with
      two elements, both of the same type or the second able to be upcast to the
      first, as explained in `apply_binary_operator_with_upcast`, and both
      containing only tuples and tensors in their type tree.

  Returns:
    A `building_blocks.CompiledComputation` encapsulating a function which
    upcasts the second element of its argument and applies the binary
    operator.
  """
  py_typecheck.check_callable(operator)
  _check_generic_operator_type(type_signature)
  type_analysis.check_tensorflow_compatible_type(type_signature)
  tf_proto, type_signature = tensorflow_computation_factory.create_binary_operator_with_upcast(
      type_signature, operator)
  compiled = building_blocks.CompiledComputation(
      tf_proto, type_signature=type_signature)
  return compiled


def apply_binary_operator_with_upcast(
    arg: building_blocks.ComputationBuildingBlock,
    operator: Callable[[Any, Any], Any]) -> building_blocks.Call:
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
    Instance of `building_blocks.Call`
    encapsulating the result of formally applying `operator` to
    `arg[0], `arg[1]`, upcasting `arg[1]` in the condition described above.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_callable(operator)
  if arg.type_signature.is_federated():
    tuple_type = arg.type_signature.member
    assert tuple_type.is_struct()
  elif arg.type_signature.is_struct():
    tuple_type = arg.type_signature
  else:
    raise TypeError(
        'Generic binary operators are only implemented for federated tuple and '
        'unplaced tuples; you have passed {}.'.format(arg.type_signature))

  tf_representing_op = create_tensorflow_binary_operator_with_upcast(
      operator, tuple_type)

  if arg.type_signature.is_federated():
    called = create_federated_map_or_apply(tf_representing_op, arg)
  else:
    called = building_blocks.Call(tf_representing_op, arg)

  return called
