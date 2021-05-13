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
"""Holds library of transformations for on compiled computations."""

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_transformations
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils
from tensorflow_federated.python.tensorflow_libs import graph_merge
from tensorflow_federated.python.tensorflow_libs import graph_optimizations
from tensorflow_federated.python.tensorflow_libs import graph_spec


def _index_from_name(type_signature, name):
  if name is None:
    raise ValueError
  return structure.name_to_index_map(type_signature)[name]


def select_graph_output(comp, name=None, index=None):
  r"""Makes `CompiledComputation` with same input as `comp` and output `output`.

  Given an instance of `building_blocks.CompiledComputation` `comp`
  with type signature (T -> <U, ...,V>), `select_output` returns a
  `CompiledComputation` representing the logic of calling `comp` and then
  selecting `name` or `index` from the resulting `tuple`. Notice that only one
  of `name` or `index` can be specified, and one of them must be specified.

  At the level of a TFF AST, `select_graph_output` is necessary to transform
  the structure below:

                                Select(x)
                                   |
                                  Call
                                 /    \
                            Graph      Comp

  into:

                                Call
                               /    \
  select_graph_output(Graph, x)      Comp


  Args:
    comp: Instance of `building_blocks.CompiledComputation` which must have
      result type `computation_types.StructType`, the function from which to
      select `output`.
    name: Instance of `str`, the name of the field to select from the output of
      `comp`. Optional, but one of `name` or `index` must be specified.
    index: Instance of `index`, the index of the field to select from the output
      of `comp`. Optional, but one of `name` or `index` must be specified.

  Returns:
    An instance of `building_blocks.CompiledComputation` as
    described, the result of selecting the appropriate output from `comp`.
  """
  py_typecheck.check_type(comp, building_blocks.CompiledComputation)
  if index and name:
    raise ValueError(
        'Please specify at most one of `name` or `index` to `select_outputs`.')
  if index is not None:
    py_typecheck.check_type(index, int)
  elif name is not None:
    py_typecheck.check_type(name, str)
  else:
    raise ValueError('Please pass a `name` or `index` to `select_outputs`.')
  proto = comp.proto
  graph_result_binding = proto.tensorflow.result
  binding_oneof = graph_result_binding.WhichOneof('binding')
  if binding_oneof != 'struct':
    raise TypeError(
        'Can only select output from a CompiledComputation with return type '
        'struct; you have attempted a selection from a CompiledComputation '
        'with return type {}'.format(binding_oneof))
  comp_result_type = comp.type_signature.result
  py_typecheck.check_type(comp_result_type, computation_types.StructType)
  if name is not None:
    index = _index_from_name(comp_result_type, name)
  result = graph_result_binding.struct.element[index]
  result_type = comp_result_type[index]
  type_signature = computation_types.FunctionType(comp.type_signature.parameter,
                                                  result_type)
  selected_proto = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=pb.TensorFlow(
          graph_def=proto.tensorflow.graph_def,
          initialize_op=proto.tensorflow.initialize_op,
          parameter=proto.tensorflow.parameter,
          result=result))
  proto_pruned = tensorflow_computation_transformations.prune_tensorflow_proto(
      selected_proto)
  return building_blocks.CompiledComputation(
      proto_pruned, type_signature=type_signature)


def permute_graph_inputs(comp, input_permutation):
  r"""Remaps input indices of `comp` to match the `input_permutation`.

  Changes the order of the parameters `comp`, an instance of
  `building_blocks.CompiledComputation`. Accepts a permutation
  of the input tuple by index, and applies this permutation to the input
  bindings of `comp`. For example, given a `comp` which accepts a 3-tuple of
  types `[tf.int32, tf.float32, tf.bool]` as its parameter, passing in the
  input permutation

                          [2, 0, 1]

  would change the order of the parameter bindings accepted, so that
  `permute_graph_inputs` returns a
  `building_blocks.CompiledComputation`
  accepting a 3-tuple of types `[tf.bool, tf.int32, tf.float32]`. Notice that
  we use one-line notation for our permutations, with beginning index 0
  (https://en.wikipedia.org/wiki/Permutation#One-line_notation).

  At the AST structural level, this is a no-op, as it simply takes in one
  instance of `building_blocks.CompiledComputation` and returns
  another. However, it is necessary to make a replacement such as transforming:

                          Call
                         /    \
                    Graph      Tuple
                              / ... \
                  Selection(i)       Selection(j)
                       |                  |
                     Comp               Comp

  into:
                                     Call
                                    /    \
  permute_graph_inputs(Graph, [...])      Comp

  Args:
    comp: Instance of `building_blocks.CompiledComputation` whose parameter
      bindings we wish to permute.
    input_permutation: The permutation we wish to apply to the parameter
      bindings of `comp` in 0-indexed one-line permutation notation. This can be
      a Python `list` or `tuple` of `int`s.

  Returns:
    An instance of `building_blocks.CompiledComputation` whose
    parameter bindings represent the same as the result of applying
    `input_permutation` to the parameter bindings of `comp`.

  Raises:
    TypeError: If the types specified in the args section do not match.
  """

  py_typecheck.check_type(comp, building_blocks.CompiledComputation)
  py_typecheck.check_type(input_permutation, (tuple, list))
  permutation_length = len(input_permutation)
  for index in input_permutation:
    py_typecheck.check_type(index, int)
  proto = comp.proto
  graph_parameter_binding = proto.tensorflow.parameter
  py_typecheck.check_type(comp.type_signature.parameter,
                          computation_types.StructType)
  binding_oneof = graph_parameter_binding.WhichOneof('binding')
  if binding_oneof != 'struct':
    raise TypeError(
        'Can only permute inputs of a CompiledComputation with parameter type '
        'struct; you have attempted a permutation with a CompiledComputation '
        'with parameter type {}'.format(binding_oneof))

  original_parameter_type_elements = structure.to_elements(
      comp.type_signature.parameter)
  original_parameter_bindings = [
      x for x in graph_parameter_binding.struct.element
  ]

  def _is_permutation(ls):
    #  Sorting since these shouldn't be long
    return list(sorted(ls)) == list(range(permutation_length))

  if len(original_parameter_bindings
        ) != permutation_length or not _is_permutation(input_permutation):
    raise ValueError(
        'Can only map the inputs with a true permutation; that '
        'is, the position of each input element must be uniquely specified. '
        'You have tried to map inputs {} with permutation {}'.format(
            original_parameter_bindings, input_permutation))

  new_parameter_bindings = [
      original_parameter_bindings[k] for k in input_permutation
  ]
  new_parameter_type_elements = [
      original_parameter_type_elements[k] for k in input_permutation
  ]
  new_parameter_type = computation_types.StructType(new_parameter_type_elements)
  type_signature = computation_types.FunctionType(new_parameter_type,
                                                  comp.type_signature.result)
  permuted_proto = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=pb.TensorFlow(
          graph_def=proto.tensorflow.graph_def,
          initialize_op=proto.tensorflow.initialize_op,
          parameter=pb.TensorFlow.Binding(
              struct=pb.TensorFlow.StructBinding(
                  element=new_parameter_bindings)),
          result=proto.tensorflow.result))
  proto_pruned = tensorflow_computation_transformations.prune_tensorflow_proto(
      permuted_proto)
  return building_blocks.CompiledComputation(
      proto_pruned, type_signature=type_signature)


def bind_graph_parameter_as_tuple(comp, name=None):
  """Wraps the parameter of `comp` in a tuple binding.

  `bind_graph_parameter_as_tuple` is intended as a preprocessing step
  to `pad_graph_inputs_to_match_type`, so that `pad_graph_inputs_to_match_type`
  can
  make the assumption that its argument `comp` always has a tuple binding,
  instead of dealing with the possibility of an unwrapped tensor or sequence
  binding.

  Args:
    comp: Instance of `building_blocks.CompiledComputation` whose parameter we
      wish to wrap in a tuple binding.
    name: Optional string argument, the name to assign to the element type in
      the constructed tuple. Defaults to `None`.

  Returns:
    A transformed version of comp representing exactly the same computation,
    but accepting a tuple containing one element--the parameter of `comp`.

  Raises:
    TypeError: If `comp` is not a
      `building_blocks.CompiledComputation`.
  """
  py_typecheck.check_type(comp, building_blocks.CompiledComputation)
  if name is not None:
    py_typecheck.check_type(name, str)
  proto = comp.proto

  parameter_binding = [proto.tensorflow.parameter]
  parameter_type = computation_types.StructType([
      (name, comp.type_signature.parameter)
  ])
  new_parameter_binding = pb.TensorFlow.Binding(
      struct=pb.TensorFlow.StructBinding(element=parameter_binding))

  new_function_type = computation_types.FunctionType(parameter_type,
                                                     comp.type_signature.result)
  serialized_type = type_serialization.serialize_type(new_function_type)
  input_padded_proto = pb.Computation(
      type=serialized_type,
      tensorflow=pb.TensorFlow(
          graph_def=proto.tensorflow.graph_def,
          initialize_op=proto.tensorflow.initialize_op,
          parameter=new_parameter_binding,
          result=proto.tensorflow.result))
  proto_pruned = tensorflow_computation_transformations.prune_tensorflow_proto(
      input_padded_proto)
  return building_blocks.CompiledComputation(
      proto_pruned, type_signature=new_function_type)


def bind_graph_result_as_tuple(comp, name=None):
  """Wraps the result of `comp` in a tuple binding.

  `bind_graph_result_as_tuple` is used when a
  `building_blocks.Struct` of length 1 containing a called graph is
  encountered; this is an equivalent construct to simply calling the graph
  with the same argument, but wrapping the result in as a tuple. This can
  be accomplished purely by manipulating proto bindings, which is the purpose
  of this function.

  Args:
    comp: Instance of `building_blocks.CompiledComputation` whose parameter we
      wish to wrap in a tuple binding.
    name: Optional string argument, the name to assign to the element type in
      the constructed tuple. Defaults to `None`.

  Returns:
    A transformed version of comp representing exactly the same computation,
    but returning a tuple containing one element--the parameter of `comp`.

  Raises:
    TypeError: If `comp` is not a
      `building_blocks.CompiledComputation`.
  """
  py_typecheck.check_type(comp, building_blocks.CompiledComputation)
  if name is not None:
    py_typecheck.check_type(name, str)
  proto = comp.proto

  result_binding = [proto.tensorflow.result]
  result_type = computation_types.StructType([(name, comp.type_signature.result)
                                             ])
  new_result_binding = pb.TensorFlow.Binding(
      struct=pb.TensorFlow.StructBinding(element=result_binding))

  new_function_type = computation_types.FunctionType(
      comp.type_signature.parameter, result_type)

  result_as_tuple_proto = pb.Computation(
      type=type_serialization.serialize_type(new_function_type),
      tensorflow=pb.TensorFlow(
          graph_def=proto.tensorflow.graph_def,
          initialize_op=proto.tensorflow.initialize_op,
          parameter=proto.tensorflow.parameter,
          result=new_result_binding))
  proto_pruned = tensorflow_computation_transformations.prune_tensorflow_proto(
      result_as_tuple_proto)
  return building_blocks.CompiledComputation(
      proto_pruned, type_signature=new_function_type)


def pad_graph_inputs_to_match_type(comp, type_signature):
  r"""Pads the parameter bindings of `comp` to match `type_signature`.

  The padded parameters here are in effect whimsy bindings--they are not
  plugged in elsewhere in `comp`. This pattern is necessary to transform TFF
  expressions of the form:

                            Lambda(arg)
                                |
                              Call
                             /     \
          CompiledComputation       Tuple
                                      |
                                  Selection[i]
                                      |
                                    Ref(arg)

  into the form:

                          CompiledComputation

  in the case where arg in the above picture represents an n-tuple, where n > 1.

  Notice that some type manipulation must take place to execute the
  transformation outlined above, or anything similar to it, since the Lambda
  we are looking to replace accepts a parameter of an n-tuple, whereas the
  `CompiledComputation` represented above accepts only a 1-tuple.
  `pad_graph_inputs_to_match_type` is intended as an intermediate transform in
  the transformation outlined above, since there may also need to be some
  parameter permutation via `permute_graph_inputs`.

  Notice also that the existing parameter bindings of `comp` must match the
  first elements of `type_signature`. This is to ensure that we are attempting
  to pad only compatible `CompiledComputation`s to a given type signature.

  Args:
    comp: Instance of `building_blocks.CompiledComputation` representing the
      graph whose inputs we want to pad to match `type_signature`.
    type_signature: Instance of `computation_types.StructType` representing the
      type signature we wish to pad `comp` to accept as a parameter.

  Returns:
    A transformed version of `comp`, instance of
    `building_blocks.CompiledComputation` which takes an argument
    of type `type_signature` and executes the same logic as `comp`. In
    particular, this transformed version will have the same return type as
    the original `comp`.

  Raises:
    TypeError: If the proto underlying `comp` has a parameter type which
      is not of `StructType`, the `type_signature` argument is not of type
      `StructType`, or there is a type mismatch between the declared
      parameters of `comp` and the requested `type_signature`.
    ValueError: If the requested `type_signature` is shorter than the
      parameter type signature declared by `comp`.
  """
  py_typecheck.check_type(type_signature, computation_types.StructType)
  py_typecheck.check_type(comp, building_blocks.CompiledComputation)
  proto = comp.proto
  graph_def = proto.tensorflow.graph_def
  graph_parameter_binding = proto.tensorflow.parameter
  binding_oneof = graph_parameter_binding.WhichOneof('binding')
  if binding_oneof != 'struct':
    raise TypeError(
        'Can only pad inputs of a CompiledComputation with parameter type '
        'struct; you have attempted to pad a CompiledComputation '
        'with parameter type {}'.format(binding_oneof))
  # This line provides protection against an improperly serialized proto
  py_typecheck.check_type(comp.type_signature.parameter,
                          computation_types.StructType)
  parameter_bindings = [x for x in graph_parameter_binding.struct.element]
  parameter_type_elements = structure.to_elements(comp.type_signature.parameter)
  type_signature_elements = structure.to_elements(type_signature)
  if len(parameter_bindings) > len(type_signature):
    raise ValueError('We can only pad graph input bindings, never mask them. '
                     'This means that a proposed type signature passed to '
                     '`pad_graph_inputs_to_match_type` must have more elements '
                     'than the existing type signature of the compiled '
                     'computation. You have proposed a type signature of '
                     'length {} be assigned to a computation with parameter '
                     'type signature of length {}.'.format(
                         len(type_signature), len(parameter_bindings)))
  if any(x[1] != type_signature_elements[idx][1]
         for idx, x in enumerate(parameter_type_elements)):
    raise TypeError(
        'The existing elements of the parameter type signature '
        'of the compiled computation in `pad_graph_inputs_to_match_type` '
        'must match the beginning of the proposed new type signature; '
        'you have proposed a parameter type of {} for a computation '
        'with existing parameter type {}.'.format(
            type_signature, comp.type_signature.parameter))
  g = tf.Graph()
  with g.as_default():
    tf.graph_util.import_graph_def(
        serialization_utils.unpack_graph_def(graph_def), name='')

  elems_to_stamp = structure.to_elements(
      type_signature)[len(parameter_bindings):]
  for name, type_spec in elems_to_stamp:
    if name is None:
      stamp_name = 'name'
    else:
      stamp_name = name
    _, stamped_binding = tensorflow_utils.stamp_parameter_in_graph(
        stamp_name, type_spec, g)
    parameter_bindings.append(stamped_binding)
    parameter_type_elements.append((name, type_spec))

  new_parameter_binding = pb.TensorFlow.Binding(
      struct=pb.TensorFlow.StructBinding(element=parameter_bindings))
  new_graph_def = g.as_graph_def()

  new_parameter_type = computation_types.StructType(type_signature_elements)
  new_function_type = computation_types.FunctionType(new_parameter_type,
                                                     comp.type_signature.result)
  serialized_type = type_serialization.serialize_type(new_function_type)

  input_padded_proto = pb.Computation(
      type=serialized_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(new_graph_def),
          initialize_op=proto.tensorflow.initialize_op,
          parameter=new_parameter_binding,
          result=proto.tensorflow.result))
  proto_pruned = tensorflow_computation_transformations.prune_tensorflow_proto(
      input_padded_proto)
  return building_blocks.CompiledComputation(
      proto_pruned, type_signature=new_function_type)


def _unpack_proto_into_graph_spec(tf_block_proto):
  """Packs a TF proto into a `graph_spec.GraphSpec`.

  Args:
    tf_block_proto: Instance of `computation_pb2.Computation` with `tensorflow`
      `computation` attribute.

  Returns:
    Instance of `graph_spec.GraphSpec` containing Python representations of
    the information present in `tf_block_proto`.
  """
  graph = serialization_utils.unpack_graph_def(
      tf_block_proto.tensorflow.graph_def)
  graph_init_op_name = tf_block_proto.tensorflow.initialize_op
  if not graph_init_op_name:
    graph_init_op_name = None
  graph_parameter_binding = tf_block_proto.tensorflow.parameter
  graph_result_binding = tf_block_proto.tensorflow.result

  if graph_parameter_binding.WhichOneof('binding') is not None:
    graph_parameter_list = tensorflow_utils.extract_tensor_names_from_binding(
        graph_parameter_binding)
  else:
    graph_parameter_list = []
  graph_result_list = tensorflow_utils.extract_tensor_names_from_binding(
      graph_result_binding)
  return graph_spec.GraphSpec(graph, graph_init_op_name, graph_parameter_list,
                              graph_result_list)


def _repack_binding_with_new_name(binding, name_map):
  """Constructs new binding via `name_map`.

  Args:
    binding: Instance of `computation_pb2.TensorFlow.Binding`, a binding from an
      old `CompiledComputation` which has perhaps had its tensor and op names
      changed. The handle or tensor names present in `binding` should be the
      full names of these objects in the graph; in particular,
      they should contain the appropriate `:n` suffix.
    name_map: A Python `dict` mapping names from the graph associated to
      `binding` to a new graph with potentially different names, e.g. the one
      constructed by `graph_merge.concatenate_inputs_and_outputs`. The values of
      `name_map` should respect the same semantics as the strings in `binding`;
      that is, they fully specify the tensor they are referring
      to, including the `:n` suffix.

  Returns:
    An instance of `computation_pb2.TensorFlow.Binding` representing the result
    of mapping the names of `binding` according to `name_map`.

  Raises:
    TypeError: If `binding` does represent a
    `computation_pb2.TensorFlow.TensorBinding`,
    `computation_pb2.TensorFlow.StructBinding`, or
    `computation_pb2.TensorFlow.SequenceBinding`.
  """
  if binding.WhichOneof('binding') == 'tensor':
    return pb.TensorFlow.Binding(
        tensor=pb.TensorFlow.TensorBinding(
            tensor_name=name_map[binding.tensor.tensor_name]))
  elif binding.WhichOneof('binding') == 'struct':
    return pb.TensorFlow.Binding(
        struct=pb.TensorFlow.StructBinding(element=[
            _repack_binding_with_new_name(e, name_map)
            for e in binding.struct.element
        ]))
  elif binding.WhichOneof('binding') == 'sequence':
    sequence_oneof = binding.sequence.WhichOneof('binding')
    if sequence_oneof == 'variant_tensor_name':
      return pb.TensorFlow.Binding(
          sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name=name_map[
              binding.sequence.variant_tensor_name]))
    else:
      raise ValueError(
          'Unsupported sequence binding \'{}\'.'.format(sequence_oneof))
  else:
    raise TypeError


def _pack_concatenated_bindings(old_bindings, tensor_name_maps):
  """Maps the old TFF bindings to the correct new names.

  Args:
    old_bindings: Python `list` or `tuple` of instances of
      `computation_pb2.TensorFlow.Binding`, representing the bindings into and
      out of each `tf.Graph` before concatenation.
    tensor_name_maps: Python `list` or `tuple` of `dicts`, mapping tensor or
      string handle names in each `tf.Graph` before concatenation to the new
      names these objects have been given when concatenating graphs.

  Returns:
    A new instance of `computation_pb2.TensorFlow.Binding` if `old_bindings`
    contains any non-`None` bindings, or `None` if `old_bindings` represents
    all `None` bindings, e.g. in the case where all the computations being
    concatenated declare no parameters.

  Raises:
    AssertionError: If `tensor_name_maps` and `old_bindings` have different
    lengths.
  """
  assert len(tensor_name_maps) == len(old_bindings)
  remapped_bindings = []
  for binding, name_map in zip(old_bindings, tensor_name_maps):
    if binding.WhichOneof('binding') is not None:
      remapped_bindings.append(_repack_binding_with_new_name(binding, name_map))
  if not remapped_bindings:
    return None
  if len(remapped_bindings) == 1:
    return remapped_bindings[0]
  return pb.TensorFlow.Binding(
      struct=pb.TensorFlow.StructBinding(element=remapped_bindings))


def _construct_concatenated_type(type_list):
  """Encodes the type convention of `concatenate_tensorflow_blocks`.

  This logic is explained in the docstring of `concatenate_tensorflow_blocks`.

  Args:
    type_list: Python `list` or `tuple` of `computation_types.Type`s, which we
      want to use to construct a new parameter or result type for a computation
      representing concatenation of inputs and outputs of two
      `building_blocks.CompiledComputation`s.

  Returns:
    Instance of `computation_types.Type` representing the appropriate
    parameter or result type, or `None` if `type_list` contains no non-`None`
    types.
  """
  non_none_type_list = [x for x in type_list if x is not None]
  if not non_none_type_list:
    return None
  elif len(non_none_type_list) == 1:
    return non_none_type_list[0]
  return computation_types.StructType(non_none_type_list)


def _called_graph_equality(comp1, comp2):
  """Structural equality function for called TensorFlow graphs."""
  if comp1 is comp2:
    return True
  elif type(comp1.argument) != type(comp2.argument):  # pylint: disable=unidiomatic-typecheck
    return False
  elif comp1.function.is_lambda() or comp2.function.is_lambda():
    return False
  elif not tree_analysis.trees_equal(comp1.function, comp2.function):
    return False
  elif comp1.argument is comp2.argument:
    return True
  return tree_analysis.trees_equal(comp1.argument, comp2.argument)


def concatenate_tensorflow_blocks(tf_comp_list, output_name_list):
  """Concatenates inputs and outputs of its argument to a single TF block.

  Takes a Python `list` or `tuple` of instances of
  `building_blocks.CompiledComputation`, and constructs a single
  instance of the same building block representing the computations present
  in this list concatenated side-by-side.

  There is one important convention here for callers to be aware of.
  `concatenate_tensorflow_blocks` does not perform any more packing into tuples
  than necessary. That is, if `tf_comp_list` contains only a single TF
  computation which declares a parameter, the parameter type of the resulting
  computation is exactly this single parameter type. Since all TF blocks declare
  a result, this is only of concern for parameters, and we will always return a
  function with a tuple for its result value.

  Args:
    tf_comp_list: Python `list` or `tuple` of
      `building_blocks.CompiledComputation`s, whose inputs and outputs we wish
      to concatenate.
    output_name_list: A list list or tuple of names to give to the result types
      in the concatenated TF computations. The elements of this list or tuple
      must be either string types or None

  Returns:
    A single instance of `building_blocks.CompiledComputation`,
    representing all the computations in `tf_comp_list` concatenated
    side-by-side.

  Raises:
    ValueError: If we are passed less than 1 computation in `tf_comp_list`.
      Also raises if `output_name_list` and `tf_comp_list` have different
      lengths.
    TypeError: If `tf_comp_list` is not a `list` or `tuple`, or if it
      contains anything other than TF blocks.
  """
  py_typecheck.check_type(tf_comp_list, (list, tuple))
  py_typecheck.check_type(output_name_list, (list, tuple))
  if len(tf_comp_list) == 1:
    return bind_graph_result_as_tuple(tf_comp_list[0], output_name_list[0])
  elif len(tf_comp_list) < 2:
    raise ValueError('We expect to concatenate at least two blocks of '
                     'TensorFlow; otherwise the transformation you seek '
                     'represents simply type manipulation, and you will find '
                     'your desired function elsewhere in '
                     '`compiled_computation_transforms`. You passed a tuple of '
                     'length {}'.format(len(tf_comp_list)))
  if len(tf_comp_list) != len(output_name_list):
    raise ValueError('`tf_comp_list` and `output_name_list` hav different '
                     'lengths; `concatenate_tensorflow_blocks` must be given '
                     'fully specified output names, even if the names are '
                     '`None`.')
  for name in output_name_list:
    if name is not None:
      py_typecheck.check_type(name, str)
  tf_proto_list = []
  for comp in tf_comp_list:
    py_typecheck.check_type(comp, building_blocks.CompiledComputation)
    tf_proto_list.append(comp.proto)

  (merged_graph, init_op_name, parameter_name_maps,
   result_name_maps) = graph_merge.concatenate_inputs_and_outputs(
       [_unpack_proto_into_graph_spec(x) for x in tf_proto_list])

  concatenated_parameter_bindings = _pack_concatenated_bindings(
      [x.tensorflow.parameter for x in tf_proto_list], parameter_name_maps)
  concatenated_result_bindings = _pack_concatenated_bindings(
      [x.tensorflow.result for x in tf_proto_list], result_name_maps)

  if concatenated_parameter_bindings:
    tf_result_proto = pb.TensorFlow(
        graph_def=serialization_utils.pack_graph_def(
            merged_graph.as_graph_def()),
        initialize_op=init_op_name,
        parameter=concatenated_parameter_bindings,
        result=concatenated_result_bindings)
  else:
    tf_result_proto = pb.TensorFlow(
        graph_def=serialization_utils.pack_graph_def(
            merged_graph.as_graph_def()),
        initialize_op=init_op_name,
        result=concatenated_result_bindings)

  parameter_type = _construct_concatenated_type(
      [x.type_signature.parameter for x in tf_comp_list])
  return_type = computation_types.StructType([
      (output_name_list[i], x.type_signature.result)
      for i, x in enumerate(tf_comp_list)
  ])
  function_type = computation_types.FunctionType(parameter_type, return_type)
  serialized_function_type = type_serialization.serialize_type(function_type)

  constructed_proto = pb.Computation(
      type=serialized_function_type, tensorflow=tf_result_proto)
  proto_pruned = tensorflow_computation_transformations.prune_tensorflow_proto(
      constructed_proto)
  return building_blocks.CompiledComputation(
      proto_pruned, type_signature=function_type)


def optimize_tensorflow_comp(tf_computation, config_proto):
  """Applies configured optimizations to the graphdef backing a TF comp.

  Args:
    tf_computation: Instance of `building_blocks.CompiledComputation` backed by
      TensorFlow.
    config_proto: Instance of `tf.compat.v1.ConfigProto` specifying the
      optimizations to apply to the graph backing this TensorFlow computation.

  Returns:
    A transformed version of `tf_computation`, which has had the
    `tf.compat.v1.GraphDef` backing it run through Grappler with the specified
    configuration.
  """
  py_typecheck.check_type(tf_computation, building_blocks.CompiledComputation)
  tf_proto = tf_computation.proto
  graph_spec_obj = _unpack_proto_into_graph_spec(tf_proto)

  optimized_graph_spec = graph_optimizations.optimize_graph_spec(
      graph_spec_obj, config_proto)
  graph_def = serialization_utils.pack_graph_def(optimized_graph_spec.graph_def)
  original_tf = tf_proto.tensorflow
  tf_result_proto = pb.TensorFlow(
      graph_def=graph_def,
      initialize_op=(original_tf.initialize_op
                     if original_tf.initialize_op else None),
      parameter=(original_tf.parameter
                 if original_tf.HasField('parameter') else None),
      result=original_tf.result)
  optimized_proto = pb.Computation(
      type=tf_proto.type, tensorflow=tf_result_proto)
  return building_blocks.CompiledComputation(
      optimized_proto, type_signature=tf_computation.type_signature)


def compose_tensorflow_blocks(tf_comps):
  """Composes TensorFlow blocks from `tf_comps`.

  Args:
    tf_comps: List or tuple of instances of
      `building_blocks.CompiledComputation` representing the functions we wish
      to compose. Notice that these must obey a certain invariant; the result
      type of computation k in this list must be identical to the parameter type
      of computation k-1. Notice also that the order of this list is quite
      important, as composition is completely noncommutative. This function
      represents the standard mathematical convention for composition; IE,
      compose(f1, f2) represents the function which first calls f2 on its
      argument, then f1 on the result of this call.

  Returns:
    Instance of `building_blocks.CompiledComputation` representing
    the composition of the functions in `tf_comps`.

  Raises:
    TypeError: If `tf_comps` is not a `tuple` or `list`, or if the parameter
      and return types of `tf_comps` do not respect the invariant mentioned
      in the args section of this docstring.
    ValueError: If we are passed a `tf_comps` with fewer than 2 elements;
      the user likely does not want this function in that case.
  """
  py_typecheck.check_type(tf_comps, (list, tuple))
  if len(tf_comps) < 2:
    raise ValueError('Encountered a `tf_comps` of fewer than 2 elements; '
                     'in this case, likely you do not want '
                     '`compose_tensorflow_blocks`.')
  tf_protos = []
  previous_param_type = None
  for comp in tf_comps:
    py_typecheck.check_type(comp, building_blocks.CompiledComputation)
    if previous_param_type is not None:
      if not previous_param_type.is_assignable_from(comp.type_signature.result):
        raise TypeError('The result type of computation k should match the '
                        'parameter type of computation k-1 in `tf_comps`, '
                        'as we are attempting to compose; we have encountered '
                        'a result of type {} attempting to match a parameter '
                        'of type {}'.format(comp.type_signature.result,
                                            previous_param_type))
    previous_param_type = comp.type_signature.parameter
    tf_protos.append(comp.proto)

  (composed_graph, init_op_name, in_name_map,
   out_name_map) = graph_merge.compose_graph_specs(
       [_unpack_proto_into_graph_spec(x) for x in tf_protos])

  last_tf_proto = tf_protos[-1]
  first_tf_proto = tf_protos[0]

  if last_tf_proto.tensorflow.parameter.WhichOneof('binding') is not None:
    parameter_binding = _repack_binding_with_new_name(
        last_tf_proto.tensorflow.parameter, in_name_map)
  else:
    parameter_binding = None

  graph_def = serialization_utils.pack_graph_def(composed_graph.as_graph_def())
  result_binding = _repack_binding_with_new_name(
      first_tf_proto.tensorflow.result, out_name_map)

  if parameter_binding:
    tf_result_proto = pb.TensorFlow(
        graph_def=graph_def,
        initialize_op=init_op_name,
        parameter=parameter_binding,
        result=result_binding)
  else:
    tf_result_proto = pb.TensorFlow(
        graph_def=graph_def, initialize_op=init_op_name, result=result_binding)

  parameter_type = tf_comps[-1].type_signature.parameter
  return_type = tf_comps[0].type_signature.result

  function_type = computation_types.FunctionType(parameter_type, return_type)
  serialized_function_type = type_serialization.serialize_type(function_type)

  constructed_proto = pb.Computation(
      type=serialized_function_type, tensorflow=tf_result_proto)
  proto_pruned = tensorflow_computation_transformations.prune_tensorflow_proto(
      constructed_proto)
  return building_blocks.CompiledComputation(
      proto_pruned, type_signature=function_type)


class LambdaWrappingNoArgGraph(transformation_utils.TransformSpec):
  r"""`TransformSpec` handling throwaway TFF arguments.

  Transforms the pattern:

                    Lambda(x):
                       |
                      Call
                     /    \
  CompiledComputation      `None`

  To simply:

                CompiledComputation

  While preserving semantics.
  """

  def should_transform(self, comp):
    return (comp.is_lambda() and comp.result.is_call() and
            comp.result.function.is_compiled_computation() and
            comp.result.argument is None)

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False

    tf_block = comp.result.function
    arg_type = comp.parameter_type
    return_type = tf_block.type_signature.result
    function_type = computation_types.FunctionType(arg_type, return_type)
    serialized_function_type = type_serialization.serialize_type(function_type)
    tf_spec = _unpack_proto_into_graph_spec(tf_block.proto)
    rebind_name = 'reimported'

    with tf.Graph().as_default() as g:
      tf.import_graph_def(
          tf_spec.graph_def,
          return_elements=tf_spec.out_names,
          name=rebind_name)

    _, param_binding = tensorflow_utils.stamp_parameter_in_graph(
        'unused_param', arg_type, g)
    out_name_map = dict((x, rebind_name + '/' + x) for x in tf_spec.out_names)
    rebound_result = _repack_binding_with_new_name(
        tf_block.proto.tensorflow.result, out_name_map)
    graph_def = serialization_utils.pack_graph_def(g.as_graph_def())

    if tf_spec.init_op is not None:
      init_op = rebind_name + '/' + tf_spec.init_op
    else:
      init_op = None
    tf_result_proto = pb.TensorFlow(
        graph_def=graph_def,
        initialize_op=init_op,
        parameter=param_binding,
        result=rebound_result)

    constructed_proto = pb.Computation(
        type=serialized_function_type, tensorflow=tf_result_proto)
    proto_pruned = tensorflow_computation_transformations.prune_tensorflow_proto(
        constructed_proto)
    return building_blocks.CompiledComputation(
        proto_pruned, type_signature=function_type), True


class CalledCompositionOfTensorFlowBlocks(transformation_utils.TransformSpec):
  """`TransformSpec` representing a composition of TF blocks."""

  def should_transform(self, comp):
    return (comp.is_call() and comp.function.is_compiled_computation() and
            comp.argument is not None and comp.argument.is_call() and
            comp.argument.function.is_compiled_computation())

  def transform(self, comp):
    if self.should_transform(comp):
      bottom_arg = comp.argument.argument
      function_1 = comp.function
      function_2 = comp.argument.function
      composed_fn = compose_tensorflow_blocks([function_1, function_2])
      return building_blocks.Call(composed_fn, bottom_arg), True
    return comp, False


class CalledGraphOnReplicatedArg(transformation_utils.TransformSpec):
  r"""`TransformSpec` representing a called graph with replicated argument.

  Transforms the pattern:

                          Call
                         /    \
      CompiledComputation      <Arg, ..., Arg>

  To

                          Call
                         /    \
      CompiledComputation      Arg

  This is necessary for preserving the invariant that we are always passing
  called graphs up the tree; concatenating a tuple of called graphs necessarily
  calls into this function to preserve this invariant.
  """

  def should_transform(self, comp):
    if not comp.is_call():
      return False
    function = comp.function
    argument = comp.argument
    if not function.is_compiled_computation():
      return False
    if not (argument is not None and argument.is_struct() and
            len(argument) > 0):  # pylint: disable=g-explicit-length-test
      return False
    first_arg = argument[0]
    return all(tree_analysis.trees_equal(x, first_arg) for x in argument[1:])  # pylint: disable=protected-access

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False

    preprocess_arg_comp = building_block_factory.create_compiled_input_replication(
        comp.argument[0].type_signature, len(comp.argument))
    logic_of_tf_comp = comp.function
    composed_tf = compose_tensorflow_blocks(
        [logic_of_tf_comp, preprocess_arg_comp])
    called_tf = building_blocks.Call(composed_tf, comp.argument[0])
    return called_tf, True


class SelectionFromCalledTensorFlowBlock(transformation_utils.TransformSpec):
  r"""`TransformSpec` representing a selection from the result of a TF block.

  That is, parses the pattern:

                            Selection(i)
                                 |
                                Call
                               /    \
            CompiledComputation      Argument

  Into:

                              Call
                             /    \
          CompiledComputation      Argument

  While preserving semantics.
  """

  def should_transform(self, comp):
    return (comp.is_selection() and comp.source.is_call() and
            comp.source.function.is_compiled_computation())

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    selected = select_graph_output(
        comp.source.function, index=comp.index, name=comp.name)
    function_type = computation_types.FunctionType(
        comp.source.function.type_signature.parameter, comp.type_signature)
    pruned = building_blocks.CompiledComputation(
        tensorflow_computation_transformations.prune_tensorflow_proto(
            selected.proto),
        type_signature=function_type)
    return building_blocks.Call(pruned, comp.source.argument), True


def _contains_reference_to(comp, name):
  """Checks that `comp` does not reference `name`."""
  if comp is None:
    return False
  contains_reference = False

  def _transform(inner_comp):
    nonlocal contains_reference
    if inner_comp.is_reference() and inner_comp.name == name:
      contains_reference = True
    return comp, False

  transformation_utils.transform_postorder(comp, _transform)
  return contains_reference


class LambdaWrappingGraph(transformation_utils.TransformSpec):
  r"""`TransformSpec` representing a lambda wrapping a call to a TF graph.

  Transforms the pattern:

                            Lambda(x)
                                |
                              Call
                             /    \
          CompiledComputation      Arg

  (where Arg either is a reference to `x`, or is does not reference `x` at all)

  Notice that the check that `Arg` does not reference `x` will only be complete
  if `Arg` does not rebind `x`. At this point in the compiler pipeline, this
  is a reasonable assumption.

                      CompiledComputation

  While preserving semantics. This represents the final stage of parsing TFF
  into TF.
  """

  def should_transform(self, comp):
    return (
        comp.is_lambda() and comp.result.is_call() and
        comp.result.function.is_compiled_computation() and
        ((comp.result.argument is not None and
          comp.result.argument.is_reference() and
          comp.result.argument.name == comp.parameter_name) or
         (not _contains_reference_to(comp.result.argument, comp.parameter_name)
          and comp.result.function.type_signature == comp.type_signature)))

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    return comp.result.function, True


class StructCalledGraphs(transformation_utils.TransformSpec):
  r"""`TransformSpec` representing a tuple of called TF graphs.

  Transforms the pattern:

                              Tuple--------------------
                             / ...                      \
                         Call                            Call
                        /    \                          /    \
      CompiledComputation      Arg(1)    CompiledComputation  Arg(n)

  Into:

                              Call
                             /    \
          CompiledComputation      Tuple
                                  / ... \
                              Arg(1)    Arg(n)

  While preserving semantics.

  In addition, this transform makes the following performance claim:
  if the arguments to these n `CompiledComputations` are identical (that is,
  if they all pass the `tree_analysis.trees_equal` function pairwise) they
  will be represented as a single tensor in the resulting graph, which is
  plumbed through to the concatenated functions. That is, the structure
  returned by this function will be more accurately represented by:

                              Call
                             /    \
          CompiledComputation      Arg

  where `tree_analysis.trees_equal(Arg, Arg(i))` is `True` for all `i`.

  In particular this implies that, although `StructCalledGraphs` will not check
  to see if the functions it is passed are identical, calling
  `StructCalledGraphs.transform` on distinct called graphs with identical
  arguments will not introduce any unwarranted duplication.
  """

  def __init__(self, only_equal_args=False):
    self._only_equal_args = only_equal_args

  def should_transform(self, comp):
    if not (comp.is_struct() and all(
        (x.is_call() and x.function.is_compiled_computation()) for x in comp)):
      return False
    if not self._only_equal_args:
      return True
    else:
      if len(comp) == 0:  # pylint: disable=g-explicit-length-test
        return False
      arg_generator = (x.argument for x in comp)
      first_arg = next(arg_generator)
      return all(tree_analysis.trees_equal(x, first_arg) for x in arg_generator)

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    if len(comp) == 0:  # pylint: disable=g-explicit-length-test
      # TODO(b/157172423): this destroys the container type.
      return building_block_factory.create_compiled_empty_tuple(), True
    compiled_computation_list = []
    arg_list = []
    name_list = [x[0] for x in structure.iter_elements(comp.type_signature)]
    for k in range(len(comp.type_signature)):
      compiled_computation_list.append(comp[k].function)
      arg_list.append(comp[k].argument)

    concatenated_tf = concatenate_tensorflow_blocks(compiled_computation_list,
                                                    name_list)
    non_none_arg_list = [x for x in arg_list if x is not None]
    if not non_none_arg_list:
      arg = None
    elif len(non_none_arg_list) == 1:
      arg = non_none_arg_list[0]
      return building_blocks.Call(concatenated_tf, arg), True
    else:
      arg = building_blocks.Struct(
          non_none_arg_list,
          container_type=comp.type_signature.python_container)
    called_tf_on_concatenated_arg = building_blocks.Call(concatenated_tf, arg)
    replicated_arg_check = CalledGraphOnReplicatedArg()
    return replicated_arg_check.transform(
        called_tf_on_concatenated_arg)[0], True


def _construct_padding(list_of_indices, tuple_type):
  """Constructs padding for `_remap_graph_inputs`.

  This function is highly coupled with `_construct_permutation` and
  intended to be invoked sequentially with it, as in `_remap_graph_inputs`.
  `_construct_padding` is present in the global scope only to ease its
  testing; one could consider it to be private to `_remap_graph_inputs`.

  Args:
    list_of_indices: Python `list` containing integers between 0 and the length
      of `tuple_type`.
    tuple_type: Instance of `computation_types.StructType` as described in the
      docstring of `_remap_graph_inputs`.

  Returns:
    An instance of `computation_types.StructType` containing
    prefix identical to that constructed from selecting `list_of_indices` in
    order from `tuple_type`, with the remaining elements being the remaining
    elements of `tuple_type` in order.
  """
  type_elements = structure.to_elements(tuple_type)
  existing_type = []
  for i in list_of_indices:
    existing_type.append(type_elements[i])
  type_padding_remaining = [
      x for i, x in enumerate(type_elements) if i not in list_of_indices
  ]
  how_to_pad = computation_types.StructType(existing_type +
                                            type_padding_remaining)
  return how_to_pad


def _construct_permutation(list_of_indices, tuple_type):
  """Constructs permutation for `_remap_graph_inputs`.

  This function is highly coupled with `_construct_padding` and
  intended to be invoked sequentially with it, as in `_remap_graph_inputs`.
  `_construct_permutation` is present in the global scope only to ease its
  testing; one could consider it to be private to `_remap_graph_inputs`.

  Args:
    list_of_indices: Python `list` containing integers between 0 and the length
      of `tuple_type`.
    tuple_type: Instance of `computation_types.StructType` as described in the
      docstring of `_remap_graph_inputs`.

  Returns:
    A permutation of the integers in `range(len(tuple_type))` in
    one-line notation such that applying `how_to_permute` to the type
    `how_to_pad` will result in `tuple_type`.
  """
  type_elements = structure.to_elements(tuple_type)
  length_of_type = len(type_elements)
  index_positions_after_padding = list(range(length_of_type))
  for idx, type_index in enumerate(list_of_indices):
    index_positions_after_padding.pop(
        index_positions_after_padding.index(type_index))
    index_positions_after_padding.insert(idx, type_index)
  how_to_permute = [
      index_positions_after_padding.index(k) for k in range(length_of_type)
  ]
  return how_to_permute


def _remap_graph_inputs(graph, list_of_indices, tuple_type):
  r"""Maps inputs of `graph` via `list_of_indices` to match `type_elements`.

  We may need to add extra inputs to a TF graph along the way of parsing TFF
  into TF. This raises the possibility that we additionally need to permute
  the arguments of the TF graph. For example, consider:

                                      Lambda(arg)
                                          |
                                        Call
                                       /    \
                    CompiledComputation     Selection(5)
                                               |
                                            Ref(arg)

  This compiled computation accepts a single argument, but the Lambda
  we will need to replace accepts a tuple. So we must first pad the
  arguments of the compiled computation, to accept a tuple. However,
  this is not sufficient. The arguments will be passed into the
  compiled computation positionally; since we have simply padded the input
  of the computation, this would result in a potential type/semantic mismatch.
  We must therefore also permute the bindings of the compiled computation so
  that they match the expectation of the lambda.

  This helper function calls two associated helpers to determine the correct
  padding and permutation to apply in order to make these types match up,
  `_construct_padding` and `_construct_permutation`. These associated helpers
  are fairly highly coupled, through their use in this `_remap_graph_inputs`,
  and their outputs respect certain invariants. The return value of
  `_construct_padding` is the type to pad compiled computation with, according
  to the semantics of `pad_graph_inputs_to_match_type`. The return value of
  `_construct_permutation` is the permutation to then apply to this
  padded graph in order to recover `tuple_type`, according to the semantics of
  `permute_graph_inputs`. The guarantees these helper functions make are
  twofold:

  * The type `_construct_padding` returns contains all type elements in
  `tuple_type`, with type prefix the same that would be constructed from
  constructing a tuple type by selecting `list_of_indices` in order from
  `tuple_type`.

  * Applying the permutation `_construct_permutation` returns (in one-line
  notation as noted in `permute_graph_inputs`) to the type returned by
  `_construct_padding` will recover `tuple_type`.

  These invariants are utilized in `_remap_graph_inputs` to construct a
  TensorFlow computation with the same semantics as `graph`, but accepting
  a parameter of type `tuple_type`.

  Args:
    graph: Instance of `building_blocks.CompiledComputation` whose parameter
      type we are trying to match with `tuple_type` if possible.
    list_of_indices: Python `list` containing integers between 0 and the length
      of `tuple_type`.
    tuple_type: Instance of `computation_types.StructType` as described above.

  Returns:
    An instance of `building_blocks.CompiledComputation` which
    contains the same logic as the input `graph`, but accepts an argument of
    type `tuple_type`.

  Raises:
    TypeError: If `tuple_type` is not a `computation_types.StructType` or
    `list_of_indices` is not a `list`.
    ValueError: If `list_of_indices` contains an index less than 0 or out of
    range for the `tuple_type`.
  """
  # TODO(b/133328350): Extend _remap_graph_inputs to allow for multiple reuse of
  # selections.
  py_typecheck.check_type(graph, building_blocks.CompiledComputation)
  py_typecheck.check_type(graph.type_signature.parameter,
                          computation_types.StructType)
  py_typecheck.check_type(tuple_type, computation_types.StructType)
  py_typecheck.check_type(list_of_indices, list)
  tuple_type_len = len(tuple_type)
  if len(set(list_of_indices)) != len(list_of_indices):
    raise ValueError('Support for repeated indices is not yet implemented.')
  if not all(k < tuple_type_len and k >= 0 for k in list_of_indices):
    raise ValueError('list_of_indices must contain only indices between 0 and '
                     'the length of tuple_type; tuple_type here is of length '
                     '{}, and you have asked for the indices {}'.format(
                         tuple_type_len, list_of_indices))
  to_pad = _construct_padding(list_of_indices, tuple_type)
  permutation = _construct_permutation(list_of_indices, tuple_type)
  graph_with_appended_inputs = pad_graph_inputs_to_match_type(graph, to_pad)
  return permute_graph_inputs(graph_with_appended_inputs, permutation)


class TensorFlowOptimizer(transformation_utils.TransformSpec):
  """Applies TF graph optimizations to `building_blocks.CompiledComputation`s.

  This `transformation_utils.TransformSpec` does not alter the TFF structure of
  the computations on which it is called; rather, it calls out to TensorFlow
  libraries which perform optimization on the underlying TensorFlow graph
  representing local processing.
  """

  def __init__(self, config_proto):
    self._config_proto = config_proto

  def should_transform(self, comp):
    return comp.is_compiled_computation()

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    return optimize_tensorflow_comp(comp, self._config_proto), True


class DisableCallOpGrappler(transformation_utils.TransformSpec):
  """Disables grappler in Call ops in `building_blocks.CompiledComputation`s.

  This overwrites the `config_proto` key of the `NodeDef.attr` field of nodes
  in a `tf.compat.v1.GraphDef` to ensure that Grappler is disabled at runtime.

  This `transformation_utils.TransformSpec` does not alter the TFF structure of
  the computations on which it is called.
  """

  def should_transform(self, comp):
    return (comp.is_compiled_computation() and
            comp.proto.WhichOneof('computation') == 'tensorflow')

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    py_typecheck.check_type(comp, building_blocks.CompiledComputation)
    new_comp_proto = tensorflow_computation_transformations.disable_grappler_for_partitioned_calls(
        comp.proto)
    return building_blocks.CompiledComputation(
        new_comp_proto, type_signature=comp.type_signature), True
