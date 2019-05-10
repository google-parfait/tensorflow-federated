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
"""Holds library of transformations for on compiled computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import type_serialization


def select_graph_output(comp, name=None, index=None):
  r"""Makes `CompiledComputation` with same input as `comp` and output `output`.

  Given an instance of `computation_building_blocks.CompiledComputation` `comp`
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
    comp: Instance of `computation_building_blocks.CompiledComputation` which
      must have result type `computation_types.NamedTupleType`, the function
      from which to select `output`.
    name: Instance of `str`, the name of the field to select from the output of
      `comp`. Optional, but one of `name` or `index` must be specified.
    index: Instance of `index`, the index of the field to select from the output
      of `comp`. Optional, but one of `name` or `index` must be specified.

  Returns:
    An instance of `computation_building_blocks.CompiledComputation` as
    described, the result of selecting the appropriate output from `comp`.
  """
  py_typecheck.check_type(comp, computation_building_blocks.CompiledComputation)
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
  if binding_oneof != 'tuple':
    raise TypeError(
        'Can only select output from a CompiledComputation with return type '
        'tuple; you have attempted a selection from a CompiledComputation '
        'with return type {}'.format(binding_oneof))
  proto_type = type_serialization.deserialize_type(proto.type)
  py_typecheck.check_type(proto_type.result, computation_types.NamedTupleType)
  if name is None:
    result = [x for x in graph_result_binding.tuple.element][index]
    result_type = proto_type.result[index]
  else:
    type_names_list = [
        x[0] for x in anonymous_tuple.to_elements(proto_type.result)
    ]
    index = type_names_list.index(name)
    result = [x for x in graph_result_binding.tuple.element][index]
    result_type = proto_type.result[index]
  serialized_type = type_serialization.serialize_type(
      computation_types.FunctionType(proto_type.parameter, result_type))
  selected_proto = pb.Computation(
      type=serialized_type,
      tensorflow=pb.TensorFlow(
          graph_def=proto.tensorflow.graph_def,
          initialize_op=proto.tensorflow.initialize_op,
          parameter=proto.tensorflow.parameter,
          result=result))
  return computation_building_blocks.CompiledComputation(selected_proto)


def permute_graph_inputs(comp, input_permutation):
  r"""Remaps input indices of `comp` to match the `input_permutation`.

  Changes the order of the parameters `comp`, an instance of
  `computation_building_blocks.CompiledComputation`. Accepts a permutation
  of the input tuple by index, and applies this permutation to the input
  bindings of `comp`. For example, given a `comp` which accepts a 3-tuple of
  types `[tf.int32, tf.float32, tf.bool]` as its parameter, passing in the
  input permutation

                          [2, 0, 1]

  would change the order of the parameter bindings accepted, so that
  `permute_graph_inputs` returns a
  `computation_building_blocks.CompiledComputation`
  accepting a 3-tuple of types `[tf.bool, tf.int32, tf.float32]`. Notice that
  we use one-line notation for our permutations, with beginning index 0
  (https://en.wikipedia.org/wiki/Permutation#One-line_notation).

  At the AST structural level, this is a no-op, as it simply takes in one
  instance of `computation_building_blocks.CompiledComputation` and returns
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
    comp: Instance of `computation_building_blocks.CompiledComputation` whose
      parameter bindings we wish to permute.
    input_permutation: The permutation we wish to apply to the parameter
      bindings of `comp` in 0-indexed one-line permutation notation. This can be
      a Python `list` or `tuple` of `int`s.

  Returns:
    An instance of `computation_building_blocks.CompiledComputation` whose
    parameter bindings represent the same as the result of applying
    `input_permutation` to the parameter bindings of `comp`.

  Raises:
    TypeError: If the types specified in the args section do not match.
  """

  py_typecheck.check_type(comp, computation_building_blocks.CompiledComputation)
  py_typecheck.check_type(input_permutation, (tuple, list))
  permutation_length = len(input_permutation)
  for index in input_permutation:
    py_typecheck.check_type(index, int)
  proto = comp.proto
  graph_parameter_binding = proto.tensorflow.parameter
  proto_type = type_serialization.deserialize_type(proto.type)
  py_typecheck.check_type(proto_type.parameter,
                          computation_types.NamedTupleType)
  binding_oneof = graph_parameter_binding.WhichOneof('binding')
  if binding_oneof != 'tuple':
    raise TypeError(
        'Can only permute inputs of a CompiledComputation with parameter type '
        'tuple; you have attempted a permutation with a CompiledComputation '
        'with parameter type {}'.format(binding_oneof))

  original_parameter_type_elements = anonymous_tuple.to_elements(
      proto_type.parameter)
  original_parameter_bindings = [
      x for x in graph_parameter_binding.tuple.element
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

  serialized_type = type_serialization.serialize_type(
      computation_types.FunctionType(new_parameter_type_elements,
                                     proto_type.result))
  permuted_proto = pb.Computation(
      type=serialized_type,
      tensorflow=pb.TensorFlow(
          graph_def=proto.tensorflow.graph_def,
          initialize_op=proto.tensorflow.initialize_op,
          parameter=pb.TensorFlow.Binding(
              tuple=pb.TensorFlow.NamedTupleBinding(
                  element=new_parameter_bindings)),
          result=proto.tensorflow.result))
  return computation_building_blocks.CompiledComputation(permuted_proto)


def wraph_graph_parameter_as_tuple(comp):
  """Wraps the parameter of `comp` in a tuple binding.

  `wraph_graph_parameter_as_tuple` is intended as a preprocessing step
  to `pad_graph_inputs_to_match_type`, so that `pad_graph_inputs_to_match_type`
  can
  make the assumption that its argument `comp` always has a tuple binding,
  instead of dealing with the possibility of an unwrapped tensor or sequence
  binding.

  Args:
    comp: Instance of `computation_building_blocks.CompiledComputation` whose
      parameter we wish to wrap in a tuple binding.

  Returns:
    A transformed version of comp representing exactly the same computation,
    but accepting a tuple containing one element--the parameter of `comp`.

  Raises:
    TypeError: If `comp` is not a
      `computation_building_blocks.CompiledComputation`.
  """
  py_typecheck.check_type(comp, computation_building_blocks.CompiledComputation)
  proto = comp.proto
  proto_type = type_serialization.deserialize_type(proto.type)

  parameter_binding = [proto.tensorflow.parameter]
  parameter_type_list = [proto_type.parameter]
  new_parameter_binding = pb.TensorFlow.Binding(
      tuple=pb.TensorFlow.NamedTupleBinding(element=parameter_binding))

  new_function_type = computation_types.FunctionType(parameter_type_list,
                                                     proto_type.result)
  serialized_type = type_serialization.serialize_type(new_function_type)

  input_padded_proto = pb.Computation(
      type=serialized_type,
      tensorflow=pb.TensorFlow(
          graph_def=proto.tensorflow.graph_def,
          initialize_op=proto.tensorflow.initialize_op,
          parameter=new_parameter_binding,
          result=proto.tensorflow.result))

  return computation_building_blocks.CompiledComputation(input_padded_proto)


def pad_graph_inputs_to_match_type(comp, type_signature):
  r"""Pads the parameter bindings of `comp` to match `type_signature`.

  The padded parameters here are in effect dummy bindings--they are not
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
    comp: Instance of `computation_building_blocks.CompiledComputation`
      representing the graph whose inputs we want to pad to match
      `type_signature`.
    type_signature: Instance of `computation_types.NamedTupleType` representing
      the type signature we wish to pad `comp` to accept as a parameter.

  Returns:
    A transformed version of `comp`, instance of
    `computation_building_blocks.CompiledComputation` which takes an argument
    of type `type_signature` and executes the same logic as `comp`. In
    particular, this transformed version will have the same return type as
    the original `comp`.

  Raises:
    TypeError: If the proto underlying `comp` has a parameter type which
      is not of `NamedTupleType`, the `type_signature` argument is not of type
      `NamedTupleType`, or there is a type mismatch between the declared
      parameters of `comp` and the requested `type_signature`.
    ValueError: If the requested `type_signature` is shorter than the
      parameter type signature declared by `comp`.
  """
  py_typecheck.check_type(type_signature, computation_types.NamedTupleType)
  py_typecheck.check_type(comp, computation_building_blocks.CompiledComputation)
  proto = comp.proto
  graph_def = proto.tensorflow.graph_def
  graph_parameter_binding = proto.tensorflow.parameter
  proto_type = type_serialization.deserialize_type(proto.type)
  binding_oneof = graph_parameter_binding.WhichOneof('binding')
  if binding_oneof != 'tuple':
    raise TypeError(
        'Can only pad inputs of a CompiledComputation with parameter type '
        'tuple; you have attempted to pad a CompiledComputation '
        'with parameter type {}'.format(binding_oneof))
  # This line provides protection against an improperly serialized proto
  py_typecheck.check_type(proto_type.parameter,
                          computation_types.NamedTupleType)
  parameter_bindings = [x for x in graph_parameter_binding.tuple.element]
  parameter_type_elements = anonymous_tuple.to_elements(proto_type.parameter)
  type_signature_elements = anonymous_tuple.to_elements(type_signature)
  if len(parameter_bindings) > len(type_signature):
    raise ValueError('We can only pad graph input bindings, never mask them. '
                     'This means that a proposed type signature passed to '
                     '`pad_graph_inputs_to_match_type` must have more elements '
                     'than the existing type signature of the compiled '
                     'computation. You have proposed a type signature of '
                     'length {} be assigned to a computation with parameter '
                     'type signature of length {}.'.format(
                         len(type_signature), len(parameter_bindings)))
  if any(x != type_signature_elements[idx]
         for idx, x in enumerate(parameter_type_elements)):
    raise TypeError(
        'The existing elements of the parameter type signature '
        'of the compiled computation in `pad_graph_inputs_to_match_type` '
        'must match the beginning of the proposed new type signature; '
        'you have proposed a parameter type of {} for a computation '
        'with existing parameter type {}.'.format(type_signature,
                                                  proto_type.parameter))
  g = tf.Graph()
  with g.as_default():
    tf.graph_util.import_graph_def(
        serialization_utils.unpack_graph_def(graph_def), name='')

  elems_to_stamp = anonymous_tuple.to_elements(
      type_signature)[len(parameter_bindings):]
  for name, type_spec in elems_to_stamp:
    if name is None:
      stamp_name = 'name'
    else:
      stamp_name = name
    _, stamped_binding = graph_utils.stamp_parameter_in_graph(
        stamp_name, type_spec, g)
    parameter_bindings.append(stamped_binding)
    parameter_type_elements.append((name, type_spec))

  new_parameter_binding = pb.TensorFlow.Binding(
      tuple=pb.TensorFlow.NamedTupleBinding(element=parameter_bindings))
  new_graph_def = g.as_graph_def()

  new_function_type = computation_types.FunctionType(parameter_type_elements,
                                                     proto_type.result)
  serialized_type = type_serialization.serialize_type(new_function_type)

  input_padded_proto = pb.Computation(
      type=serialized_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(new_graph_def),
          initialize_op=proto.tensorflow.initialize_op,
          parameter=new_parameter_binding,
          result=proto.tensorflow.result))

  return computation_building_blocks.CompiledComputation(input_padded_proto)
