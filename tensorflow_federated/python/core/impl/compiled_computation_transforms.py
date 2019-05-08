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

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
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
