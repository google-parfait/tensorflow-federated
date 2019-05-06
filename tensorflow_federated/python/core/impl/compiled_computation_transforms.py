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

  At the level of a TFF AST, `select_graph_output` transforms the structure
  below:

                                Select(x)
                                   |
                                  Call
                                 /    \
                            Graph      Comp

  into the following:

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
