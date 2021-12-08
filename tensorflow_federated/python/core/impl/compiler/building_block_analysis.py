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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A library of static analysis functions for building blocks."""

import collections

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks


def is_called_intrinsic(comp, uri=None):
  """Tests if `comp` is a called intrinsic with the given `uri`.

  Args:
    comp: The computation building block to test.
    uri: An optional URI or list of URIs; the same form as what is accepted by
      isinstance.

  Returns:
    `True` if `comp` is a called intrinsic with the given `uri`, otherwise
    `False`.
  """
  if isinstance(uri, str):
    uri = [uri]
  return (comp.is_call() and comp.function.is_intrinsic() and
          (uri is None or comp.function.uri in uri))


def is_identity_function(comp):
  """Returns `True` if `comp` is an identity function, otherwise `False`."""
  return (comp.is_lambda() and comp.result.is_reference() and
          comp.parameter_name == comp.result.name)


def count_tensorflow_ops_in(comp):
  """Counts TF ops in `comp` if `comp` is a TF block."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  if (not comp.is_compiled_computation()) or (
      comp.proto.WhichOneof('computation') != 'tensorflow'):
    raise ValueError('Please pass a '
                     '`building_blocks.CompiledComputation` of the '
                     '`tensorflow` variety to `count_tensorflow_ops_in`.')
  graph_def = serialization_utils.unpack_graph_def(
      comp.proto.tensorflow.graph_def)
  return len(graph_def.node) + sum(
      [len(graph_func.node_def) for graph_func in graph_def.library.function])


def count_tensorflow_variables_in(comp):
  """Counts TF Variables in `comp` if `comp` is a TF block."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  if (not comp.is_compiled_computation()) or (
      comp.proto.WhichOneof('computation') != 'tensorflow'):
    raise ValueError('Please pass a '
                     '`building_blocks.CompiledComputation` of the '
                     '`tensorflow` variety to `count_tensorflow_variables_in`.')
  graph_def = serialization_utils.unpack_graph_def(
      comp.proto.tensorflow.graph_def)

  def _node_is_variable(node):
    # TODO(b/137887596): Follow up on ways to count Variables on the GraphDef
    # level.
    op_name = str(node.op).lower()
    return ((op_name.startswith('variable') and
             op_name not in ['variableshape']) or op_name == 'varhandleop')

  def _count_vars_in_function_lib(func_library):
    total_nodes = 0
    for graph_func in func_library.function:
      total_nodes += sum(
          _node_is_variable(node) for node in graph_func.node_def)
    return total_nodes

  return (sum(_node_is_variable(node) for node in graph_def.node) +
          _count_vars_in_function_lib(graph_def.library))


def get_device_placement_in(comp):
  """Gets counter of device placement for tensorflow compuation `comp`."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  if (not comp.is_compiled_computation()) or (
      comp.proto.WhichOneof('computation') != 'tensorflow'):
    raise ValueError('Please pass a '
                     '`building_blocks.CompiledComputation` of the '
                     '`tensorflow` variety to `get_device_placement_in`. (Got '
                     'a [{t}]).'.format(t=type(comp)))
  graph_def = serialization_utils.unpack_graph_def(
      comp.proto.tensorflow.graph_def)

  counter = collections.Counter()

  def _populate_counter_in_function_lib(func_library):
    for graph_func in func_library.function:
      counter.update(node.device for node in graph_func.node_def)
    for graph_func in func_library.gradient:
      counter.update(node.device for node in graph_func.node_def)

  counter.update(node.device for node in graph_def.node)
  _populate_counter_in_function_lib(graph_def.library)

  return counter
