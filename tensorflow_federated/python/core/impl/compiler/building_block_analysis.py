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
"""Utils for TFF computation building blocks."""

import six

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks


def is_called_intrinsic(comp, uri=None):
  """Tests if `comp` is a called intrinsic with the given `uri`.

  Args:
    comp: The computation building block to test.
    uri: A uri or a collection of uris; the same as what is accepted by
      isinstance.

  Returns:
    `True` if `comp` is a called intrinsic with the given `uri`, otherwise
    `False`.
  """
  if isinstance(uri, six.string_types):
    uri = [uri]
  return (isinstance(comp, building_blocks.Call) and
          isinstance(comp.function, building_blocks.Intrinsic) and
          (uri is None or comp.function.uri in uri))


def is_identity_function(comp):
  """Returns `True` if `comp` is an identity function, otherwise `False`."""
  return (isinstance(comp, building_blocks.Lambda) and
          isinstance(comp.result, building_blocks.Reference) and
          comp.parameter_name == comp.result.name)


def count_tensorflow_ops_in(comp):
  """Counts TF ops in `comp` if `comp` is a TF block."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  if (not isinstance(comp, building_blocks.CompiledComputation)) or (
      comp.proto.WhichOneof('computation') != 'tensorflow'):
    raise ValueError('Please pass a '
                     '`building_blocks.CompiledComputation` of the '
                     '`tensorflow` variety to `count_tensorflow_ops_in`.')
  graph_def = serialization_utils.unpack_graph_def(
      comp.proto.tensorflow.graph_def)
  return len(graph_def.node)


def count_tensorflow_variables_in(comp):
  """Counts TF Variables in `comp` if `comp` is a TF block."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  if (not isinstance(comp, building_blocks.CompiledComputation)) or (
      comp.proto.WhichOneof('computation') != 'tensorflow'):
    raise ValueError('Please pass a '
                     '`building_blocks.CompiledComputation` of the '
                     '`tensorflow` variety to `count_tensorflow_variables_in`.')
  graph_def = serialization_utils.unpack_graph_def(
      comp.proto.tensorflow.graph_def)

  def _node_is_variable(node):
    # TODO(b/137887596): Follow up on ways to count Variables on the GraphDef
    # level.
    return (str(node.op).lower().startswith('variable') or
            str(node.op).lower() == 'varhandleop')

  return len([x for x in graph_def.node if _node_is_variable(x)])
