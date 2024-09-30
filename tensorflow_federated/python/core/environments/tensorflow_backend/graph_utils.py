# Copyright 2022, The TensorFlow Federated Authors.
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
"""Tools for operating on TensorFlow graphs."""


import tensorflow as tf

CONTROL_DEPENDENCY_PREFIX = '^'


def is_control_dependency(input_name: str) -> bool:
  """Returns `True` iff an input to a graph node is a control dependency."""
  return input_name.startswith(CONTROL_DEPENDENCY_PREFIX)


def get_node_name(name: str) -> str:
  """Returns the graph node name from a tensor or node name."""
  if is_control_dependency(name):
    return name[1:]
  return name.split(':', maxsplit=1)[0]


def make_control_dependency(name: str) -> str:
  """Given a tensor or node name, return a control dependency on the node."""
  if is_control_dependency(name):
    return name
  node_name = get_node_name(name)
  return CONTROL_DEPENDENCY_PREFIX + node_name


def add_control_dep_mappings(
    input_map: dict[str, tf.Tensor]
) -> dict[str, tf.Tensor]:
  """Add control dependency mappings for all tensors in an input map.

  Intended to be used for the `input_map` argument of
  `tf.graph_util.import_graph_def` to ensure that any tensors that are remapped
  also have their control dependencies remapped.

  Args:
    input_map: The `input_map` argument that will be passed to
      `tf.graph_util.import_graph_def`.

  Returns:
    A new new `dict` with potentially additional keys for control dependencies.
  """
  return dict(
      **input_map,
      **{
          make_control_dependency(k): make_control_dependency(v.name)
          for k, v in input_map.items()
          if not is_control_dependency(k)
      },
  )
