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
"""Library of TensorFlow graph-merging functions and associated helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import six

import tensorflow as tf


class GraphSpec(object):
  """Container class for validating input to graph merging functions.

  Mirrors the serialization format of TFF, the main difference being that
  `GraphSpec` takes an already flattened list of names as `in_names` and
  `out_names`, as opposed to the single binding taken by TFF serialization.

  Attributes:
    graph_def: Instance of `tf.GraphDef`.
    init_op: Either a string name of the init op in the graph represented by
      `graph_def`, or `None` if there is no such init op.
    in_names: A Python `list` or `tuple of string names corresponding to the
      input objects in `graph_def`. Must be empty if there are no such inputs.
    out_names: An iterable of string names of the output tensors in `graph_def`,
      subject to the same restrictions as `in_names`.
  """

  def __init__(self, graph_def, init_op, in_names, out_names):
    """Performs typechecking and stashes attributes on constructed instance.

    This constructor performs no additional validation; if, for example, the
    caller passes in a name of a nonexistent op for the `init_op` argument,
    this will not be validated in this class.

    Args:
      graph_def: As defined in the class doc.
      init_op: As defined in the class doc.
      in_names: As defined in the class doc.
      out_names: As defined in the class doc.

    Raises:
      TypeError: If the types of the arguments don't match the type
      specifications in the class doc.
    """
    if not isinstance(graph_def, tf.GraphDef):
      raise TypeError('graph_def must be of type `tf.GraphDef`; you have '
                      'passed a value of type {}'.format(type(graph_def)))
    if not isinstance(init_op, (six.string_types, type(None))):
      raise TypeError('init_op must be string type or `NoneType`; you '
                      'have passed a value of type {}'.format(type(init_op)))
    if not isinstance(in_names, (list, tuple)):
      raise TypeError('`in_names` must be a list or tuple; you have passed '
                      'a value of type {}'.format(type(in_names)))
    for name in in_names:
      if not isinstance(name, six.string_types):
        raise TypeError(
            'Each entry in the `in_names` list must be of string type; you have '
            'passed an in_names list of {}'.format(in_names))
    if not isinstance(out_names, (list, tuple)):
      raise TypeError('`out_names` must be a list or tuple; you have '
                      'passed a value of type {}'.format(out_names))
    for name in out_names:
      if not isinstance(name, six.string_types):
        raise TypeError(
            'Each entry in the `out_names` list must be of string type; '
            'you have passed an in_names list of {}'.format(out_names))
    self.graph_def = graph_def
    self.init_op = init_op
    self.in_names = in_names
    self.out_names = out_names


def _uniquify_shared_names(graph):
  """Appends unique identifier to any shared names present in `graph`."""
  # TODO(b/117428091): Upgrade our TF serialization mechanisms in order to
  # unblock using more modern TF compositional constructs, and avoid direct
  # proto manipulation as is happening here.
  graph_def = graph.as_graph_def()
  for x in graph_def.node:
    if 'shared_name' in x.attr.keys():
      uid = tf.compat.as_bytes(str(uuid.uuid1())[:8])
      x.attr['shared_name'].s += uid
  with tf.Graph().as_default() as new_graph:
    tf.import_graph_def(graph_def, name='')
  return new_graph


def _concat_graphs(graph_def_list, graph_names_list):
  """Imports graphs from `graph_def_list` under the names `graph_names_list`.

  `concat_graphs` is important here to isolate the necessary logic to keep
  barriers between variables defined in separate graphs but with conflicting
  names. In particular, the `shared_names` field can cause these imported
  variables to be wired up incorrectly.

  Args:
    graph_def_list: Python iterable of `tf.GraphDef` objects.
    graph_names_list: Parallel Python iterable containing the names under which
      we wish to import the `tf.GraphDef`s in `graph_def_list`.

  Returns:
    An instance of `tf.Graph`, representing the computations in
    `graph_def_list` side-by-side in a single new `tf.Graph`. The names of ops
    and tensors in this new graph will be the same as those in the old graphs,
    but prepended by the appropriate name in `graph_names_list`.
  """
  merged_graph = tf.Graph()
  for k in range(len(graph_names_list)):
    # merged_graph must be overwritten here with unique shared_names to prevent
    # variables being wired together incorrectly.
    merged_graph = _uniquify_shared_names(merged_graph)
    with merged_graph.as_default():
      tf.import_graph_def(graph_def_list[k], name=graph_names_list[k])
  return merged_graph


def concatenate_inputs_and_outputs(arg_list):
  """Concatenates computations in `arg_list` side-by-side into one `tf.Graph`.

  `concatenate_inputs_and_outputs` is used to combine multiple computations into
  one in the case where none is intended to consume outputs of any other, and we
  simply wish to concatenate the inputs and outputs side-by-side into a single
  `tf.Graph`.

  Args:
    arg_list: Python iterable of `GraphSpec` instances, containing the
      computations we wish to concatenate side-by-side.

  Returns:
    A 4-tuple:
      merged_graph: An instance of `tf.Graph` representing the concatenated
        computations.
      init_op_name: A string representing the op in `merged_graph` that runs
        any initializers passed in with `arg_list`.
      in_name_maps: A Python `list` of `dict`s, representing how names from
        `arg_list` map to names in `merged_graph`. That is, for the `GraphSpec`
        `x` in index `i` of `arg_list`, the `i`th element of `in_name_maps` is
        a dict containing keys the elements of `x.in_names`, and values the
        new names of these elements in `merged_graph`.
      out_name_maps: Similar to `in_name_maps`.

  """
  (graph_def_list, init_op_names_list, in_names_list, out_names_list,
   graph_names_list) = _parse_concatenate_args(arg_list)

  merged_graph = _concat_graphs(graph_def_list, graph_names_list)

  init_op_name = _get_merged_init_op_name(merged_graph, graph_names_list,
                                          init_op_names_list)

  in_name_maps = []
  out_name_maps = []
  for k in range(len(arg_list)):
    in_name_maps.append(
        {x: '{}/{}'.format(graph_names_list[k], x) for x in in_names_list[k]})
    out_name_maps.append(
        {x: '{}/{}'.format(graph_names_list[k], x) for x in out_names_list[k]})

  return merged_graph, init_op_name, in_name_maps, out_name_maps


def _get_merged_init_op_name(merged_graph, graph_names_list,
                             init_op_names_list):
  """Groups init ops and returns name of group."""
  merged_init_op_list = []
  proposed_name = 'merged_init'
  for graph_name, init_op_name in zip(graph_names_list, init_op_names_list):
    if init_op_name is None:
      continue
    else:
      merged_init_op_list.append(
          merged_graph.get_operation_by_name('{}/{}'.format(
              graph_name, init_op_name)))
  with merged_graph.as_default():
    init_op = tf.group(merged_init_op_list, name=proposed_name)
  return init_op.name


def _parse_concatenate_args(arg_list):
  """Flattens list of `GraphSpec` instances."""
  if not all(isinstance(x, GraphSpec) for x in arg_list):
    raise TypeError('Please pass an iterable of `GraphSpec`s to '
                    '`concatenate_inputs_and_outputs`.')
  graph_defs = [x.graph_def for x in arg_list]
  init_op_names = [x.init_op for x in arg_list]
  in_names = [x.in_names for x in arg_list]
  out_names = [x.out_names for x in arg_list]
  graph_names = ['graph_{}'.format(k) for k in range(len(arg_list))]

  return (graph_defs, init_op_names, in_names, out_names, graph_names)
