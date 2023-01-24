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
"""A library of transformations for tensorflow computations."""

import itertools
from typing import Optional

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils

# List of op names that are eligible for Grappler disabling.
CALL_OPS = frozenset(['StatefulPartitionedCall', 'PartitionedCall'])


def disable_grappler_for_partitioned_calls(proto):
  """Disables grappler for `PartitionedCall` and `StatefulPartitionedCall` nodes in the graph.

  TensorFlow serializes a `ConfigProto` into `PartitionedCall` and
  `StatefulPartitionedCall` the `config_proto` `attr` of graph nodes. This
  overrides any session config that might disable runtime grappler. The disable
  grappler for these nodes as well, this function overwrites the serialized
  configproto, setting the `disable_meta_optimizer` field to `True.

  Args:
    proto: Instance of `computation_pb2.Computation` with the `tensorflow` field
      populated.

  Returns:
    A transformed instance of `computation_pb2.Computation` with a `tensorflow`
    field.
  """
  py_typecheck.check_type(proto, computation_pb2.Computation)
  computation_oneof = proto.WhichOneof('computation')
  if computation_oneof != 'tensorflow':
    raise TypeError(
        '`prune_tensorflow_proto` only accepts `Computation` '
        'protos of the "tensorflow" variety; you have passed '
        'one of variety {}.'.format(computation_oneof)
    )
  original_tf = proto.tensorflow
  graph_def = serialization_utils.unpack_graph_def(original_tf.graph_def)
  all_nodes = itertools.chain(
      graph_def.node, *[f.node_def for f in graph_def.library.function]
  )
  for node in all_nodes:
    if node.op not in CALL_OPS:
      continue
    attr_str = node.attr.get('config_proto')
    if attr_str is None:
      config_proto = tf.compat.v1.ConfigProto()
    else:
      config_proto = tf.compat.v1.ConfigProto.FromString(attr_str.s)
    config_proto.graph_options.rewrite_options.disable_meta_optimizer = True
    attr_str.s = config_proto.SerializeToString(deterministic=True)
  tf_block = computation_pb2.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph_def),
      initialize_op=original_tf.initialize_op
      if original_tf.initialize_op
      else None,
      parameter=original_tf.parameter
      if original_tf.HasField('parameter')
      else None,
      result=original_tf.result,
  )
  new_proto = computation_pb2.Computation(type=proto.type, tensorflow=tf_block)
  return new_proto


class DisallowedOpInTensorFlowComputationError(Exception):
  """Error raised when a TensorFlow computation contains a disallowed op."""


def _check_ops(
    proto: computation_pb2.Computation,
    allowed_op_names: Optional[frozenset[str]] = None,
    disallowed_op_names: Optional[frozenset[str]] = None,
):
  """Checks the ops in the TensorFlow computation.

  If allowed_op_names is specified, then _check_ops checks the incoming proto
  contains only ops in the set. On the other hand, if disallowed_op_names is
  specified, then _check_ops checks the proto contains no ops contained in the
  set. One of the two op set arguments must be non-empty, and if both are, then
  allowed_op_names takes precedent.

  Args:
    proto: Instance of `computation_pb2.Computation` with the `tensorflow` field
      populated.
    allowed_op_names: Set of allowed op names.
    disallowed_op_names: Set of disallowed op names.

  Raises:
    DisallowedOpInTensorFlowComputationError: If the computation contains a
      disallowed op.
    RuntimeError: If both allowed_op_names and disallowed_op_names are empty.
  """
  py_typecheck.check_type(proto, computation_pb2.Computation)
  computation_oneof = proto.WhichOneof('computation')
  if computation_oneof != 'tensorflow':
    raise TypeError(
        '`prune_tensorflow_proto` only accepts `Computation` '
        'protos of the "tensorflow" variety; you have passed '
        'one of variety {}.'.format(computation_oneof)
    )
  graph_def = serialization_utils.unpack_graph_def(proto.tensorflow.graph_def)
  all_nodes = itertools.chain(
      graph_def.node, *[f.node_def for f in graph_def.library.function]
  )
  found_disallowed_op_names = set()

  if allowed_op_names:
    for node in all_nodes:
      if node.op not in allowed_op_names:
        found_disallowed_op_names.add(node.op)
  elif disallowed_op_names:
    for node in all_nodes:
      if node.op in disallowed_op_names:
        found_disallowed_op_names.add(node.op)
  else:
    raise RuntimeError(
        'One of allowed_op_names or disallowed_op_names must be non-empty'
    )

  if found_disallowed_op_names:
    found_disallowed_op_names_str = ', '.join(found_disallowed_op_names)
    raise DisallowedOpInTensorFlowComputationError(
        f'Found disallowed ops: {found_disallowed_op_names_str}'
    )


def check_allowed_ops(
    proto: computation_pb2.Computation, allowed_op_names: frozenset[str]
):
  """Checks the TensorFlow computation contains allowed ops.

  Args:
    proto: Instance of `computation_pb2.Computation` with the `tensorflow` field
      populated.
    allowed_op_names: Set of allowed op names.

  Raises:
    DisallowedOpInTensorFlowComputationError: If the computation contains an op
      not in allowed_op_names.
  """
  _check_ops(proto, allowed_op_names=allowed_op_names)


def check_no_disallowed_ops(
    proto: computation_pb2.Computation, disallowed_op_names: frozenset[str]
):
  """Checks the TensorFlow computation for disallowed ops.

  Args:
    proto: Instance of `computation_pb2.Computation` with the `tensorflow` field
      populated.
    disallowed_op_names: Set of disallowed op names.

  Raises:
    DisallowedOpInTensorFlowComputationError: If the computation contains a
      disallowed op.
  """
  _check_ops(proto, disallowed_op_names=disallowed_op_names)
