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
"""A library of transformation functions for tensorflow computation."""

import itertools
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
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
    proto: Instance of `pb.Computation` with the `tensorflow` field populated.

  Returns:
    A transformed instance of `pb.Computation` with a `tensorflow` field.
  """
  py_typecheck.check_type(proto, pb.Computation)
  computation_oneof = proto.WhichOneof('computation')
  if computation_oneof != 'tensorflow':
    raise TypeError('`prune_tensorflow_proto` only accepts `Computation` '
                    'protos of the "tensorflow" variety; you have passed '
                    'one of variety {}.'.format(computation_oneof))
  original_tf = proto.tensorflow
  graph_def = serialization_utils.unpack_graph_def(original_tf.graph_def)
  all_nodes = itertools.chain(graph_def.node,
                              *[f.node_def for f in graph_def.library.function])
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
  tf_block = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph_def),
      initialize_op=original_tf.initialize_op
      if original_tf.initialize_op else None,
      parameter=original_tf.parameter
      if original_tf.HasField('parameter') else None,
      result=original_tf.result)
  new_proto = pb.Computation(type=proto.type, tensorflow=tf_block)
  return new_proto
