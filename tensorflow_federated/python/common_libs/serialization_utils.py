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
"""Utilities for serializing and deserializing protocol buffers."""

import tensorflow as tf

from google.protobuf import any_pb2
from tensorflow_federated.python.common_libs import py_typecheck

# The default seed below matches the seed in
# `tensorflow/python/framework/random_seed.py`; this is an unexported symbol, so
# replicating here. If this seed is broken, likely something major has changed
# in TensorFlow, so it may be preferable to contact TensorFlow before attempting
# serious debugging.
DEFAULT_GRAPH_SEED = 87654321


def _check_no_graph_level_seed(graph_def):
  for x in graph_def.node:
    seed_attr = x.attr.get('seed')
    seed2_attr = x.attr.get('seed2')
    if seed_attr is not None and not (seed_attr.i == DEFAULT_GRAPH_SEED or
                                      (seed_attr.i == 0 and seed2_attr.i == 0)):
      raise ValueError(
          'TFF disallows the setting of a graph-level random seed. See the '
          'FAQ for more details on reasoning and preferred randomness in TFF.')


def pack_graph_def(graph_def):
  """Pack a `tf.compat.v1.GraphDef` into a proto3 `Any` message.

  Args:
    graph_def: the `tf.compat.v1.GraphDef` to pack into a protocol buffer
      message.

  Returns:
    A `google.protobuf.Any` protocol buffer message.

  Raises:
    TypeError: if `graph_def` is not a `tf.compat.v1.GraphDef`.
  """
  py_typecheck.check_type(graph_def, tf.compat.v1.GraphDef)
  _check_no_graph_level_seed(graph_def)
  any_pb = any_pb2.Any()
  any_pb.Pack(graph_def)
  return any_pb


def unpack_graph_def(any_pb):
  """Unpacks a proto3 `Any` message to a `tf.compat.v1.GraphDef`.

  Args:
    any_pb: the `Any` message to unpack.

  Returns:
    A `tf.compat.v1.GraphDef`.

  Raises:
    ValueError: if the object packed into `any_pb` cannot be unpacked as
      `tf.compat.v1.GraphDef`.
    TypeError: if `any_pb` is not an `Any` protocol buffer message.
  """
  py_typecheck.check_type(any_pb, any_pb2.Any)
  graph_def = tf.compat.v1.GraphDef()
  if not any_pb.Unpack(graph_def):
    raise ValueError(
        'Unable to unpack value [{}] as a tf.compat.v1.GraphDef'.format(any_pb))
  return graph_def
