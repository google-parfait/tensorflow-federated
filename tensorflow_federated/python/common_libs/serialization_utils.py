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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google.protobuf import any_pb2
from tensorflow_federated.python.common_libs import py_typecheck


def pack_graph_def(graph_def):
  """Pack a `tf.GraphDef` into a proto3 `Any` message.

  Args:
    graph_def: the `tf.GraphDef` to pack into a protocol buffer message.

  Returns:
    A `google.protobuf.Any` protocol buffer message.

  Raises:
    TypeError: if `graph_def` is not a `tf.GraphDef`.
  """
  py_typecheck.check_type(graph_def, tf.GraphDef)
  any_pb = any_pb2.Any()
  any_pb.Pack(graph_def)
  return any_pb


def unpack_graph_def(any_pb):
  """Unpacks a proto3 `Any` message to a `tf.GraphDef`.

  Args:
    any_pb: the `Any` message to unpack.

  Returns:
    A `tf.GraphDef`.

  Raises:
    ValueError: if the object packed into `any_pb` cannot be unpacked as
      `tf.GraphDef`.
    TypeError: if `any_pb` is not an `Any` protocol buffer message.
  """
  py_typecheck.check_type(any_pb, any_pb2.Any)
  graph_def = tf.GraphDef()
  if not any_pb.Unpack(graph_def):
    raise ValueError(
        'Unable to unpack value [{}] as a tf.GraphDef'.format(any_pb))
  return graph_def
