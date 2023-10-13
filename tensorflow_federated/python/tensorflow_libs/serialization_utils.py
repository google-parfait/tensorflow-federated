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
  """Ensures graph-level random seed hasn't been set.

  Args:
    graph_def: the `tf.compat.v1.GraphDef` to pack into a protocol buffer
      message.

  Raises:
    ValueError: If the graph-level random seed been set.
  """
  for x in graph_def.node:
    seed_attr = x.attr.get('seed')
    seed2_attr = x.attr.get('seed2')
    if (
        seed_attr is not None
        and seed2_attr is not None
        and not (
            seed_attr.i == DEFAULT_GRAPH_SEED
            or (seed_attr.i == 0 and seed2_attr.i == 0)
        )
    ):
      raise ValueError(
          f'Found a graph-level seed on node {x.name} with op {x.op}. '
          'TFF disallows the setting of a graph-level random seed. See the '
          'FAQ for more details on reasoning and preferred randomness in TFF.'
      )


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
  # Perform deterministic Any packing by setting the fields explicitly.
  any_pb.type_url = 'type.googleapis.com/' + graph_def.DESCRIPTOR.full_name
  any_pb.value = graph_def.SerializeToString(deterministic=True)
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
        'Unable to unpack value [{}] as a tf.compat.v1.GraphDef'.format(any_pb)
    )
  return graph_def


# The FunctionDef protocol buffer message type isn't exposed in the `tensorflow`
# package, this is a less than ideal workaround.
FunctionDef = type(tf.compat.v1.GraphDef().library.function.add())


def pack_function_def(function_def: FunctionDef) -> any_pb2.Any:
  """Pack a `FunctionDef` into a proto3 `Any` message.

  Args:
    function_def: the `FunctionDef` to pack into a protocol buffer message.

  Returns:
    A `google.protobuf.Any` protocol buffer message.

  Raises:
    TypeError: if `function_def` is not a `FunctionDef`.
  """
  py_typecheck.check_type(function_def, FunctionDef)
  # Perform deterministic Any packing by setting the fields explicitly and not
  # using the Any.Pack() method.
  return any_pb2.Any(
      type_url='type.googleapis.com/' + function_def.DESCRIPTOR.full_name,
      value=function_def.SerializeToString(deterministic=True),
  )
