# Copyright 2018, The TensorFlow Federated Authors.
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
"""Utilities for serializing to/from structures defined in computation.proto."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import types as py_types

# Dependency imports
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import graph_utils


def serialize_type(type_spec):
  """Serializes 'type_spec' as a pb.Type.

  NOTE: Currently only serialization for tensor, named tuple, sequence, and
  function types is implemented.

  Args:
    type_spec: Either an instance of types.Type, or something convertible to it
      by types.to_type(), or None.

  Returns:
    The corresponding instance of pb.Type, or None if the argument was None.

  Raises:
    TypeError: if the argument is of the wrong type.
    NotImplementedError: for type variants for which serialization is not
      implemented.
  """
  # TODO(b/113112885): Implement serialization of the remaining types.
  if type_spec is None:
    return None
  target = types.to_type(type_spec)
  if not isinstance(target, types.Type):
    raise TypeError('Argument {} is not convertible to {}.'.format(
        type(type_spec).__name__, types.Type.__name__))
  if isinstance(target, types.TensorType):
    return pb.Type(tensor=pb.TensorType(
        dtype=target.dtype.as_datatype_enum,
        shape=target.shape.as_proto()))
  elif isinstance(target, types.SequenceType):
    return pb.Type(sequence=pb.SequenceType(
        element=serialize_type(target.element)))
  elif isinstance(target, types.NamedTupleType):
    return pb.Type(tuple=pb.NamedTupleType(element=[
        pb.NamedTupleType.Element(name=e[0], value=serialize_type(e[1]))
        for e in target.elements]))
  elif isinstance(target, types.FunctionType):
    return pb.Type(function=pb.FunctionType(
        parameter=serialize_type(target.parameter),
        result=serialize_type(target.result)))
  else:
    raise NotImplementedError


def deserialize_type(type_proto):
  """Deserializes 'type_proto' as a types.Type.

  NOTE: Currently only deserialization for tensor, named tuple, sequence, and
  function types is implemented.

  Args:
    type_proto: An instance of pb.Type or None.

  Returns:
    The corresponding instance of types.Type (or None if the argument was None).

  Raises:
    TypeError: if the argument is of the wrong type.
    NotImplementedError: for type variants for which deserialization is not
      implemented.
  """
  # TODO(b/113112885): Implement deserialization of the remaining types.
  if type_proto is None:
    return None
  if not isinstance(type_proto, pb.Type):
    raise TypeError('Expected {}, found {}.'.format(
        pb.Type.__name__, type(type_proto).__name__))
  type_variant = type_proto.WhichOneof('type')
  if type_variant is None:
    return None
  elif type_variant == 'tensor':
    return types.TensorType(
        dtype=tf.DType(type_proto.tensor.dtype),
        shape=tf.TensorShape(type_proto.tensor.shape))
  elif type_variant == 'sequence':
    return types.SequenceType(deserialize_type(type_proto.sequence.element))
  elif type_variant == 'tuple':
    return types.NamedTupleType([
        (lambda k, v: (k, v) if k else v)(e.name, deserialize_type(e.value))
        for e in type_proto.tuple.element])
  elif type_variant == 'function':
    return types.FunctionType(
        parameter=deserialize_type(type_proto.function.parameter),
        result=deserialize_type(type_proto.function.result))
  else:
    raise NotImplementedError('Unknown type variant {}.'.format(type_variant))


def serialize_py_func_as_tf_computation(target, parameter_type=None):
  """Serializes the 'target' as a TF computation with a given parameter type.

  Args:
    target: The entity to convert into and serialize as a TF computation. This
      can currently only be a Python function. In the future, we will add here
      support for serializing the various kinds of non-eager and eager defuns,
      and eventually aim at full support for and compliance with TF 2.0.
      This function is currently required to declare either zero parameters if
      parameter_type is None, or exactly one parameter if it's not None. The
      nested structure of this parameter must correspond to the structure of
      the 'parameter_type'. In the future, we may support targets with multiple
      args/keyword args (to be documented in the API and referenced from here).
    parameter_type: The parameter type specification if the target accepts a
      parameter, or None if the target doesn't declare any parameters. Either
      an instance of types.Type, or something that's convertible to it by
      types.to_type().

  Returns:
    The constructed pb.Computation instance with the pb.TensorFlow variant set.

  Raises:
    TypeError: if the arguments are of the wrong types.
    ValueError: if the signature of the target is not compatible with the given
      parameter type.
  """
  # TODO(b/113112108): Support a greater variety of target type signatures,
  # with keyword args or multiple args corresponding to elements of a tuple.
  # Document all accepted forms with examples in the API, and point to there
  # from here.

  if not isinstance(target, py_types.FunctionType):
    raise TypeError('Expected target to be a function type, found {}.'.format(
        type(target).__name__))

  parameter_type = types.to_type(parameter_type)
  argspec = inspect.getargspec(target)

  with tf.Graph().as_default() as graph:
    args = []
    if parameter_type:
      if len(argspec.args) != 1:
        raise ValueError(
            'Expected the target to declare exactly one parameter, '
            'found {}.'.format(repr(argspec.args)))
      parameter_name = argspec.args[0]
      parameter_value, parameter_binding = graph_utils.stamp_parameter_in_graph(
          parameter_name, parameter_type, graph)
      args.append(parameter_value)
    else:
      if argspec.args:
        raise ValueError(
            'Expected the target to declare no parameters, found {}.'.format(
                repr(argspec.args)))
      parameter_binding = None
    result = target(*args)
    result_type, result_binding = graph_utils.capture_result_from_graph(result)

  return pb.Computation(
      type=pb.Type(function=pb.FunctionType(
          parameter=serialize_type(parameter_type),
          result=serialize_type(result_type))),
      tensorflow=pb.TensorFlow(
          graph_def=graph.as_graph_def(),
          parameter=parameter_binding,
          result=result_binding))
