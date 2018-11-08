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
"""Utilities for serializing to/from TensorFlow computation structure."""

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
from tensorflow_federated.python.core.impl import type_serialization


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
          parameter=type_serialization.serialize_type(parameter_type),
          result=type_serialization.serialize_type(result_type))),
      tensorflow=pb.TensorFlow(
          graph_def=graph.as_graph_def(),
          parameter=parameter_binding,
          result=result_binding))
