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
"""Utilities for testing tensorflow computations."""

from typing import Optional

import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def _stamp_value_into_graph(
    value: Optional[object],
    type_signature: computation_types.Type,
    graph: tf.Graph,
) -> object:
  """Stamps `value` in `graph` as an object of type `type_signature`.

  Args:
    value: A value to stamp.
    type_signature: The type of the value to stamp.
    graph: The graph to stamp in.

  Returns:
    A Python object made of tensors stamped into `graph`, `tf.data.Dataset`s,
    or `structure.Struct`s that structurally corresponds to the value passed at
    input.
  """
  if value is None:
    return None
  if isinstance(type_signature, computation_types.TensorType):
    if isinstance(value, np.ndarray) or tf.is_tensor(value):
      value_type = computation_types.TensorType(value.dtype, value.shape)
      type_signature.check_assignable_from(value_type)
      with graph.as_default():
        return tf.constant(value)
    else:
      with graph.as_default():
        return tf.constant(
            value,
            dtype=type_signature.dtype,  # pytype: disable=attribute-error
            shape=type_signature.shape,  # pytype: disable=attribute-error
        )
  elif isinstance(type_signature, computation_types.StructType):
    if isinstance(value, (list, dict)):
      value = structure.from_container(value)
    stamped_elements = []
    named_type_signatures = structure.to_elements(type_signature)
    for (name, type_signature), element in zip(named_type_signatures, value):
      stamped_element = _stamp_value_into_graph(element, type_signature, graph)
      stamped_elements.append((name, stamped_element))
    return structure.Struct(stamped_elements)
  elif isinstance(type_signature, computation_types.SequenceType):
    return tensorflow_utils.make_data_set_from_elements(
        graph, value, type_signature.element
    )
  else:
    raise NotImplementedError(
        'Unable to stamp a value of type {} in graph.'.format(type_signature)
    )


def run_tensorflow(
    computation_proto: pb.Computation, arg: Optional[object] = None
) -> object:
  """Runs a TensorFlow computation with argument `arg`.

  Args:
    computation_proto: An instance of `pb.Computation`.
    arg: The argument to invoke the computation with, or None if the computation
      does not specify a parameter type and does not expects one.

  Returns:
    The result of the computation.
  """
  with tf.Graph().as_default() as graph:
    type_signature = type_serialization.deserialize_type(computation_proto.type)
    if type_signature.parameter is not None:  # pytype: disable=attribute-error
      stamped_arg = _stamp_value_into_graph(
          arg,
          type_signature.parameter,  # pytype: disable=attribute-error
          graph,
      )
    else:
      stamped_arg = None
    init_op, result = tensorflow_utils.deserialize_and_call_tf_computation(
        computation_proto, stamped_arg, graph, '', tf.constant('bogus_token')
    )
  with tf.compat.v1.Session(graph=graph) as sess:
    if init_op:
      sess.run(init_op)
    result = tensorflow_utils.fetch_value_in_session(sess, result)
  return result
