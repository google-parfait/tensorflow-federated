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
"""Utilities for deserializing TensorFlow computations.

NOTE: This is separate from `tensorflow_serialization.py` to avoid a circular
dependency through `tf_computation_context.py`. The context code depends on
the deserialization code (to implement invocation), whereas the serialization
code depends on the context code (to invoke the Python function in context).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import six
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb

from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import type_serialization


def deserialize_and_call_tf_computation(computation_proto, arg, graph):
  """Deserializes and invokes a serialized TF computation with a given argument.

  This method performs an action that can be considered roughly the opposite of
  what `serialize_py_func_as_tf_computation` in `tensorflow_serialization` does.
  At the moment, it simply imports the graph in the current context. A future
  implementation may rely on different mechanisms. The caller should not be
  concerned with the specifics of the implementation. At this point, the method
  is expected to only be used within the body of another TF computation (within
  an instance of `TensorFlowComputationContext` at the top of the stack), and
  potentially also in certain types of interpreted execution contexts (TBD).

  Args:
    computation_proto: An instance of `pb.Computation` with the `computation`
      one of equal to `tensorflow` to be deserialized and called.
    arg: The argument to invoke the computation with, or None if the computation
      does not specify a parameter type and does not expects one.
    graph: The graph to stamp into.

  Returns:
    The result of calling the computation in the current context. Depending on
    the type of the result, this can be `tf.Tensor` or `tf.Dataset` instances,
    or a nested structure (such as an `AnonymousTuple`).

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If `computation_proto` is not a TensorFlow computation proto.
  """
  py_typecheck.check_type(computation_proto, pb.Computation)
  computation_oneof = computation_proto.WhichOneof('computation')
  if computation_oneof != 'tensorflow':
    raise ValueError(
        'Expected a TensorFlow computation, got {}.'.format(computation_oneof))
  py_typecheck.check_type(graph, tf.Graph)
  with graph.as_default():
    type_spec = type_serialization.deserialize_type(computation_proto.type)
    if not type_spec.parameter:
      if arg is None:
        input_map = None
      else:
        raise TypeError(
            'The computation declared no parameters; encountered an unexpected '
            'argument {}.'.format(str(arg)))
    elif arg is None:
      raise TypeError(
          'The computation declared a parameter of type {}, but the argument '
          'was not supplied.'.format(str(type_spec.parameter)))
    else:
      arg_type, arg_binding = graph_utils.capture_result_from_graph(arg)
      if not type_spec.parameter.is_assignable_from(arg_type):
        raise TypeError(
            'The computation declared a parameter of type {}, but the argument '
            'is of a mismatching type {}.'.format(
                str(type_spec.parameter), str(arg_type)))
      else:
        input_map = {k: graph.get_tensor_by_name(v) for k, v in six.iteritems(
            graph_utils.compute_map_from_bindings(
                computation_proto.tensorflow.parameter, arg_binding))}
    return_elements = graph_utils.extract_tensor_names_from_binding(
        computation_proto.tensorflow.result)
    output_tensors = tf.import_graph_def(
        computation_proto.tensorflow.graph_def,
        input_map,
        return_elements,
        name='')
    output_map = {k: v for k, v in zip(return_elements, output_tensors)}
    return graph_utils.assemble_result_from_graph(
        type_spec.result, computation_proto.tensorflow.result, output_map)
