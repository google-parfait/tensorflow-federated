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

Note: This is separate from `tensorflow_serialization.py` to avoid a circular
dependency through `tensorflow_computation_context.py`. The context code depends
on the deserialization code (to implement invocation), whereas the serialization
code depends on the context code (to invoke the Python function in context).
"""

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def deserialize_and_call_tf_computation(computation_proto, arg, graph):
  """Deserializes a TF computation and inserts it into `graph`.

  This method performs an action that can be considered roughly the opposite of
  what `tensorflow_serialization.tf_computation_serializer` does. At
  the moment, it simply imports the graph in the current context. A future
  implementation may rely on different mechanisms. The caller should not be
  concerned with the specifics of the implementation. At this point, the method
  is expected to only be used within the body of another TF computation (within
  an instance of `tensorflow_computation_context.TensorFlowComputationContext`
  at the top of the stack), and potentially also in certain types of interpreted
  execution contexts (TBD).

  Args:
    computation_proto: An instance of `pb.Computation` with the `computation`
      one of equal to `tensorflow` to be deserialized and called.
    arg: The argument to invoke the computation with, or None if the computation
      does not specify a parameter type and does not expects one.
    graph: The graph to stamp into.

  Returns:
    A tuple (init_op, result) where:
       init_op:  String name of an op to initialize the graph.
       result: The results to be fetched from TensorFlow. Depending on
           the type of the result, this can be `tf.Tensor` or `tf.data.Dataset`
           instances, or a nested structure (such as an
           `structure.Struct`).

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
    if type_spec.parameter is None:
      if arg is None:
        input_map = None
      else:
        raise TypeError(
            'The computation declared no parameters; encountered an unexpected '
            'argument {}.'.format(arg))
    elif arg is None:
      raise TypeError(
          'The computation declared a parameter of type {}, but the argument '
          'was not supplied.'.format(type_spec.parameter))
    else:
      arg_type, arg_binding = tensorflow_utils.capture_result_from_graph(
          arg, graph)
      if not type_spec.parameter.is_assignable_from(arg_type):
        raise TypeError(
            'The computation declared a parameter of type {}, but the argument '
            'is of a mismatching type {}.'.format(type_spec.parameter,
                                                  arg_type))
      else:
        input_map = {
            k: graph.get_tensor_by_name(v)
            for k, v in tensorflow_utils.compute_map_from_bindings(
                computation_proto.tensorflow.parameter, arg_binding).items()
        }
    return_elements = tensorflow_utils.extract_tensor_names_from_binding(
        computation_proto.tensorflow.result)
    orig_init_op_name = computation_proto.tensorflow.initialize_op
    if orig_init_op_name:
      return_elements.append(orig_init_op_name)
    # N. B. Unlike MetaGraphDef, the GraphDef alone contains no information
    # about collections, and hence, when we import a graph with Variables,
    # those Variables are not added to global collections, and hence
    # functions like tf.compat.v1.global_variables_initializers() will not
    # contain their initialization ops.
    output_tensors = tf.import_graph_def(
        serialization_utils.unpack_graph_def(
            computation_proto.tensorflow.graph_def),
        input_map,
        return_elements,
        # N. B. It is very important not to return any names from the original
        # computation_proto.tensorflow.graph_def, those names might or might not
        # be valid in the current graph. Using a different scope makes the graph
        # somewhat more readable, since _N style de-duplication of graph
        # node names is less likely to be needed.
        name='subcomputation')

    output_map = {k: v for k, v in zip(return_elements, output_tensors)}
    new_init_op_name = output_map.pop(orig_init_op_name, None)
    return (new_init_op_name,
            tensorflow_utils.assemble_result_from_graph(
                type_spec.result, computation_proto.tensorflow.result,
                output_map))
