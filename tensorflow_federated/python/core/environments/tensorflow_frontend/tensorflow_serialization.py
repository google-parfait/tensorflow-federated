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
"""Utilities for serializing TensorFlow computations."""

import inspect
from typing import Optional

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation_context
from tensorflow_federated.python.core.environments.tensorflow_frontend import variable_utils
from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils
from tensorflow_federated.python.tensorflow_libs import serialization_utils


def serialize_py_fn_as_tf_computation(
    fn,
    parameter_type: Optional[computation_types.Type],
    context_stack,
):
  """Serializes a TF computation with a given parameter type.

  Args:
    fn: The entity to convert into and serialize as a TF computation. This can
      currently only be a Python function. In the future, we will add here
      support for serializing the various kinds of non-eager and eager
      functions, and eventually aim at full support for and compliance with TF
      2.0. This function is currently required to declare either zero parameters
      if `parameter_type` is `None`, or exactly one parameter if it's not
      `None`.  The nested structure of this parameter must correspond to the
      structure of the 'parameter_type'. In the future, we may support functions
      with multiple args/keyword args (to be documented in the API and
      referenced from here).
    parameter_type: The parameter type specification if the fn accepts a
      parameter, or `None` if the fn doesn't declare any parameters. Either an
      instance of `computation_types.Type`.
    context_stack: The context stack to use.

  Returns:
    A tuple of (`pb.Computation`, `tff.Type`), where the computation contains
    the instance with the `pb.TensorFlow` variant set, and the type is an
    instance of `tff.Type`, potentially including Python container annotations,
    for use by TensorFlow computation wrappers.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the signature of the fn is not compatible with the given
      parameter type.
  """
  # TODO: b/113112108 - Support a greater variety of fn type signatures,
  # with keyword args or multiple args corresponding to elements of a tuple.
  # Document all accepted forms with examples in the API, and point to there
  # from here.

  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  if parameter_type is not None:
    py_typecheck.check_type(parameter_type, computation_types.Type)
  signature = inspect.signature(fn)

  with tf.Graph().as_default() as graph:
    session_token_tensor = tf.compat.v1.placeholder(
        tf.string, shape=(), name='session_token_tensor'
    )
    if parameter_type is not None:
      if len(signature.parameters) != 1:
        raise ValueError(
            'Expected the fn to declare exactly one parameter, found {!r}.'
            .format(signature.parameters)
        )
      parameter_name = next(iter(signature.parameters))
      parameter_value, parameter_binding = (
          tensorflow_utils.stamp_parameter_in_graph(
              parameter_name, parameter_type, graph
          )
      )
    else:
      if signature.parameters:
        raise ValueError(
            'Expected the fn to declare no parameters, found {!r}.'.format(
                signature.parameters
            )
        )
      parameter_value = None
      parameter_binding = None
    context = tensorflow_computation_context.TensorFlowComputationContext(
        graph, session_token_tensor
    )
    with context_stack.install(context):
      with variable_utils.record_variable_creation_scope() as all_variables:
        if parameter_value is not None:
          result = fn(parameter_value)
        else:
          result = fn()
        if result is None:
          raise computation_wrapper.ComputationReturnedNoneError(fn)
      initializer_ops = []
      if all_variables:
        # Use a readable but not-too-long name for the init_op.
        name = 'init_op_for_' + '_'.join(
            [v.name.replace(':0', '') for v in all_variables]
        )
        if len(name) > 50:
          name = 'init_op_for_{}_variables'.format(len(all_variables))
        initializer_ops.append(
            tf.compat.v1.initializers.variables(all_variables, name=name)
        )
      initializer_ops.extend(
          tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS)
      )
      if initializer_ops:
        # Before running the main new init op, run any initializers for sub-
        # computations from context.init_ops. Variables from import_graph_def
        # will not make it into the global collections, and so will not be
        # initialized without this code path.
        with tf.compat.v1.control_dependencies(context.init_ops):
          init_op_name = tf.group(
              *initializer_ops, name='grouped_initializers'
          ).name
      elif context.init_ops:
        init_op_name = tf.group(
            *context.init_ops, name='subcomputation_init_ops'
        ).name
      else:
        init_op_name = None

    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph
    )

  type_signature = computation_types.FunctionType(parameter_type, result_type)

  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding,
      session_token_tensor_name=session_token_tensor.name,
      initialize_op=init_op_name,
  )
  return (
      pb.Computation(
          type=type_serialization.serialize_type(type_signature),
          tensorflow=tensorflow,
      ),
      type_signature,
  )
