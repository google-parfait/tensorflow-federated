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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import types

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import tf_computation_context
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.tensorflow_libs import graph_keys


def serialize_py_fn_as_tf_computation(target, parameter_type, context_stack):
  """Serializes the 'target' as a TF computation with a given parameter type.

  Args:
    target: The entity to convert into and serialize as a TF computation. This
      can currently only be a Python function. In the future, we will add here
      support for serializing the various kinds of non-eager and eager defuns,
      and eventually aim at full support for and compliance with TF 2.0. This
      function is currently required to declare either zero parameters if
      `parameter_type` is `None`, or exactly one parameter if it's not `None`.
      The nested structure of this parameter must correspond to the structure of
      the 'parameter_type'. In the future, we may support targets with multiple
      args/keyword args (to be documented in the API and referenced from here).
    parameter_type: The parameter type specification if the target accepts a
      parameter, or `None` if the target doesn't declare any parameters. Either
      an instance of `types.Type`, or something that's convertible to it by
      `types.to_type()`.
    context_stack: The context stack to use.

  Returns:
    The constructed `pb.Computation` instance with the `pb.TensorFlow` variant
      set.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the signature of the target is not compatible with the given
      parameter type.
  """
  # TODO(b/113112108): Support a greater variety of target type signatures,
  # with keyword args or multiple args corresponding to elements of a tuple.
  # Document all accepted forms with examples in the API, and point to there
  # from here.

  py_typecheck.check_type(target, types.FunctionType)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  parameter_type = computation_types.to_type(parameter_type)
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
    context = tf_computation_context.TensorFlowComputationContext(graph)
    with context_stack.install(context):
      result = target(*args)

      # TODO(b/122081673): This needs to change for TF 2.0. We may also
      # want to allow the person creating a tff.tf_computation to specify
      # a different initializer; e.g., if it is known that certain
      # variables will be assigned immediately to arguments of the function,
      # then it is wasteful to initialize them before this.
      #
      # The following is a bit of a work around: the collections below may
      # contain variables more than once, hence we throw into a set. TFF needs
      # to ensure all variables are initialized, but not all variables are
      # always in the collections we expect. tff.learning._KerasModel tries to
      # pull Keras variables (that may or may not be in GLOBAL_VARIABLES) into
      # TFF_MODEL_VARIABLES for now.
      all_variables = set(
          tf.global_variables() + tf.local_variables() +
          tf.get_collection(graph_keys.GraphKeys.VARS_FOR_TFF_TO_INITIALIZE))
      if all_variables:
        # Use a readable but not-too-long name for the init_op.
        name = 'init_op_for_' + '_'.join(
            [v.name.replace(':0', '') for v in all_variables])
        if len(name) > 50:
          name = 'init_op_for_{}_variables'.format(len(all_variables))
        with tf.control_dependencies(context.init_ops):
          # Before running the main new init op, run any initializers for sub-
          # computations from context.init_ops. Variables from import_graph_def
          # will not make it into the global collections, and so will not be
          # initialized without this code path.
          init_op_name = tf.initializers.variables(
              all_variables, name=name).name
      elif context.init_ops:
        init_op_name = tf.group(
            *context.init_ops, name='subcomputation_init_ops').name
      else:
        init_op_name = None

    result_type, result_binding = graph_utils.capture_result_from_graph(
        result, graph)

  return pb.Computation(
      type=pb.Type(
          function=pb.FunctionType(
              parameter=type_serialization.serialize_type(parameter_type),
              result=type_serialization.serialize_type(result_type))),
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=parameter_binding,
          result=result_binding,
          initialize_op=init_op_name))
