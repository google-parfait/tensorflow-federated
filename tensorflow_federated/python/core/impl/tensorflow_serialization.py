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
import shutil
import tempfile
import types

# Dependency imports

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import func_utils
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import tf_computation_context
from tensorflow_federated.python.core.impl import type_serialization


nest = tf.contrib.framework.nest


def finalize_binding(binding, tensor_info_map):
  """Mutates binding by filling in actual tensor names."""
  if not binding:
    if tensor_info_map:
      raise ValueError('Empty binding, but non-empty tensor_info_map {}:\n' +
                       str(tensor_info_map))
    return
  if isinstance(binding, pb.TensorFlow.Binding):
    sub_binding = {
        'tuple': binding.tuple,
        'sequence': binding.sequence,
        'tensor': binding.tensor
    }[binding.WhichOneof('binding')]
    finalize_binding(sub_binding, tensor_info_map)

  elif isinstance(binding, pb.TensorFlow.TensorBinding):
    name = binding.tensor_name
    if name not in tensor_info_map:
      raise ValueError(
          'Did not find tensor_name {} in provided tensor_info_map with '
          'keys {}'.format(name, str(tensor_info_map.keys())))
    binding.tensor_name = tensor_info_map[name].name
  elif isinstance(binding, pb.TensorFlow.NamedTupleBinding):
    for sub_binding in binding.element:
      finalize_binding(sub_binding, tensor_info_map)
  else:
    raise ValueError('Unsupported binding type {}'.format(
        py_typecheck.type_string(type(binding))))


def serialize_tf2_as_tf_computation(target, parameter_type):
  """Serializes the 'target' as a TF computation with a given parameter type.

  Args:
    target: The entity to convert into and serialize as a TF computation. This
      can currently only be a Python function or tf.Function, with arguments
      matching the 'parameter_type'.
    parameter_type: The parameter type specification if the target accepts a
      parameter, or `None` if the target doesn't declare any parameters. Either
      an instance of `types.Type`, or something that's convertible to it by
      `types.to_type()`.

  Returns:
    The constructed `pb.Computation` instance with the `pb.TensorFlow` variant
      set.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the signature of the target is not compatible with the given
      parameter type.
  """
  if not callable(target):
    raise ValueError('target must be callable')
  parameter_type = computation_types.to_type(parameter_type)
  argspec = func_utils.get_argspec(target)
  if argspec.args and not parameter_type:
    raise ValueError(
        'Expected the target to declare no parameters, found {}.'.format(
            repr(argspec.args)))

  arg_typespecs, kwarg_typespecs, parameter_binding = (
      graph_utils.get_tf_typespec_and_binding(
          parameter_type, arg_names=argspec.args))

  # Pseudo-global to be appended to once when target_poly below is traced.
  type_and_binding_slot = []

  # To serialize a tf.function or eager pythong code,
  # the return type must be a flat list, tuple, or dict. Thus, we intercept
  # the result of calling the original target, introspect its structure
  # to create a result_type and bindings, and then return a flat dict output.
  @tf.contrib.eager.function(autograph=False)
  def target_poly(*args, **kwargs):
    result = target(*args, **kwargs)
    result_dict, result_type, result_binding = (
        graph_utils.get_result_dict_and_binding(result))
    assert not type_and_binding_slot
    type_and_binding_slot.append((result_type, result_binding))
    return result_dict

  cc_fn = target_poly.get_concrete_function(*arg_typespecs, **kwarg_typespecs)

  # Associate vars with unique names and explicitly attach to the Checkpoint:
  var_dict = {
      'var{:02d}'.format(i): v for i, v in enumerate(cc_fn.graph.variables)}

  saveable = tf.train.Checkpoint(fn=target_poly, **var_dict)
  try:
    outdir = tempfile.mkdtemp('savedmodel')
    tf.saved_model.experimental.save(saveable, outdir, signatures=cc_fn)

    # This should be set now that tracing has happened in save():
    result_type, result_binding = type_and_binding_slot[0]

    # XXX Q - I'm inclined to leave this as is and file a bug.
    # All we really need is the  meta graph def, we could probably just load
    # that directly, e.g., using parse_saved_model from
    # tensorflow/python/saved_model/loader_impl.py, But I'm not sure we want to
    # depend on that presumably non-public symbol, and if we make a request
    # to the TF team, it should probably be to just be able to get the
    # MetaGraphDef without writing to disk at all. This looks like a small
    # change to v2.saved_model.save()
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
      mgd = tf.saved_model.loader.load(
          sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir=outdir)
  finally:
    shutil.rmtree(outdir)
  sigs = mgd.signature_def

  # TODO(b/123102455): Figure out how to support the init_op. The meta graph def
  # contains sigs['__saved_model_init_op'].outputs['__saved_model_init_op']. It
  # probably won't do what we want, because it will want to read from
  # Checkpoints, not just run Variable initializerse. The right solution may be
  # to grab the target_poly.get_initialization_function(), and save a sig for
  # that.

  # Now, traverse the signature from the MetaGraphDef to find
  # find the actual tensor names and write them into the bindings.
  finalize_binding(parameter_binding, sigs['serving_default'].inputs)
  finalize_binding(result_binding, sigs['serving_default'].outputs)

  return pb.Computation(
      type=pb.Type(
          function=pb.FunctionType(
              parameter=type_serialization.serialize_type(parameter_type),
              result=type_serialization.serialize_type(result_type))),
      tensorflow=pb.TensorFlow(
          graph_def=mgd.graph_def,
          parameter=parameter_binding,
          result=result_binding))


def serialize_py_func_as_tf_computation(target, parameter_type, context_stack):
  """Serializes the 'target' as a TF computation with a given parameter type.

  Args:
    target: The entity to convert into and serialize as a TF computation. This
      can currently only be a Python function. See
      `serialize_tf2_as_tf_computation` for serializing eager-mode tf.functions.
      This function is currently required to declare either zero parameters if
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
    result_type, result_binding = graph_utils.capture_result_from_graph(result)

  return pb.Computation(
      type=pb.Type(
          function=pb.FunctionType(
              parameter=type_serialization.serialize_type(parameter_type),
              result=type_serialization.serialize_type(result_type))),
      tensorflow=pb.TensorFlow(
          graph_def=graph.as_graph_def(),
          parameter=parameter_binding,
          result=result_binding))
