# Lint as: python3
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

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import function_utils
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import tf_computation_context
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.tensorflow_libs import graph_keys


def finalize_binding(binding, tensor_info_map):
  """Mutates binding by filling in actual tensor names.

  Args:
    binding: A `pb.Binding` or one of its submessages.
    tensor_info_map: A dict mapping the placeholder `tensor_name`s found
      in `binding` to final tensor names.
  """
  if not binding:
    if tensor_info_map:
      raise ValueError('Empty binding, but non-empty tensor_info_map {}:\n' +
                       str(tensor_info_map))
    return
  if isinstance(binding, pb.TensorFlow.Binding):
    sub_binding = getattr(binding, binding.WhichOneof('binding'))
    finalize_binding(sub_binding, tensor_info_map)

  elif isinstance(binding, pb.TensorFlow.TensorBinding):
    name = binding.tensor_name
    if name not in tensor_info_map:
      raise ValueError(
          'Did not find tensor_name {} in provided tensor_info_map with keys {}'
          .format(name, list(tensor_info_map.keys())))
    binding.tensor_name = tensor_info_map[name].name
  elif isinstance(binding, pb.TensorFlow.NamedTupleBinding):
    for sub_binding in binding.element:
      finalize_binding(sub_binding, tensor_info_map)
  else:
    raise ValueError('Unsupported binding type {}'.format(
        py_typecheck.type_string(type(binding))))


def serialize_tf2_as_tf_computation(target, parameter_type, unpack=None):
  """Serializes the 'target' as a TF computation with a given parameter type.

  Args:
    target: The entity to convert into and serialize as a TF computation. This
      can currently only be a Python function or `tf.function`, with arguments
      matching the 'parameter_type'.
    parameter_type: The parameter type specification if the target accepts a
      parameter, or `None` if the target doesn't declare any parameters. Either
      an instance of `types.Type`, or something that's convertible to it by
      `types.to_type()`.
    unpack: Whether to always unpack the parameter_type. Necessary for support
      of polymorphic tf2_computations.

  Returns:
    The constructed `pb.Computation` instance with the `pb.TensorFlow` variant
      set.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If the signature of the target is not compatible with the given
      parameter type.
  """
  py_typecheck.check_callable(target)
  parameter_type = computation_types.to_type(parameter_type)
  argspec = function_utils.get_argspec(target)
  if argspec.args and not parameter_type:
    raise ValueError(
        'Expected the target to declare no parameters, found {}.'.format(
            repr(argspec.args)))

  # In the codepath for TF V1 based serialization (tff.tf_computation),
  # we get the "wrapped" function to serialize. Here, target is the
  # raw function to be wrapped; however, we still need to know if
  # the parameter_type should be unpacked into multiple args and kwargs
  # in order to construct the TensorSpecs to be passed in the call
  # to get_concrete_fn below.
  unpack = function_utils.infer_unpack_needed(target, parameter_type, unpack)
  arg_typespecs, kwarg_typespecs, parameter_binding = (
      graph_utils.get_tf_typespec_and_binding(
          parameter_type, arg_names=argspec.args, unpack=unpack))

  # Pseudo-global to be appended to once when target_poly below is traced.
  type_and_binding_slot = []

  # N.B. To serialize a tf.function or eager python code,
  # the return type must be a flat list, tuple, or dict. However, the
  # tff.tf_computation must be able to handle structured inputs and outputs.
  # Thus, we intercept the result of calling the original target fn, introspect
  # its structure to create a result_type and bindings, and then return a
  # flat dict output. It is this new "unpacked" tf.function that we will
  # serialize using tf.saved_model.save.
  #
  # TODO(b/117428091): The return type limitation is primarily a limitation of
  # SignatureDefs  and therefore of the signatures argument to
  # tf.saved_model.save. tf.functions attached to objects and loaded back with
  # tf.saved_model.load can take/return nests; this might offer a better
  # approach to the one taken here.

  @tf.function(autograph=False)
  def target_poly(*args, **kwargs):
    result = target(*args, **kwargs)
    result_dict, result_type, result_binding = (
        graph_utils.get_tf2_result_dict_and_binding(result))
    assert not type_and_binding_slot
    # A "side channel" python output.
    type_and_binding_slot.append((result_type, result_binding))
    return result_dict

  # Triggers tracing so that type_and_binding_slot is filled.
  cc_fn = target_poly.get_concrete_function(*arg_typespecs, **kwarg_typespecs)
  assert len(type_and_binding_slot) == 1
  result_type, result_binding = type_and_binding_slot[0]

  # N.B. Note that cc_fn does *not* accept the same args and kwargs as the
  # Python target_poly; instead, it must be called with **kwargs based on the
  # unique names embedded in the TensorSpecs inside arg_typespecs and
  # kwarg_typespecs. The (preliminary) parameter_binding tracks the mapping
  # between these tensor names and the components of the (possibly nested) TFF
  # input type. When cc_fn is serialized, concrete tensors for each input are
  # introduced, and the call finalize_binding(parameter_binding,
  # sigs['serving_default'].inputs) updates the bindings to reference these
  # concrete tensors.

  # Associate vars with unique names and explicitly attach to the Checkpoint:
  var_dict = {
      'var{:02d}'.format(i): v for i, v in enumerate(cc_fn.graph.variables)}
  saveable = tf.train.Checkpoint(fn=target_poly, **var_dict)

  try:
    # TODO(b/122081673): All we really need is the  meta graph def, we could
    # probably just load that directly, e.g., using parse_saved_model from
    # tensorflow/python/saved_model/loader_impl.py, but I'm not sure we want to
    # depend on that presumably non-public symbol. Perhaps TF can expose a way
    # to just get the MetaGraphDef directly without saving to a tempfile? This
    # looks like a small change to v2.saved_model.save().
    outdir = tempfile.mkdtemp('savedmodel')
    tf.saved_model.save(saveable, outdir, signatures=cc_fn)

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
  # Checkpoints, not just run Variable initializerse (?). The right solution may
  # be to grab the target_poly.get_initialization_function(), and save a sig for
  # that.

  # Now, traverse the signature from the MetaGraphDef to find
  # find the actual tensor names and write them into the bindings.
  finalize_binding(parameter_binding, sigs['serving_default'].inputs)
  finalize_binding(result_binding, sigs['serving_default'].outputs)

  annotated_type = computation_types.FunctionType(parameter_type, result_type)

  return pb.Computation(
      type=pb.Type(
          function=pb.FunctionType(
              parameter=type_serialization.serialize_type(parameter_type),
              result=type_serialization.serialize_type(result_type))),
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(mgd.graph_def),
          parameter=parameter_binding,
          result=result_binding)), annotated_type


def serialize_py_fn_as_tf_computation(target, parameter_type, context_stack):
  """Serializes the 'target' as a TF computation with a given parameter type.

  See also `serialize_tf2_as_tf_computation` for TensorFlow 2
  serialization.

  Args:
    target: The entity to convert into and serialize as a TF computation. This
      can currently only be a Python function. In the future, we will add here
      support for serializing the various kinds of non-eager and eager
      functions, and eventually aim at full support for and compliance with TF
      2.0. This function is currently required to declare either zero parameters
      if `parameter_type` is `None`, or exactly one parameter if it's not
      `None`.  The nested structure of this parameter must correspond to the
      structure of the 'parameter_type'. In the future, we may support targets
      with multiple args/keyword args (to be documented in the API and
      referenced from here).
    parameter_type: The parameter type specification if the target accepts a
      parameter, or `None` if the target doesn't declare any parameters. Either
      an instance of `types.Type`, or something that's convertible to it by
      `types.to_type()`.
    context_stack: The context stack to use.

  Returns:
    A tuple of (`pb.Computation`, `tff.Type`), where the computation contains
    the instance with the `pb.TensorFlow` variant set, and the type is an
    instance of `tff.Type`, potentially including Python container annotations,
    for use by TensorFlow computation wrappers.

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
  argspec = inspect.getargspec(target)  # pylint: disable=deprecated-method

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

  annotated_type = computation_types.FunctionType(parameter_type, result_type)

  return pb.Computation(
      type=pb.Type(
          function=pb.FunctionType(
              parameter=type_serialization.serialize_type(parameter_type),
              result=type_serialization.serialize_type(result_type))),
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=parameter_binding,
          result=result_binding,
          initialize_op=init_op_name)), annotated_type
