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

import os
import os.path
import shutil
import tempfile
import types
from typing import Dict, Optional, Set, MutableSequence
import zipfile

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation_context
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import function_utils
from tensorflow_federated.python.core.impl.utils import tensorflow_utils
from tensorflow_federated.python.tensorflow_libs import variable_utils


class SerializationError(Exception):
  """Error raised during value serialization or deserialization."""
  pass


def finalize_binding(binding, tensor_info_map):
  """Mutates binding by filling in actual tensor names.

  Args:
    binding: A `pb.Binding` or one of its submessages.
    tensor_info_map: A dict mapping the placeholder `tensor_name`s found in
      `binding` to final tensor names.
  """
  if not binding:
    if tensor_info_map:
      raise ValueError('Empty binding, but non-empty tensor_info_map {}'.format(
          tensor_info_map))
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
  elif isinstance(binding, pb.TensorFlow.StructBinding):
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
  signature = function_utils.get_signature(target)
  if signature.parameters and parameter_type is None:
    raise ValueError(
        'Expected the target to declare no parameters, found {!r}.'.format(
            signature.parameters))

  # In the codepath for TF V1 based serialization (tff.tf_computation),
  # we get the "wrapped" function to serialize. Here, target is the
  # raw function to be wrapped; however, we still need to know if
  # the parameter_type should be unpacked into multiple args and kwargs
  # in order to construct the TensorSpecs to be passed in the call
  # to get_concrete_fn below.
  unpack = function_utils.infer_unpack_needed(target, parameter_type, unpack)
  arg_typespecs, kwarg_typespecs, parameter_binding = (
      tensorflow_utils.get_tf_typespec_and_binding(
          parameter_type,
          arg_names=list(signature.parameters.keys()),
          unpack=unpack))

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

  @tf.function
  def target_poly(*args, **kwargs):
    result = target(*args, **kwargs)
    result_dict, result_type, result_binding = (
        tensorflow_utils.get_tf2_result_dict_and_binding(result))
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
      'var{:02d}'.format(i): v for i, v in enumerate(cc_fn.graph.variables)
  }
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
    with tf.compat.v1.Session(graph=graph) as sess:
      mgd = tf.compat.v1.saved_model.load(
          sess, tags=[tf.saved_model.SERVING], export_dir=outdir)
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
  signature = function_utils.get_signature(target)

  with tf.Graph().as_default() as graph:
    if parameter_type is not None:
      if len(signature.parameters) != 1:
        raise ValueError(
            'Expected the target to declare exactly one parameter, found {!r}.'
            .format(signature.parameters))
      parameter_name = next(iter(signature.parameters))
      parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
          parameter_name, parameter_type, graph)
    else:
      if signature.parameters:
        raise ValueError(
            'Expected the target to declare no parameters, found {!r}.'.format(
                signature.parameters))
      parameter_value = None
      parameter_binding = None
    context = tensorflow_computation_context.TensorFlowComputationContext(graph)
    with context_stack.install(context):
      with variable_utils.record_variable_creation_scope() as all_variables:
        if parameter_value is not None:
          result = target(parameter_value)
        else:
          result = target()
      initializer_ops = []
      if all_variables:
        # Use a readable but not-too-long name for the init_op.
        name = 'init_op_for_' + '_'.join(
            [v.name.replace(':0', '') for v in all_variables])
        if len(name) > 50:
          name = 'init_op_for_{}_variables'.format(len(all_variables))
        initializer_ops.append(
            tf.compat.v1.initializers.variables(all_variables, name=name))
      initializer_ops.extend(
          tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.TABLE_INITIALIZERS))
      if initializer_ops:
        # Before running the main new init op, run any initializers for sub-
        # computations from context.init_ops. Variables from import_graph_def
        # will not make it into the global collections, and so will not be
        # initialized without this code path.
        with tf.compat.v1.control_dependencies(context.init_ops):
          init_op_name = tf.group(
              *initializer_ops, name='grouped_initializers').name
      elif context.init_ops:
        init_op_name = tf.group(
            *context.init_ops, name='subcomputation_init_ops').name
      else:
        init_op_name = None

    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(parameter_type, result_type)

  # WARNING: we do not really want to be modifying the graph here if we can
  # avoid it. This is purely to work around performance issues uncovered with
  # the non-standard usage of Tensorflow and have been discussed with the
  # Tensorflow core team before being added.
  clean_graph_def = _clean_graph_def(graph.as_graph_def())
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(clean_graph_def),
      parameter=parameter_binding,
      result=result_binding,
      initialize_op=init_op_name)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow), type_signature


def _clean_graph_def(graph_def: tf.compat.v1.GraphDef) -> tf.compat.v1.GraphDef:
  """Edit the GraphDef proto to make it more performant for TFF.

  WARNING: This method must _NOT_ make any semantic changes (those that would
  change the results of the computation). TFF does not really want to be
  modifying the graph here if we can avoid it. This is purely to work around
  performance issues uncovered with the non-standard usage of Tensorflow and
  have been discussed with the Tensorflow core team before being added.

  Args:
    graph_def: the proto message to modify.

  Returns:
    A GraphDef that has been altered for performance, with no semantic
    modifications.
  """
  # TODO(b/153565654): remove this workaround once there is a way to prevent
  # the OptimizeDataset ops from being added when serializing FunctionDef that
  # received a tf.data.Dataset argument.
  _remove_optimize_dataset_ops(graph_def)
  return graph_def


# TODO(b/153565654): cleanup this workaround method when no longer needed.
def _remove_optimize_dataset_ops(graph_def: tf.compat.v1.GraphDef):
  """Removes `OptimizeDataset` and `ModelDataset` ops from the graph.

  TensorFlow Federated creates and tears down datasets frequently (one for each
  user); which is contrary to the TF expected usage of a Dataset where it is
  setup once and used throughout a long training process.

  In the TFF execution stack this leads to performance degradation where a lot
  of time is spent optimizing a dataset that will soon be thrown away. The time
  spent optimizing can even be as long as it takes to train on the dataset. For
  that reason TFF turns off this optimization.

  Luckily, it appears that generally `ModelDataset` and `OptimizeDataset` have a
  fairly straightforward pattern in graphs (though a fragile assumption); they
  have so far always appeared as:

    dataset -> OptimizeDataset -> ModelDataset -> (DatasetReduce ...)

  Each node is the first input to the next node. The following function simply
  cuts out the middle nodes and connect the first input into OptimizeDataset
  as the input to replace ModelDataset into the last op in the chain.

  Args:
    graph_def: the proto message to mutate in place.
  """
  optimize_dataset_ops = ['OptimizeDataset', 'OptimizeDatasetV2']
  ops_to_remove = optimize_dataset_ops + ['ModelDataset']

  def is_control_dep(tensor_name: str) -> bool:
    return tensor_name.startswith('^')

  def normalized_tensor_name(tensor_name: str) -> str:
    if is_control_dep(tensor_name):
      return tensor_name[1:]
    return tensor_name.split(':', maxsplit=2)[0]

  def clean_input_tensor(
      tensor_name: str,
      names_to_nodes: Dict[str, tf.compat.v1.NodeDef],
      input_args: Set[str],
  ) -> Optional[str]:
    """Rewire an input tensor that is output by a removed node."""
    node_name = normalized_tensor_name(tensor_name)
    if is_control_dep(tensor_name):
      # Simply delete control deps on removed nodes, otherwise pass through.
      input_node = names_to_nodes[node_name]
      if input_node.op in ops_to_remove:
        return None
      return tensor_name
    node = names_to_nodes.get(node_name)
    if node is None:
      if tensor_name in input_args:
        return node_name
      else:
        raise ValueError('cannot handle input {n} ({nn})'.format(
            n=tensor_name, nn=node_name))
    if node.op not in ops_to_remove:
      return tensor_name
    if node.op in optimize_dataset_ops:
      # The dataset is the first input to OptimizeDataset, so return to replace
      # the dependency on OptimizeDataset.
      return node.input[0]
    elif node.op == 'ModelDataset':
      # ModelDataset's first input is expected to be OptimizeDataset, we can
      # walk up input chain and find the input to the OptimizeDataset and return
      # that instead.
      input_node_name = normalized_tensor_name(node.input[0])
      input_node = names_to_nodes.get(input_node_name)
      if input_node is None or input_node.op not in optimize_dataset_ops:
        raise ValueError(
            'Input to ModelDataset node was {o}, expected OptimizeDataset or '
            'OptimizeDatasetV2. Unknown graph structure, aborting.'.format(
                o=input_node.op if input_node is not None else 'None'))
      return input_node.input[0]
    else:
      raise ValueError('Encoutered node [{n}] which is an op to remove, but '
                       'is not handled properly.'.format(n=node))

  def filter_nodes(node_defs: MutableSequence[tf.compat.v1.NodeDef], args):
    nodes_to_keep = []
    names_to_nodes = {}
    for node in node_defs:
      names_to_nodes[node.name] = node
      if node.op not in ops_to_remove:
        nodes_to_keep.append(node)
    func_arg_names = {arg.name for arg in args}
    for node in nodes_to_keep:
      clean_inputs = []
      for input_name in node.input:
        clean_input = clean_input_tensor(input_name, names_to_nodes,
                                         func_arg_names)
        if clean_input is not None:
          clean_inputs.append(clean_input)
      del node.input[:]
      node.input.extend(clean_inputs)
    del node_defs[:]
    node_defs.extend(nodes_to_keep)

  filter_nodes(graph_def.node, args=[])
  for function in graph_def.library.function:
    filter_nodes(function.node_def, args=function.signature.input_arg)


# The maximum size allowed for serialized sequence values. Sequence that
# serialize to values larger than this will result in errors being raised.  This
# likely occurs when the sequence is dependent on, and thus pulling in, many of
# variables from the graph.
DEFAULT_MAX_SERIALIZED_SEQUENCE_SIZE_BYTES = 20 * (1024**2)  # 20 MB


# TODO(b/137880330): there is likely opportunity here to share implementation
# with the serialization happening in
# `tensorflow_serialization.serialize_tf2_as_tf_computation()`. It would be good
# to sync with TF team about options for ensuring graph-only (variable-less)
# serializations.
def serialize_dataset(
    dataset,
    max_serialized_size_bytes=DEFAULT_MAX_SERIALIZED_SEQUENCE_SIZE_BYTES):
  """Serializes a `tf.data.Dataset` value into a `bytes` object.

  Args:
    dataset: A `tf.data.Dataset`.
    max_serialized_size_bytes: An `int` size in bytes designating the threshold
      on when to raise an error if the resulting serialization is too big.

  Returns:
    A `bytes` object that can be sent to
  `tensorflow_serialization.deserialize_dataset` to recover the original
  `tf.data.Dataset`.

  Raises:
    SerializationError: if there was an error in TensorFlow during
      serialization.
  """
  py_typecheck.check_type(dataset,
                          type_conversions.TF_DATASET_REPRESENTATION_TYPES)
  module = tf.Module()
  module.dataset = dataset
  module.dataset_fn = tf.function(lambda: module.dataset, input_signature=())

  temp_dir = tempfile.mkdtemp('dataset')
  fd, temp_zip = tempfile.mkstemp('zip')
  os.close(fd)
  try:
    tf.saved_model.save(module, temp_dir, signatures={})
    with zipfile.ZipFile(temp_zip, 'w') as z:
      for topdir, _, filenames in tf.io.gfile.walk(temp_dir):
        dest_dir = topdir[len(temp_dir):]
        for filename in filenames:
          z.write(
              os.path.join(topdir, filename), os.path.join(dest_dir, filename))
    with open(temp_zip, 'rb') as z:
      zip_bytes = z.read()
  except Exception as e:  # pylint: disable=broad-except
    raise SerializationError(
        'Error serializing tff.Sequence value. Inner error: {!s}'.format(
            e)) from e
  finally:
    tf.io.gfile.rmtree(temp_dir)
    tf.io.gfile.remove(temp_zip)

  if len(zip_bytes) > max_serialized_size_bytes:
    raise ValueError('Serialized size of Dataset ({:d} bytes) exceeds maximum '
                     'allowed ({:d} bytes)'.format(
                         len(zip_bytes), max_serialized_size_bytes))
  return zip_bytes


def deserialize_dataset(serialized_bytes):
  """Deserializes a `bytes` object to a `tf.data.Dataset`.

  Args:
    serialized_bytes: `bytes` object produced by
      `tensorflow_serialization.serialize_dataset`

  Returns:
    A `tf.data.Dataset` instance.

  Raises:
    SerializationError: if there was an error in TensorFlow during
      serialization.
  """
  py_typecheck.check_type(serialized_bytes, bytes)
  temp_dir = tempfile.mkdtemp('dataset')
  fd, temp_zip = tempfile.mkstemp('zip')
  os.close(fd)
  try:
    with open(temp_zip, 'wb') as f:
      f.write(serialized_bytes)
    with zipfile.ZipFile(temp_zip, 'r') as z:
      z.extractall(path=temp_dir)
    loaded = tf.saved_model.load(temp_dir)
    # TODO(b/156302055): Follow up here when bug is resolved, either remove
    # if this function call stops failing by default, or leave if this is
    # working as intended.
    with tf.device('cpu'):
      ds = loaded.dataset_fn()
  except Exception as e:  # pylint: disable=broad-except
    raise SerializationError(
        'Error deserializing tff.Sequence value. Inner error: {!s}'.format(
            e)) from e
  finally:
    tf.io.gfile.rmtree(temp_dir)
    tf.io.gfile.remove(temp_zip)
  return ds
