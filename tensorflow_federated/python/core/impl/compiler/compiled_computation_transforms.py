# Copyright 2019, The TensorFlow Federated Authors.
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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Holds library of transformations for on compiled computations."""

import ctypes

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_transformations
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.utils import tensorflow_utils
from tensorflow_federated.python.tensorflow_libs import graph_optimizations
from tensorflow_federated.python.tensorflow_libs import graph_spec


def _unpack_proto_into_graph_spec(tf_block_proto):
  """Packs a TF proto into a `graph_spec.GraphSpec`.

  Args:
    tf_block_proto: Instance of `computation_pb2.Computation` with `tensorflow`
      `computation` attribute.

  Returns:
    Instance of `graph_spec.GraphSpec` containing Python representations of
    the information present in `tf_block_proto`.
  """
  graph = serialization_utils.unpack_graph_def(
      tf_block_proto.tensorflow.graph_def)
  graph_init_op_name = tf_block_proto.tensorflow.initialize_op
  if not graph_init_op_name:
    graph_init_op_name = None
  graph_parameter_binding = tf_block_proto.tensorflow.parameter
  graph_result_binding = tf_block_proto.tensorflow.result

  if graph_parameter_binding.WhichOneof('binding') is not None:
    graph_parameter_list = tensorflow_utils.extract_tensor_names_from_binding(
        graph_parameter_binding)
  else:
    graph_parameter_list = []
  graph_result_list = tensorflow_utils.extract_tensor_names_from_binding(
      graph_result_binding)
  return graph_spec.GraphSpec(graph, graph_init_op_name, graph_parameter_list,
                              graph_result_list)


def optimize_tensorflow_comp(tf_computation, config_proto):
  """Applies configured optimizations to the graphdef backing a TF comp.

  Args:
    tf_computation: Instance of `building_blocks.CompiledComputation` backed by
      TensorFlow.
    config_proto: Instance of `tf.compat.v1.ConfigProto` specifying the
      optimizations to apply to the graph backing this TensorFlow computation.

  Returns:
    A transformed version of `tf_computation`, which has had the
    `tf.compat.v1.GraphDef` backing it run through Grappler with the specified
    configuration.
  """
  py_typecheck.check_type(tf_computation, building_blocks.CompiledComputation)
  tf_proto = tf_computation.proto
  graph_spec_obj = _unpack_proto_into_graph_spec(tf_proto)

  optimized_graph_spec = graph_optimizations.optimize_graph_spec(
      graph_spec_obj, config_proto)
  graph_def = serialization_utils.pack_graph_def(optimized_graph_spec.graph_def)
  original_tf = tf_proto.tensorflow
  tf_result_proto = pb.TensorFlow(
      graph_def=graph_def,
      initialize_op=(original_tf.initialize_op
                     if original_tf.initialize_op else None),
      session_token_tensor_name=(original_tf.session_token_tensor_name
                                 if original_tf.session_token_tensor_name else
                                 None),
      parameter=(original_tf.parameter
                 if original_tf.HasField('parameter') else None),
      result=original_tf.result)
  optimized_proto = pb.Computation(
      type=tf_proto.type, tensorflow=tf_result_proto)
  return building_blocks.CompiledComputation(
      optimized_proto, type_signature=tf_computation.type_signature)


class TensorFlowOptimizer(transformation_utils.TransformSpec):
  """Applies TF graph optimizations to `building_blocks.CompiledComputation`s.

  This `transformation_utils.TransformSpec` does not alter the TFF structure of
  the computations on which it is called; rather, it calls out to TensorFlow
  libraries which perform optimization on the underlying TensorFlow graph
  representing local processing.
  """

  def __init__(self, config_proto):
    self._config_proto = config_proto

  def should_transform(self, comp):
    return comp.is_compiled_computation()

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    return optimize_tensorflow_comp(comp, self._config_proto), True


class DisableCallOpGrappler(transformation_utils.TransformSpec):
  """Disables grappler in Call ops in `building_blocks.CompiledComputation`s.

  This overwrites the `config_proto` key of the `NodeDef.attr` field of nodes
  in a `tf.compat.v1.GraphDef` to ensure that Grappler is disabled at runtime.

  This `transformation_utils.TransformSpec` does not alter the TFF structure of
  the computations on which it is called.
  """

  def should_transform(self, comp):
    return (comp.is_compiled_computation() and
            comp.proto.WhichOneof('computation') == 'tensorflow')

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    py_typecheck.check_type(comp, building_blocks.CompiledComputation)
    new_comp_proto = tensorflow_computation_transformations.disable_grappler_for_partitioned_calls(
        comp.proto)
    return building_blocks.CompiledComputation(
        new_comp_proto, type_signature=comp.type_signature), True


class AddUniqueIDs(transformation_utils.TransformSpec):
  """Populates unique IDs for compiled computations.

  This overwrites the `tensorlfow.id` field (and in the future other compiled
  computations) with a unique ID. The IDs produced should be determinstic and
  reproducible when the transform is applied to the same computation.

  This `transformation_utils.TransformSpec` does not alter the TFF structure of
  the computations on which it is called.
  """

  def should_transform(self, comp):
    return (comp.is_compiled_computation() and
            comp.proto.WhichOneof('computation') == 'tensorflow')

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    py_typecheck.check_type(comp, building_blocks.CompiledComputation)
    new_tf_proto = pb.TensorFlow()
    new_tf_proto.CopyFrom(comp.proto.tensorflow)
    # Important: we must also serialize the type_signature because TFF might
    # produce (<> -> <>) or (<> -> <<>>) functions, which both could be
    # represented as the same graph with a single NoOp node. This can occur
    # particularly in MapReduceForm compiltion for secure_sum intrinsics over
    # empty structures.
    hash_value = hash(
        (comp.type_signature, comp.proto.tensorflow.graph_def.value))
    new_tf_proto.cache_key.id = ctypes.c_uint64(hash_value).value
    new_comp_proto = pb.Computation(
        type=comp.proto.type, tensorflow=new_tf_proto)
    return building_blocks.CompiledComputation(
        new_comp_proto, type_signature=comp.type_signature), True
