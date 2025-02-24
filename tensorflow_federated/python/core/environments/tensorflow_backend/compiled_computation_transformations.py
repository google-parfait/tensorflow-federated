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
"""A library of transformations for compiled computations."""

import ctypes

import federated_language
from federated_language.proto import computation_pb2

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_backend import graph_optimizations
from tensorflow_federated.python.core.environments.tensorflow_backend import graph_spec
from tensorflow_federated.python.core.environments.tensorflow_backend import serialization_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_computation_transformations
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_utils


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
      tf_block_proto.tensorflow.graph_def
  )
  graph_init_op_name = tf_block_proto.tensorflow.initialize_op
  if not graph_init_op_name:
    graph_init_op_name = None
  graph_parameter_binding = tf_block_proto.tensorflow.parameter
  graph_result_binding = tf_block_proto.tensorflow.result

  if graph_parameter_binding.WhichOneof('binding') is not None:
    graph_parameter_list = tensorflow_utils.extract_tensor_names_from_binding(
        graph_parameter_binding
    )
  else:
    graph_parameter_list = []
  graph_result_list = tensorflow_utils.extract_tensor_names_from_binding(
      graph_result_binding
  )
  return graph_spec.GraphSpec(
      graph, graph_init_op_name, graph_parameter_list, graph_result_list
  )


def optimize_tensorflow_comp(tf_computation, config_proto):
  """Applies configured optimizations to the graphdef backing a TF comp.

  Args:
    tf_computation: Instance of
      `federated_language.framework.CompiledComputation` backed by TensorFlow.
    config_proto: Instance of `tf.compat.v1.ConfigProto` specifying the
      optimizations to apply to the graph backing this TensorFlow computation.

  Returns:
    A transformed version of `tf_computation`, which has had the
    `tf.compat.v1.GraphDef` backing it run through Grappler with the specified
    configuration.
  """
  py_typecheck.check_type(
      tf_computation, federated_language.framework.CompiledComputation
  )
  tf_proto = tf_computation.proto
  graph_spec_obj = _unpack_proto_into_graph_spec(tf_proto)

  optimized_graph_spec = graph_optimizations.optimize_graph_spec(
      graph_spec_obj, config_proto
  )
  graph_def = serialization_utils.pack_graph_def(optimized_graph_spec.graph_def)
  original_tf = tf_proto.tensorflow
  tf_result_proto = computation_pb2.TensorFlow(
      graph_def=graph_def,
      initialize_op=(
          original_tf.initialize_op if original_tf.initialize_op else None
      ),
      session_token_tensor_name=(
          original_tf.session_token_tensor_name
          if original_tf.session_token_tensor_name
          else None
      ),
      parameter=(
          original_tf.parameter if original_tf.HasField('parameter') else None
      ),
      result=original_tf.result,
  )
  optimized_proto = computation_pb2.Computation(
      type=tf_proto.type, tensorflow=tf_result_proto
  )
  return federated_language.framework.CompiledComputation(
      optimized_proto, type_signature=tf_computation.type_signature
  )


class TensorFlowOptimizer:
  """Applies TF graph optimizations to `federated_language.framework.CompiledComputation`s.

  This transformer does not alter the TFF structure of the computations on which
  it is called; rather, it calls out to TensorFlow libraries which perform
  optimization on the underlying TensorFlow graph representing local processing.
  """

  def __init__(self, config_proto):
    self._config_proto = config_proto

  def should_transform(self, comp):
    return isinstance(comp, federated_language.framework.CompiledComputation)

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    return optimize_tensorflow_comp(comp, self._config_proto), True


def optimize_tensorflow_graphs(comp, grappler_config_proto):
  """Performs any static optimization on TensorFlow subcomputations."""
  transform_spec = TensorFlowOptimizer(grappler_config_proto)
  return federated_language.framework.transform_postorder(
      comp, transform_spec.transform
  )


class DisableCallOpGrappler:
  """Disables grappler in Call ops in `federated_language.framework.CompiledComputation`s.

  This overwrites the `config_proto` key of the `NodeDef.attr` field of nodes
  in a `tf.compat.v1.GraphDef` to ensure that Grappler is disabled at runtime.

  This transformer does not alter the TFF structure of the computations on which
  it is called.
  """

  def should_transform(self, comp):
    return (
        isinstance(comp, federated_language.framework.CompiledComputation)
        and comp.to_proto().WhichOneof('computation') == 'tensorflow'
    )

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    py_typecheck.check_type(
        comp, federated_language.framework.CompiledComputation
    )
    new_comp_proto = tensorflow_computation_transformations.disable_grappler_for_partitioned_calls(
        comp.proto
    )
    return (
        federated_language.framework.CompiledComputation(
            new_comp_proto, type_signature=comp.type_signature
        ),
        True,
    )


def transform_tf_call_ops_to_disable_grappler(comp):
  """Performs grappler disabling on TensorFlow subcomputations."""
  transform_spec = DisableCallOpGrappler()
  return federated_language.framework.transform_postorder(
      comp, transform_spec.transform
  )


class VerifyAllowedOps:
  """Identity transformation that verifies computation contains only allowed ops.

  This tranverses Tensorflow compiled computations and checks each op is
  permitted. If a disallowed op is found, raises a
  `DisallowedOpInTensorFlowComputationError`. Otherwise if only allowed ops are
  found, the original computation is returned.

  This transformer does not alter the TFF structure of the computations on which
  it is called.
  """

  def __init__(self, allowed_op_names: frozenset[str]):
    self._allowed_op_names = allowed_op_names

  def should_transform(
      self, comp: federated_language.framework.ComputationBuildingBlock
  ) -> bool:
    return (
        isinstance(comp, federated_language.framework.CompiledComputation)
        and comp.to_proto().WhichOneof('computation') == 'tensorflow'
    )

  def transform(
      self, comp: federated_language.framework.ComputationBuildingBlock
  ) -> tuple[federated_language.framework.ComputationBuildingBlock, bool]:
    if not self.should_transform(comp):
      return comp, False
    py_typecheck.check_type(
        comp, federated_language.framework.CompiledComputation
    )
    tensorflow_computation_transformations.check_allowed_ops(
        comp.to_proto(), self._allowed_op_names
    )
    return comp, False


def check_allowed_ops(
    comp: federated_language.framework.ComputationBuildingBlock,
    allowed_op_names: frozenset[str],
) -> tuple[federated_language.framework.ComputationBuildingBlock, bool]:
  """Checks any Tensorflow computation contains only allowed ops."""
  transform_spec = VerifyAllowedOps(allowed_op_names)
  return federated_language.framework.transform_postorder(
      comp, transform_spec.transform
  )


class RaiseOnDisallowedOp:
  """Identity transformation that raises an error if a disallowed op is found.

  This tranverses Tensorflow compiled computations searching for ops that have
  been disallowed. If a disallowed op is found, raises a
  `DisallowedOpInTensorFlowComputationError`. Otherwise if no disallowed ops are
  found, the original computation is returned.

  This transformer does not alter the TFF structure of the computations on which
  it is called.
  """

  def __init__(self, disallowed_op_names: frozenset[str]):
    self._disallowed_op_names = disallowed_op_names

  def should_transform(
      self, comp: federated_language.framework.ComputationBuildingBlock
  ) -> bool:
    return (
        isinstance(comp, federated_language.framework.CompiledComputation)
        and comp.to_proto().WhichOneof('computation') == 'tensorflow'
    )

  def transform(
      self, comp: federated_language.framework.ComputationBuildingBlock
  ) -> tuple[federated_language.framework.ComputationBuildingBlock, bool]:
    if not self.should_transform(comp):
      return comp, False
    py_typecheck.check_type(
        comp, federated_language.framework.CompiledComputation
    )
    tensorflow_computation_transformations.check_no_disallowed_ops(
        comp.to_proto(), self._disallowed_op_names
    )
    return comp, False


def check_disallowed_ops(
    comp: federated_language.framework.ComputationBuildingBlock,
    disallowed_op_names: frozenset[str],
) -> tuple[federated_language.framework.ComputationBuildingBlock, bool]:
  """Raises error on disallowed ops in any Tensorflow computation."""
  transform_spec = RaiseOnDisallowedOp(disallowed_op_names)
  return federated_language.framework.transform_postorder(
      comp, transform_spec.transform
  )


class AddUniqueIDs:
  """Populates unique IDs for compiled computations.

  This overwrites the `tensorlfow.id` field (and in the future other compiled
  computations) with a unique ID. The IDs produced should be determinstic and
  reproducible when the transform is applied to the same computation.

  This transformer does not alter the TFF structure of the computations on which
  it is called.
  """

  def should_transform(self, comp):
    return (
        isinstance(comp, federated_language.framework.CompiledComputation)
        and comp.to_proto().WhichOneof('computation') == 'tensorflow'
    )

  def transform(self, comp):
    """Transforms `comp`."""
    if not self.should_transform(comp):
      return comp, False
    py_typecheck.check_type(
        comp, federated_language.framework.CompiledComputation
    )
    new_tf_proto = computation_pb2.TensorFlow()
    new_tf_proto.CopyFrom(comp.to_proto().tensorflow)
    # Important: we must also serialize the type_signature because TFF might
    # produce (<> -> <>) or (<> -> <<>>) functions, which both could be
    # represented as the same graph with a single NoOp node. This can occur
    # particularly in MapReduceForm compiltion for secure_sum intrinsics over
    # empty structures.
    hash_value = hash(
        (comp.type_signature, comp.to_proto().tensorflow.graph_def.value)
    )
    new_tf_proto.cache_key.id = ctypes.c_uint64(hash_value).value
    new_comp_proto = computation_pb2.Computation(
        type=comp.to_proto().type, tensorflow=new_tf_proto
    )
    return (
        federated_language.framework.CompiledComputation(
            new_comp_proto, type_signature=comp.type_signature
        ),
        True,
    )


def transform_tf_add_ids(comp):
  """Adds unique IDs to each TensorFlow subcomputations."""
  transform_spec = AddUniqueIDs()
  return federated_language.framework.transform_postorder(
      comp, transform_spec.transform
  )
