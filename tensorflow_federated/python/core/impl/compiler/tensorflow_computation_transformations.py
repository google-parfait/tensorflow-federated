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
"""A library of transformation functions for tensorflow computation."""

import itertools
from typing import Any, Dict, FrozenSet, Optional

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import type_analysis


# List of op names that are eligible for Grappler disabling.
CALL_OPS = frozenset(['StatefulPartitionedCall', 'PartitionedCall'])


def disable_grappler_for_partitioned_calls(proto):
  """Disables grappler for `PartitionedCall` and `StatefulPartitionedCall` nodes in the graph.

  TensorFlow serializes a `ConfigProto` into `PartitionedCall` and
  `StatefulPartitionedCall` the `config_proto` `attr` of graph nodes. This
  overrides any session config that might disable runtime grappler. The disable
  grappler for these nodes as well, this function overwrites the serialized
  configproto, setting the `disable_meta_optimizer` field to `True.

  Args:
    proto: Instance of `pb.Computation` with the `tensorflow` field populated.

  Returns:
    A transformed instance of `pb.Computation` with a `tensorflow` field.
  """
  py_typecheck.check_type(proto, pb.Computation)
  computation_oneof = proto.WhichOneof('computation')
  if computation_oneof != 'tensorflow':
    raise TypeError('`prune_tensorflow_proto` only accepts `Computation` '
                    'protos of the "tensorflow" variety; you have passed '
                    'one of variety {}.'.format(computation_oneof))
  original_tf = proto.tensorflow
  graph_def = serialization_utils.unpack_graph_def(original_tf.graph_def)
  all_nodes = itertools.chain(graph_def.node,
                              *[f.node_def for f in graph_def.library.function])
  for node in all_nodes:
    if node.op not in CALL_OPS:
      continue
    attr_str = node.attr.get('config_proto')
    if attr_str is None:
      config_proto = tf.compat.v1.ConfigProto()
    else:
      config_proto = tf.compat.v1.ConfigProto.FromString(attr_str.s)
    config_proto.graph_options.rewrite_options.disable_meta_optimizer = True
    attr_str.s = config_proto.SerializeToString(deterministic=True)
  tf_block = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph_def),
      initialize_op=original_tf.initialize_op
      if original_tf.initialize_op else None,
      parameter=original_tf.parameter
      if original_tf.HasField('parameter') else None,
      result=original_tf.result)
  new_proto = pb.Computation(type=proto.type, tensorflow=tf_block)
  return new_proto


class DisallowedOpInTensorFlowComputationError(Exception):
  """Error raised when a TensorFlow computation contains a disallowed op."""


def _check_ops(proto: pb.Computation,
               allowed_op_names: Optional[FrozenSet[str]] = None,
               disallowed_op_names: Optional[FrozenSet[str]] = None):
  """Checks the ops in the TensorFlow computation.

  If allowed_op_names is specified, then _check_ops checks the incoming proto
  contains only ops in the set. On the other hand, if disallowed_op_names is
  specified, then _check_ops checks the proto contains no ops contained in the
  set. One of the two op set arguments must be non-empty, and if both are, then
  allowed_op_names takes precedent.

  Args:
    proto: Instance of `pb.Computation` with the `tensorflow` field populated.
    allowed_op_names: Set of allowed op names.
    disallowed_op_names: Set of disallowed op names.

  Raises:
    DisallowedOpInTensorFlowComputationError: If the computation contains a
      disallowed op.
    RuntimeError: If both allowed_op_names and disallowed_op_names are empty.
  """
  py_typecheck.check_type(proto, pb.Computation)
  computation_oneof = proto.WhichOneof('computation')
  if computation_oneof != 'tensorflow':
    raise TypeError('`prune_tensorflow_proto` only accepts `Computation` '
                    'protos of the "tensorflow" variety; you have passed '
                    'one of variety {}.'.format(computation_oneof))
  tensorflow_pb = proto.tensorflow
  graph_def = serialization_utils.unpack_graph_def(tensorflow_pb.graph_def)
  all_nodes = itertools.chain(graph_def.node,
                              *[f.node_def for f in graph_def.library.function])
  found_disallowed_op_names = set()

  if allowed_op_names:
    for node in all_nodes:
      if node.op not in allowed_op_names:
        found_disallowed_op_names.add(node.op)
  elif disallowed_op_names:
    for node in all_nodes:
      if node.op in disallowed_op_names:
        found_disallowed_op_names.add(node.op)
  else:
    raise RuntimeError(
        'One of allowed_op_names or disallowed_op_names must be non-empty')

  if found_disallowed_op_names:
    found_disallowed_op_names_str = ', '.join(found_disallowed_op_names)
    raise DisallowedOpInTensorFlowComputationError(
        f'Found disallowed ops: {found_disallowed_op_names_str}')


def check_allowed_ops(proto: pb.Computation, allowed_op_names: FrozenSet[str]):
  """Checks the TensorFlow computation contains allowed ops.

  Args:
    proto: Instance of `pb.Computation` with the `tensorflow` field populated.
    allowed_op_names: Set of allowed op names.

  Raises:
    DisallowedOpInTensorFlowComputationError: If the computation contains an op
      not in allowed_op_names.
  """
  _check_ops(proto, allowed_op_names=allowed_op_names)


def check_no_disallowed_ops(proto: pb.Computation,
                            disallowed_op_names: FrozenSet[str]):
  """Checks the TensorFlow computation for disallowed ops.

  Args:
    proto: Instance of `pb.Computation` with the `tensorflow` field populated.
    disallowed_op_names: Set of disallowed op names.

  Raises:
    DisallowedOpInTensorFlowComputationError: If the computation contains a
      disallowed op.
  """
  _check_ops(proto, disallowed_op_names=disallowed_op_names)


def _unpack_compiled_computations(
    comp: building_blocks.ComputationBuildingBlock
) -> building_blocks.ComputationBuildingBlock:
  """Deserializes compiled computations into building blocks where possible."""

  def _unpack(subcomp):
    if not subcomp.is_compiled_computation():
      return subcomp, False
    kind = subcomp.proto.WhichOneof('computation')
    if kind == 'tensorflow' or kind == 'xla':
      return subcomp, False
    return building_blocks.ComputationBuildingBlock.from_proto(
        subcomp.proto), True

  comp, _ = transformation_utils.transform_postorder(comp, _unpack)
  return comp


class XlaToTensorFlowError(ValueError):
  """An error indicating an attempt to compile XLA code to TensorFlow."""


class ExternalBlockToTensorFlowError(ValueError):
  """An error indicating an attempt to compile external blocks to TensorFlow."""


def _evaluate_to_tensorflow(
    comp: building_blocks.ComputationBuildingBlock,
    bindings: Dict[str, Any],
) -> Any:
  """Evaluates `comp` within a TensorFlow context, returning a tensor structure.

  Args:
    comp: A building block to evaluate. In order to evaluate to TensorFlow, this
      block must not contain any `Intrinsic`, `Data`, or `Placement` blocks, and
      must not contain `CompiledComputation` blocks of kinds other than
      `tensorflow`. `comp` must also have unique names.
    bindings: A mapping from names to values. Since names in `comp` must be
      unique, all block locals and lambda arguments can be inserted into this
      flat-level map.

  Returns:
    A structure of TensorFlow values representing the result of evaluating
    `comp`. Functional building blocks are represented as callables.

  Raises:
    XlaToTensorFlowError: If `comp` contains a `CompiledComputation` containing
      XLA.
    ExternalBlockToTensorFlowError: If `comp` contains an `Intrinsic`, `Data`,
      or `Placement` block.
    ValueError: If `comp` contains `CompiledCompilations` other than
      TensorFlow or XLA.
  """
  if comp.is_block():
    for name, value in comp.locals:
      bindings[name] = _evaluate_to_tensorflow(value, bindings)
    return _evaluate_to_tensorflow(comp.result, bindings)
  if comp.is_compiled_computation():
    kind = comp.proto.WhichOneof('computation')
    if kind == 'tensorflow':

      def call_concrete(*args):
        concrete = computation_impl.ConcreteComputation(
            comp.proto, context_stack_impl.context_stack)
        result = concrete(*args)
        if comp.type_signature.result.is_struct():
          return structure.from_container(result, recursive=True)
        return result

      return call_concrete
    if kind == 'xla':
      raise XlaToTensorFlowError(
          f'Cannot compile XLA subcomptation to TensorFlow:\n{comp}')
    raise ValueError(f'Unexpected compiled computation kind:\n{kind}')
  if comp.is_call():
    function = _evaluate_to_tensorflow(comp.function, bindings)
    if comp.argument is None:
      return function()
    else:
      return function(_evaluate_to_tensorflow(comp.argument, bindings))
  if comp.is_lambda():
    if comp.parameter_type is None:
      return lambda: _evaluate_to_tensorflow(comp.result, bindings)
    else:

      def lambda_function(arg):
        bindings[comp.parameter_name] = arg
        return _evaluate_to_tensorflow(comp.result, bindings)

      return lambda_function
  if comp.is_reference():
    return bindings[comp.name]
  if comp.is_selection():
    return _evaluate_to_tensorflow(comp.source, bindings)[comp.as_index()]
  if comp.is_struct():
    elements = []
    for name, element in structure.iter_elements(comp):
      elements.append((name, _evaluate_to_tensorflow(element, bindings)))
    return structure.Struct(elements)
  if comp.is_intrinsic() or comp.is_data() or comp.is_placement():
    raise ExternalBlockToTensorFlowError(
        f'Cannot evaluate intrinsic, data, or placement blocks to tensorflow, found {comp}'
    )


def compile_local_computation_to_tensorflow(
    comp: building_blocks.ComputationBuildingBlock,
) -> building_blocks.ComputationBuildingBlock:
  """Compiles a fully specified local computation to TensorFlow.

  Args:
    comp: A `building_blocks.ComputationBuildingBlock` which can be compiled to
      TensorFlow. In order to compile a computation to TensorFlow, it must not
      contain 1. References to values defined outside of comp, 2. `Data`,
      `Intrinsic`, or `Placement` blocks, or 3. Calls to intrinsics or
      non-TensorFlow computations.

  Returns:
    A `building_blocks.ComputationBuildingBlock` containing a TensorFlow-only
    representation of `comp`. If `comp` is of functional type, this will be
    a `building_blocks.CompiledComputation`. Otherwise, it will be a
    `building_blocks.Call` which wraps a `building_blocks.CompiledComputation`.
  """
  if not comp.type_signature.is_function():
    lambda_wrapped = building_blocks.Lambda(None, None, comp)
    return building_blocks.Call(
        compile_local_computation_to_tensorflow(lambda_wrapped), None)

  parameter_type = comp.type_signature.parameter
  type_analysis.check_tensorflow_compatible_type(parameter_type)
  type_analysis.check_tensorflow_compatible_type(comp.type_signature.result)

  if (comp.is_compiled_computation() and
      comp.proto.WhichOneof('computation') == 'tensorflow'):
    return comp

  # Ensure that unused values are removed and that reference bindings have
  # unique names.
  comp = _unpack_compiled_computations(comp)
  comp = transformations.to_call_dominant(comp)

  if parameter_type is None:
    to_evaluate = building_blocks.Call(comp)

    @tensorflow_computation.tf_computation
    def result_computation():
      return _evaluate_to_tensorflow(to_evaluate, {})
  else:
    name_generator = building_block_factory.unique_name_generator(comp)
    parameter_name = next(name_generator)
    to_evaluate = building_blocks.Call(
        comp, building_blocks.Reference(parameter_name, parameter_type))

    @tensorflow_computation.tf_computation(parameter_type)
    def result_computation(arg):
      if parameter_type.is_struct():
        arg = structure.from_container(arg, recursive=True)
      return _evaluate_to_tensorflow(to_evaluate, {parameter_name: arg})

  return result_computation.to_compiled_building_block()


def compile_local_subcomputations_to_tensorflow(
    comp: building_blocks.ComputationBuildingBlock,
) -> building_blocks.ComputationBuildingBlock:
  """Compiles subcomputations to TensorFlow where possible."""
  comp = _unpack_compiled_computations(comp)
  local_cache = {}

  def _is_local(comp):
    cached = local_cache.get(comp, None)
    if cached is not None:
      return cached
    if (comp.is_intrinsic() or comp.is_data() or comp.is_placement() or
        type_analysis.contains_federated_types(comp.type_signature)):
      local_cache[comp] = False
      return False
    if (comp.is_compiled_computation() and
        comp.proto.WhichOneof('computation') == 'xla'):
      local_cache[comp] = False
      return False
    for child in comp.children():
      if not _is_local(child):
        local_cache[comp] = False
        return False
    return True

  unbound_ref_map = transformation_utils.get_map_of_unbound_references(comp)

  def _compile_if_local(comp):
    if _is_local(comp) and not unbound_ref_map[comp]:
      return compile_local_computation_to_tensorflow(comp), True
    return comp, False

  # Note: this transformation is preorder so that local subcomputations are not
  # first transformed to TensorFlow if they have a parent local computation
  # which could have instead been transformed into a larger single block of
  # TensorFlow.
  comp, _ = transformation_utils.transform_preorder(comp, _compile_if_local)
  return comp
