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
"""Test utils for TFF computations."""

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def create_chained_calls(functions, arg):
  r"""Creates a chain of `n` calls.

       Call
      /    \
  Comp      ...
               \
                Call
               /    \
           Comp      Comp

  The first functional computation in `functions` must have a parameter type
  that is assignable from the type of `arg`, each other functional computation
  in `functions` must have a parameter type that is assignable from the previous
  functional computations result type.

  Args:
    functions: A Python list of functional computations.
    arg: A `building_blocks.ComputationBuildingBlock`.

  Returns:
    A `building_blocks.Call`.
  """
  for fn in functions:
    if not fn.parameter_type.is_assignable_from(arg.type_signature):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature)))
    call = building_blocks.Call(fn, arg)
    arg = call
  return call


def create_whimsy_block(comp, variable_name, variable_type=tf.int32):
  r"""Returns an identity block.

           Block
          /     \
  [x=data]       Comp

  Args:
    comp: The computation to use as the result.
    variable_name: The name of the variable.
    variable_type: The type of the variable.
  """
  data = building_blocks.Data('data', variable_type)
  return building_blocks.Block([(variable_name, data)], comp)


def create_whimsy_called_intrinsic(parameter_name, parameter_type=tf.int32):
  r"""Returns a whimsy called intrinsic.

            Call
           /    \
  intrinsic      Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  intrinsic_type = computation_types.FunctionType(parameter_type,
                                                  parameter_type)
  intrinsic = building_blocks.Intrinsic('intrinsic', intrinsic_type)
  ref = building_blocks.Reference(parameter_name, parameter_type)
  return building_blocks.Call(intrinsic, ref)


def create_whimsy_called_federated_aggregate(
    accumulate_parameter_name='acc_param',
    merge_parameter_name='merge_param',
    report_parameter_name='report_param',
    value_type=tf.int32):
  r"""Returns a whimsy called federated aggregate.

                      Call
                     /    \
  federated_aggregate      Tuple
                           |
                           [data, data, Lambda(x), Lambda(x), Lambda(x)]
                                        |          |          |
                                        data       data       data

  Args:
    accumulate_parameter_name: The name of the accumulate parameter.
    merge_parameter_name: The name of the merge parameter.
    report_parameter_name: The name of the report parameter.
    value_type: The TFF type of the value to be aggregated, placed at CLIENTS.
  """
  federated_value_type = computation_types.FederatedType(
      value_type, placements.CLIENTS)
  value = building_blocks.Data('data', federated_value_type)
  zero = building_blocks.Data('data', value_type)
  accumulate_type = computation_types.StructType((value_type, value_type))
  accumulate_result = building_blocks.Data('data', value_type)
  accumulate = building_blocks.Lambda(accumulate_parameter_name,
                                      accumulate_type, accumulate_result)
  merge_type = computation_types.StructType((value_type, value_type))
  merge_result = building_blocks.Data('data', value_type)
  merge = building_blocks.Lambda(merge_parameter_name, merge_type, merge_result)
  report_result = building_blocks.Data('data', value_type)
  report = building_blocks.Lambda(report_parameter_name, value_type,
                                  report_result)
  return building_block_factory.create_federated_aggregate(
      value, zero, accumulate, merge, report)


def create_whimsy_called_federated_apply(parameter_name,
                                         parameter_type=tf.int32):
  r"""Returns a whimsy called federated apply.

                  Call
                 /    \
  federated_apply      Tuple
                       |
                       [Lambda(x), data]
                        |
                        Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  fn = create_identity_function(parameter_name, parameter_type)
  arg_type = computation_types.FederatedType(parameter_type, placements.SERVER)
  arg = building_blocks.Data('data', arg_type)
  return building_block_factory.create_federated_apply(fn, arg)


def create_whimsy_called_federated_broadcast(value_type=tf.int32):
  r"""Returns a whimsy called federated broadcast.

                      Call
                     /    \
  federated_broadcast      data

  Args:
    value_type: The type of the value.
  """
  federated_type = computation_types.FederatedType(value_type,
                                                   placements.SERVER)
  value = building_blocks.Data('data', federated_type)
  return building_block_factory.create_federated_broadcast(value)


def create_whimsy_called_federated_collect(value_type=tf.int32):
  federated_type = computation_types.at_clients(value_type)
  value = building_blocks.Data('data', federated_type)
  return building_block_factory.create_federated_collect(value)


def create_whimsy_called_federated_map(parameter_name, parameter_type=tf.int32):
  r"""Returns a whimsy called federated map.

                Call
               /    \
  federated_map      Tuple
                     |
                     [Lambda(x), data]
                      |
                      Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  fn = create_identity_function(parameter_name, parameter_type)
  arg_type = computation_types.FederatedType(parameter_type, placements.CLIENTS)
  arg = building_blocks.Data('data', arg_type)
  return building_block_factory.create_federated_map(fn, arg)


def create_whimsy_called_federated_map_all_equal(parameter_name,
                                                 parameter_type=tf.int32):
  r"""Returns a whimsy called federated map.

                          Call
                         /    \
  federated_map_all_equal      Tuple
                               |
                               [Lambda(x), data]
                                |
                                Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  fn = create_identity_function(parameter_name, parameter_type)
  arg_type = computation_types.FederatedType(
      parameter_type, placements.CLIENTS, all_equal=True)
  arg = building_blocks.Data('data', arg_type)
  return building_block_factory.create_federated_map_all_equal(fn, arg)


def create_whimsy_called_federated_mean(value_type=tf.float32,
                                        weights_type=None):
  fed_value_type = computation_types.at_clients(value_type)
  values = building_blocks.Data('values', fed_value_type)
  if weights_type is not None:
    fed_weights_type = computation_types.at_clients(weights_type)
    weights = building_blocks.Data('weights', fed_weights_type)
  else:
    weights = None
  return building_block_factory.create_federated_mean(values, weights)


def create_whimsy_called_federated_secure_sum_bitwidth(value_type=tf.int32):
  r"""Returns a whimsy called secure sum.

                       Call
                      /    \
  federated_secure_sum_bitwidth      [data, data]

  Args:
    value_type: The type of the value.
  """
  federated_type = computation_types.FederatedType(value_type,
                                                   placements.CLIENTS)
  value = building_blocks.Data('data', federated_type)
  bitwidth = building_blocks.Data('data', value_type)
  return building_block_factory.create_federated_secure_sum_bitwidth(
      value, bitwidth)


def create_whimsy_called_federated_sum(value_type=tf.int32):
  r"""Returns a whimsy called federated sum.

                Call
               /    \
  federated_sum      data

  Args:
    value_type: The type of the value.
  """
  federated_type = computation_types.FederatedType(value_type,
                                                   placements.CLIENTS)
  value = building_blocks.Data('data', federated_type)
  return building_block_factory.create_federated_sum(value)


def create_whimsy_called_sequence_map(parameter_name, parameter_type=tf.int32):
  r"""Returns a whimsy called sequence map.

               Call
              /    \
  sequence_map      data

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  fn = create_identity_function(parameter_name, parameter_type)
  arg_type = computation_types.SequenceType(parameter_type)
  arg = building_blocks.Data('data', arg_type)
  return building_block_factory.create_sequence_map(fn, arg)


def create_whimsy_called_federated_value(placement: placements.PlacementLiteral,
                                         value_type=tf.int32):
  value = building_blocks.Data('data', value_type)
  return building_block_factory.create_federated_value(value, placement)


def create_identity_block(variable_name, comp):
  r"""Returns an identity block.

           Block
          /     \
  [x=comp]       Ref(x)

  Args:
    variable_name: The name of the variable.
    comp: The computation to use as the variable.
  """
  ref = building_blocks.Reference(variable_name, comp.type_signature)
  return building_blocks.Block([(variable_name, comp)], ref)


def create_identity_block_with_whimsy_data(variable_name,
                                           variable_type=tf.int32):
  r"""Returns an identity block with a whimsy `Data` computation.

           Block
          /     \
  [x=data]       Ref(x)

  Args:
    variable_name: The name of the variable.
    variable_type: The type of the variable.
  """
  data = building_blocks.Data('data', variable_type)
  return create_identity_block(variable_name, data)


def create_identity_function(parameter_name, parameter_type=tf.int32):
  r"""Returns an identity function.

  Lambda(x)
  |
  Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  ref = building_blocks.Reference(parameter_name, parameter_type)
  return building_blocks.Lambda(ref.name, ref.type_signature, ref)


def create_lambda_to_whimsy_called_intrinsic(parameter_name,
                                             parameter_type=tf.int32):
  r"""Returns a lambda to call a whimsy intrinsic.

            Lambda(x)
            |
            Call
           /    \
  intrinsic      Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  call = create_whimsy_called_intrinsic(
      parameter_name=parameter_name, parameter_type=parameter_type)
  return building_blocks.Lambda(parameter_name, parameter_type, call)


def create_nested_syntax_tree():
  r"""Constructs computation with explicit ordering for testing traversals.

  The goal of this computation is to exercise each switch
  in transform_postorder_with_symbol_bindings, at least all those that recurse.

  The computation this function constructs can be represented as below.

  Notice that the body of the Lambda *does not depend on the Lambda's
  parameter*, so that if we were actually executing this call the argument will
  be thrown away.

  All leaf nodes are instances of `building_blocks.Data`.

                            Call
                           /    \
                 Lambda('arg')   Data('k')
                     |
                   Block('y','z')-------------
                  /                          |
  ['y'=Data('a'),'z'=Data('b')]              |
                                           Tuple
                                         /       \
                                   Block('v')     Block('x')-------
                                     / \              |            |
                       ['v'=Selection]   Data('g') ['x'=Data('h']  |
                             |                                     |
                             |                                     |
                             |                                 Block('w')
                             |                                   /   \
                           Tuple ------            ['w'=Data('i']     Data('j')
                         /              \
                 Block('t')             Block('u')
                  /     \              /          \
    ['t'=Data('c')]    Data('d') ['u'=Data('e')]  Data('f')


  Postorder traversals:
  If we are reading Data URIs, results of a postorder traversal should be:
  [a, b, c, d, e, f, g, h, i, j, k]

  If we are reading locals declarations, results of a postorder traversal should
  be:
  [t, u, v, w, x, y, z]

  And if we are reading both in an interleaved fashion, results of a postorder
  traversal should be:
  [a, b, c, d, t, e, f, u, g, v, h, i, j, w, x, y, z, k]

  Preorder traversals:
  If we are reading Data URIs, results of a preorder traversal should be:
  [a, b, c, d, e, f, g, h, i, j, k]

  If we are reading locals declarations, results of a preorder traversal should
  be:
  [y, z, v, t, u, x, w]

  And if we are reading both in an interleaved fashion, results of a preorder
  traversal should be:
  [y, z, a, b, v, t, c, d, u, e, f, g, x, h, w, i, j, k]

  Since we are also exposing the ability to hook into variable declarations,
  it is worthwhile considering the order in which variables are assigned in
  this tree. Notice that this order maps neither to preorder nor to postorder
  when purely considering the nodes of the tree above. This would be:
  [arg, y, z, t, u, v, x, w]

  Returns:
    An instance of `building_blocks.ComputationBuildingBlock`
    satisfying the description above.
  """
  data_c = building_blocks.Data('c', tf.float32)
  data_d = building_blocks.Data('d', tf.float32)
  left_most_leaf = building_blocks.Block([('t', data_c)], data_d)

  data_e = building_blocks.Data('e', tf.float32)
  data_f = building_blocks.Data('f', tf.float32)
  center_leaf = building_blocks.Block([('u', data_e)], data_f)
  inner_tuple = building_blocks.Struct([left_most_leaf, center_leaf])

  selected = building_blocks.Selection(inner_tuple, index=0)
  data_g = building_blocks.Data('g', tf.float32)
  middle_block = building_blocks.Block([('v', selected)], data_g)

  data_i = building_blocks.Data('i', tf.float32)
  data_j = building_blocks.Data('j', tf.float32)
  right_most_endpoint = building_blocks.Block([('w', data_i)], data_j)

  data_h = building_blocks.Data('h', tf.int32)
  right_child = building_blocks.Block([('x', data_h)], right_most_endpoint)

  result = building_blocks.Struct([middle_block, right_child])
  data_a = building_blocks.Data('a', tf.float32)
  data_b = building_blocks.Data('b', tf.float32)
  whimsy_outer_block = building_blocks.Block([('y', data_a), ('z', data_b)],
                                             result)
  whimsy_lambda = building_blocks.Lambda('arg', tf.float32, whimsy_outer_block)
  whimsy_arg = building_blocks.Data('k', tf.float32)
  called_lambda = building_blocks.Call(whimsy_lambda, whimsy_arg)

  return called_lambda


def _stamp_value_into_graph(value, type_signature, graph):
  """Stamps `value` in `graph` as an object of type `type_signature`.

  Args:
    value: An value to stamp.
    type_signature: An instance of `computation_types.Type`.
    graph: The graph to stamp in.

  Returns:
    A Python object made of tensors stamped into `graph`, `tf.data.Dataset`s,
    and `structure.Struct`s that structurally corresponds to the
    value passed at input.
  """
  py_typecheck.check_type(type_signature, computation_types.Type)
  py_typecheck.check_type(graph, tf.Graph)
  if value is None:
    return None
  if type_signature.is_tensor():
    if isinstance(value, np.ndarray):
      value_type = computation_types.TensorType(
          tf.dtypes.as_dtype(value.dtype), tf.TensorShape(value.shape))
      type_signature.check_assignable_from(value_type)
      with graph.as_default():
        return tf.constant(value)
    else:
      with graph.as_default():
        return tf.constant(
            value, dtype=type_signature.dtype, shape=type_signature.shape)
  elif type_signature.is_struct():
    if isinstance(value, (list, dict)):
      value = structure.from_container(value)
    stamped_elements = []
    named_type_signatures = structure.to_elements(type_signature)
    for (name, type_signature), element in zip(named_type_signatures, value):
      stamped_element = _stamp_value_into_graph(element, type_signature, graph)
      stamped_elements.append((name, stamped_element))
    return structure.Struct(stamped_elements)
  elif type_signature.is_sequence():
    return tensorflow_utils.make_data_set_from_elements(graph, value,
                                                        type_signature.element)
  else:
    raise NotImplementedError(
        'Unable to stamp a value of type {} in graph.'.format(type_signature))


# TODO(b/139439722): Consolidate implementation to run a TF comp with an arg.
def run_tensorflow(computation_proto, arg=None):
  """Runs a TensorFlow computation with argument `arg`.

  Args:
    computation_proto: An instance of `pb.Computation`.
    arg: The argument to invoke the computation with, or None if the computation
      does not specify a parameter type and does not expects one.

  Returns:
    The result of the computation.
  """
  with tf.Graph().as_default() as graph:
    type_signature = type_serialization.deserialize_type(computation_proto.type)
    if type_signature.parameter is not None:
      stamped_arg = _stamp_value_into_graph(arg, type_signature.parameter,
                                            graph)
    else:
      stamped_arg = None
    init_op, result = tensorflow_utils.deserialize_and_call_tf_computation(
        computation_proto, stamped_arg, graph)
  with tf.compat.v1.Session(graph=graph) as sess:
    if init_op:
      sess.run(init_op)
    result = tensorflow_utils.fetch_value_in_session(sess, result)
  return result
