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
"""Test utils for TFF computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import type_utils


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
    arg: A `computation_building_blocks.ComputationBuildingBlock`.

  Returns:
    A `computation_building_blocks.Call`.
  """
  for fn in functions:
    if not type_utils.is_assignable_from(fn.parameter_type, arg.type_signature):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature)))
    call = computation_building_blocks.Call(fn, arg)
    arg = call
  return call


def create_dummy_block(comp, variable_name, variable_type=tf.int32):
  r"""Returns an identity block.

           Block
          /     \
  [x=data]       Comp

  Args:
    comp: The computation to use as the result.
    variable_name: The name of the variable.
    variable_type: The type of the variable.
  """
  data = computation_building_blocks.Data('data', variable_type)
  return computation_building_blocks.Block([(variable_name, data)], comp)


def create_dummy_called_intrinsic(parameter_name, parameter_type=tf.int32):
  r"""Returns a dummy called intrinsic.

            Call
           /    \
  intrinsic      Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  intrinsic_type = computation_types.FunctionType(parameter_type,
                                                  parameter_type)
  intrinsic = computation_building_blocks.Intrinsic('intrinsic', intrinsic_type)
  ref = computation_building_blocks.Reference(parameter_name, parameter_type)
  return computation_building_blocks.Call(intrinsic, ref)


def create_dummy_called_federated_aggregate(accumulate_parameter_name,
                                            merge_parameter_name,
                                            report_parameter_name):
  r"""Returns a dummy called federated aggregate.

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
  """
  value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
  value = computation_building_blocks.Data('data', value_type)
  zero = computation_building_blocks.Data('data', tf.float32)
  accumulate_type = computation_types.NamedTupleType((tf.float32, tf.int32))
  accumulate_result = computation_building_blocks.Data('data', tf.float32)
  accumulate = computation_building_blocks.Lambda(accumulate_parameter_name,
                                                  accumulate_type,
                                                  accumulate_result)
  merge_type = computation_types.NamedTupleType((tf.float32, tf.float32))
  merge_result = computation_building_blocks.Data('data', tf.float32)
  merge = computation_building_blocks.Lambda(merge_parameter_name, merge_type,
                                             merge_result)
  report_result = computation_building_blocks.Data('data', tf.bool)
  report = computation_building_blocks.Lambda(report_parameter_name, tf.float32,
                                              report_result)
  return computation_constructing_utils.create_federated_aggregate(
      value, zero, accumulate, merge, report)


def create_dummy_called_federated_apply(parameter_name,
                                        parameter_type=tf.int32):
  r"""Returns a dummy called federated apply.

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
  arg = computation_building_blocks.Data('data', arg_type)
  return computation_constructing_utils.create_federated_apply(fn, arg)


def create_dummy_called_federated_broadcast(value_type=tf.int32):
  r"""Returns a dummy called federated broadcast.

                Call
               /    \
  federated_map      data

  Args:
    value_type: The type of the value.
  """
  federated_type = computation_types.FederatedType(value_type,
                                                   placements.SERVER)
  value = computation_building_blocks.Data('data', federated_type)
  return computation_constructing_utils.create_federated_broadcast(value)


def create_dummy_called_federated_map(parameter_name, parameter_type=tf.int32):
  r"""Returns a dummy called federated map.

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
  arg = computation_building_blocks.Data('data', arg_type)
  return computation_constructing_utils.create_federated_map(fn, arg)


def create_dummy_called_federated_map_all_equal(parameter_name,
                                                parameter_type=tf.int32):
  r"""Returns a dummy called federated map.

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
  arg = computation_building_blocks.Data('data', arg_type)
  return computation_constructing_utils.create_federated_map_all_equal(fn, arg)


def create_dummy_called_sequence_map(parameter_name, parameter_type=tf.int32):
  r"""Returns a dummy called sequence map.

               Call
              /    \
  sequence_map      data

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  fn = create_identity_function(parameter_name, parameter_type)
  arg_type = computation_types.SequenceType(parameter_type)
  arg = computation_building_blocks.Data('data', arg_type)
  return computation_constructing_utils.create_sequence_map(fn, arg)


def create_identity_block(variable_name, comp):
  r"""Returns an identity block.

           Block
          /     \
  [x=comp]       Ref(x)

  Args:
    variable_name: The name of the variable.
    comp: The computation to use as the variable.
  """
  ref = computation_building_blocks.Reference(variable_name,
                                              comp.type_signature)
  return computation_building_blocks.Block([(variable_name, comp)], ref)


def create_identity_block_with_dummy_data(variable_name,
                                          variable_type=tf.int32):
  r"""Returns an identity block with a dummy `Data` computation.

           Block
          /     \
  [x=data]       Ref(x)

  Args:
    variable_name: The name of the variable.
    variable_type: The type of the variable.
  """
  data = computation_building_blocks.Data('data', variable_type)
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
  ref = computation_building_blocks.Reference(parameter_name, parameter_type)
  return computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)


def create_lambda_to_dummy_called_intrinsic(parameter_name,
                                            parameter_type=tf.int32):
  r"""Returns a lambda to call a dummy intrinsic.

            Lambda(x)
            |
            Call
           /    \
  intrinsic      Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.
  """
  call = create_dummy_called_intrinsic(
      parameter_name=parameter_name, parameter_type=parameter_type)
  return computation_building_blocks.Lambda(parameter_name, parameter_type,
                                            call)


def create_nested_syntax_tree():
  r"""Constructs computation with explicit ordering for testing traversals.

  The goal of this computation is to exercise each switch
  in transform_postorder_with_symbol_bindings, at least all those that recurse.

  The computation this function constructs can be represented as below.

  Notice that the body of the Lambda *does not depend on the Lambda's
  parameter*, so that if we were actually executing this call the argument will
  be thrown away.

  All leaf nodes are instances of `computation_building_blocks.Data`.

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


  If we are reading Data URIs, results of a postorder traversal should be:
  [a, b, c, d, e, f, g, h, i, j, k]

  If we are reading locals declarations, results of a postorder traversal should
  be:
  [t, u, v, w, x, y, z]

  And if we are reading both in an interleaved fashion, results of a postorder
  traversal should be:
  [a, b, c, d, t, e, f, u, g, v, h, i, j, w, x, y, z, k]

  Since we are also exposing the ability to hook into variable declarations,
  it is worthwhile considering the order in which variables are assigned in
  this tree. Notice that this order maps neither to preorder nor to postorder
  when purely considering the nodes of the tree above. This would be:
  [arg, y, z, t, u, v, x, w]

  Returns:
    An instance of `computation_building_blocks.ComputationBuildingBlock`
    satisfying the description above.
  """
  data_c = computation_building_blocks.Data('c', tf.float32)
  data_d = computation_building_blocks.Data('d', tf.float32)
  left_most_leaf = computation_building_blocks.Block([('t', data_c)], data_d)

  data_e = computation_building_blocks.Data('e', tf.float32)
  data_f = computation_building_blocks.Data('f', tf.float32)
  center_leaf = computation_building_blocks.Block([('u', data_e)], data_f)
  inner_tuple = computation_building_blocks.Tuple([left_most_leaf, center_leaf])

  selected = computation_building_blocks.Selection(inner_tuple, index=0)
  data_g = computation_building_blocks.Data('g', tf.float32)
  middle_block = computation_building_blocks.Block([('v', selected)], data_g)

  data_i = computation_building_blocks.Data('i', tf.float32)
  data_j = computation_building_blocks.Data('j', tf.float32)
  right_most_endpoint = computation_building_blocks.Block([('w', data_i)],
                                                          data_j)

  data_h = computation_building_blocks.Data('h', tf.int32)
  right_child = computation_building_blocks.Block([('x', data_h)],
                                                  right_most_endpoint)

  result = computation_building_blocks.Tuple([middle_block, right_child])
  data_a = computation_building_blocks.Data('a', tf.float32)
  data_b = computation_building_blocks.Data('b', tf.float32)
  dummy_outer_block = computation_building_blocks.Block([('y', data_a),
                                                         ('z', data_b)], result)
  dummy_lambda = computation_building_blocks.Lambda('arg', tf.float32,
                                                    dummy_outer_block)
  dummy_arg = computation_building_blocks.Data('k', tf.float32)
  called_lambda = computation_building_blocks.Call(dummy_lambda, dummy_arg)

  return called_lambda
