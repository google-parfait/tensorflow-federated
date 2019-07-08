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

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import transformation_utils


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
    value_type: The type of the parameter.
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


def create_identity_block_with_dummy_data(variable_name):
  r"""Returns an identity block with a dummy `Data` computation."""
  data = computation_building_blocks.Data('data', tf.int32)
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


def count(comp, predicate=None):
  """Returns the number of computations in `comp` matching `predicate`.

  Args:
    comp: The computation to test.
    predicate: A Python function that takes a computation as a parameter and
      returns a boolean value.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  counter = [0]

  def _function(comp):
    if predicate is None or predicate(comp):
      counter[0] += 1
    return comp, False

  transformation_utils.transform_postorder(comp, _function)
  return counter[0]


def count_types(comp, types):
  return count(comp, lambda x: isinstance(x, types))
