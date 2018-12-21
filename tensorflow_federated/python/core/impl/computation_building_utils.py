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
# limitations under the License.
"""Helpers for creating larger structures out of computating building blocks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import federated_computation_context
from tensorflow_federated.python.core.impl import value_impl


def zero_or_one_arg_func_to_building_block(func, parameter_name, parameter_type,
                                           context_stack):
  """Converts a zero- or one-argument `func` into a computation building block.

  Args:
    func: A function with 0 or 1 arguments that contains orchestration logic,
      i.e., that expects zero or one `values_base.Value` and returns a result
      convertible to the same.
    parameter_name: The name of the parameter, or `None` if there is't any.
    parameter_type: The TFF type of the parameter, or `None` if there's none.
    context_stack: The context stack to use.

  Returns:
    An instance of `computation_building_blocks.ComputationBuildingBlock` that
    contains the logic from `func`.

  Raises:
    ValueError: if `func` is incompatible with `parameter_type`.
  """
  py_typecheck.check_callable(func)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  parameter_type = computation_types.to_type(parameter_type)
  context = federated_computation_context.FederatedComputationContext(
      context_stack)
  with context_stack.install(context):
    if parameter_type is not None:
      result = func(
          value_impl.ValueImpl(
              computation_building_blocks.Reference(
                  parameter_name, parameter_type), context_stack))
    else:
      result = func()
    result = value_impl.to_value(result, None, context_stack)
    result_comp = value_impl.ValueImpl.get_comp(result)
    if parameter_type is None:
      return result_comp
    else:
      return computation_building_blocks.Lambda(parameter_name, parameter_type,
                                                result_comp)
