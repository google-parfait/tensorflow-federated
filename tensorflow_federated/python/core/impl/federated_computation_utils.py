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
# limitations under the License.
"""Helpers for creating larger structures out of computating building blocks."""

from typing import Any, Optional

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import federated_computation_context
from tensorflow_federated.python.core.impl import value_impl
from tensorflow_federated.python.core.impl.compiler import building_blocks


def zero_or_one_arg_fn_to_building_block(
    fn,
    parameter_name: Optional[str],
    parameter_type: Optional[Any],
    context_stack: context_stack_base.ContextStack,
    suggested_name: Optional[str] = None,
) -> building_blocks.ComputationBuildingBlock:
  """Converts a zero- or one-argument `fn` into a computation building block.

  Args:
    fn: A function with 0 or 1 arguments that contains orchestration logic,
      i.e., that expects zero or one `values_base.Value` and returns a result
      convertible to the same.
    parameter_name: The name of the parameter, or `None` if there is't any.
    parameter_type: The TFF type of the parameter, or `None` if there's none.
    context_stack: The context stack to use.
    suggested_name: The optional suggested name to use for the federated context
      that will be used to serialize this function's body (ideally the name of
      the underlying Python function). It might be modified to avoid conflicts.

  Returns:
    An instance of `building_blocks.ComputationBuildingBlock` that
    contains the logic from `fn`.

  Raises:
    ValueError: if `fn` is incompatible with `parameter_type`.
  """
  py_typecheck.check_callable(fn)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  if suggested_name is not None:
    py_typecheck.check_type(suggested_name, str)
  parameter_type = computation_types.to_type(parameter_type)
  if isinstance(context_stack.current,
                federated_computation_context.FederatedComputationContext):
    parent_context = context_stack.current
  else:
    parent_context = None
  context = federated_computation_context.FederatedComputationContext(
      context_stack, suggested_name=suggested_name, parent=parent_context)
  if parameter_name is not None:
    py_typecheck.check_type(parameter_name, str)
    parameter_name = '{}_{}'.format(context.name, str(parameter_name))
  with context_stack.install(context):
    if parameter_type is not None:
      result = fn(
          value_impl.ValueImpl(
              building_blocks.Reference(parameter_name, parameter_type),
              context_stack))
    else:
      result = fn()
    if result is None:
      raise ValueError(
          'The function defined on line {} of file {} has returned a '
          '`NoneType`, but all TFF functions must return some non-`None` '
          'value.'.format(fn.__code__.co_firstlineno, fn.__code__.co_filename))
    result = value_impl.to_value(result, None, context_stack)
    result_comp = value_impl.ValueImpl.get_comp(result)
    return building_blocks.Lambda(parameter_name, parameter_type, result_comp)
