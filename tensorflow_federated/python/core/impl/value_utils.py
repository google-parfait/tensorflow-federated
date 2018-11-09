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
"""Utilities for working with abstractions defined in values.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import values


def expand_tuple(arg):
  """Expands a value of a tuple type into an instance of values.Tuple.

  Args:
    arg: An instance of value_base.Value, or something convertible to it, that
      has a TFF type of a named tuple.

  Returns:
    Either 'arg' itself if it's already a values.Tuple, or a new instance of
    values.Tuple with embedded instances of value.Selection for all elements
    of 'arg'.
  """
  arg = values.to_value(arg)
  if isinstance(arg, values.Tuple):
    return arg
  else:
    py_typecheck.check_type(arg.type_signature, types.NamedTupleType)
    return values.Tuple([
        (name, getattr(arg, name)) if name else (None, arg[index])
        for index, (name, _) in enumerate(arg.type_signature.elements)])


def zero_or_one_arg_func_to_lambda(func, parameter_name, parameter_type):
  """Converts a zero- or one-argument 'func' into a corresponding values.Lambda.

  Args:
    func: A function with 0 or 1 arguments that contains orchestration logic,
      i.e., that expects zero or one values_base.Value and returns a result
      conveetible to the same.
    parameter_name: The name of the parameter, or None if there is't any.
    parameter_type: The TFF type of the parameter, or None if there's none.

  Returns:
    An instance of values.Lambda containing the orchestration logic from 'func'.

  Raises:
    ValueError: if 'func' is incompatible with 'parameter_type'.
  """
  py_typecheck.check_callable(func)
  parameter_type = types.to_type(parameter_type)
  if parameter_type:
    arg = values.Reference(parameter_name, parameter_type)
    if isinstance(parameter_type, types.NamedTupleType):
      arg = expand_tuple(arg)
    result = func(arg)
  else:
    result = func()
  return values.Lambda(parameter_name, parameter_type, values.to_value(result))
