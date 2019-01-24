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
"""The implementation of a context to use in building federated computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


class FederatedComputationContext(context_base.Context):
  """The context for building federated computations."""

  def __init__(self, context_stack):
    """Creates this context.

    Args:
      context_stack: The context stack to use.
    """
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    self._context_stack = context_stack

  def ingest(self, val, type_spec):
    val = value_impl.to_value(val, type_spec, self._context_stack)
    type_utils.check_type(val, type_spec)
    return val

  def invoke(self, comp, arg):
    func = value_impl.to_value(comp, None, self._context_stack)
    if isinstance(func.type_signature, computation_types.FunctionType):
      if arg is not None:
        type_utils.check_type(arg, func.type_signature.parameter)
        ret_val = func(arg)
      else:
        ret_val = func()
      type_utils.check_type(ret_val, func.type_signature.result)
      return ret_val
    elif arg is not None:
      raise ValueError(
          'A computation of type {} does not expect any arguments, '
          'but got an argument {}.'.format(str(func.type_signature), str(arg)))
    else:
      return func
