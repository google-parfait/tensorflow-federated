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

import six

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


class FederatedComputationContext(context_base.Context):
  """The context for building federated computations."""

  def __init__(self, context_stack, suggested_name=None, parent=None):
    """Creates this context.

    Args:
      context_stack: The context stack to use.
      suggested_name: The optional suggested name of the context, a string. It
        may be modified to make it different from the names of any of the
        ancestors on the context stack.
      parent: The optional parent context. If not `None`, it must be an instance
        of `FederatedComputationContext`.
    """
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    if suggested_name:
      py_typecheck.check_type(suggested_name, six.string_types)
      suggested_name = str(suggested_name)
    else:
      suggested_name = 'FEDERATED'
    if parent is not None:
      py_typecheck.check_type(parent, FederatedComputationContext)
    ancestor = parent
    ancestor_names = set()
    while ancestor is not None:
      ancestor_names.add(ancestor.name)
      ancestor = ancestor.parent
    name = suggested_name
    name_count = 0
    while name in ancestor_names:
      name_count = name_count + 1
      name = '{}_{}'.format(suggested_name, str(name_count))
    self._context_stack = context_stack
    self._parent = parent
    self._name = name

  @property
  def name(self):
    return self._name

  @property
  def parent(self):
    return self._parent

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
