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
"""Utility functions for slicing and dicing intrinsics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import type_constructors
from tensorflow_federated.python.core.impl import value_impl


def zero_for(type_spec, context_stack):
  """Constructs ZERO intrinsic of TFF type `type_spec`.

  Args:
    type_spec: An instance of `types.Type` or something convertible to it.
      intrinsic.
    context_stack: The context stack to use.

  Returns:
    The `ZERO` intrinsic of the same TFF type as that of `val`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  return value_impl.ValueImpl(
      computation_building_blocks.Intrinsic(intrinsic_defs.GENERIC_ZERO.uri,
                                            type_spec), context_stack)


def plus_for(type_spec, context_stack):
  """Constructs PLUS intrinsic that operates on values of TFF type `type_spec`.

  Args:
    type_spec: An instance of `types.Type` or something convertible to it.
      intrinsic.
    context_stack: The context stack to use.

  Returns:
    The `PLUS` intrinsic of type `<T,T> -> T`, where `T` represents `type_spec`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  return value_impl.ValueImpl(
      computation_building_blocks.Intrinsic(
          intrinsic_defs.GENERIC_PLUS.uri,
          type_constructors.binary_op(type_spec)), context_stack)
