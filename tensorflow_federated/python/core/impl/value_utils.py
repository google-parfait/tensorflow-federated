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
"""Utilities file for functions with TFF `Value`s as inputs and outputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import value_impl


def get_curried(fn):
  """Returns a curried version of function `fn` that takes a parameter tuple.

  For functions `fn` of types <T1,T2,....,Tn> -> U, the result is a function
  of the form T1 -> (T2 -> (T3 -> .... (Tn -> U) ... )).

  NOTE: No attempt is made at avoiding naming conflicts in cases where `fn`
  contains references. The arguments of the curriend function are named `argN`
  with `N` starting at 0.

  Args:
    fn: A value of a functional TFF type.

  Returns:
    A value that represents the curried form of `fn`.
  """
  py_typecheck.check_type(fn, value_base.Value)
  py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
  py_typecheck.check_type(fn.type_signature.parameter,
                          computation_types.NamedTupleType)
  param_elements = anonymous_tuple.to_elements(fn.type_signature.parameter)
  references = []
  for idx, (_, elem_type) in enumerate(param_elements):
    references.append(
        computation_building_blocks.Reference('arg{}'.format(idx), elem_type))
  result = computation_building_blocks.Call(
      value_impl.ValueImpl.get_comp(fn),
      computation_building_blocks.Tuple(references))
  for ref in references[::-1]:
    result = computation_building_blocks.Lambda(ref.name, ref.type_signature,
                                                result)
  return value_impl.ValueImpl(result,
                              value_impl.ValueImpl.get_context_stack(fn))
