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

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl import value_impl
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks


def get_curried(fn):
  """Returns a curried version of function `fn` that takes a parameter tuple.

  For functions `fn` of types <T1,T2,....,Tn> -> U, the result is a function
  of the form T1 -> (T2 -> (T3 -> .... (Tn -> U) ... )).

  Note: No attempt is made at avoiding naming conflicts in cases where `fn`
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
    references.append(building_blocks.Reference('arg{}'.format(idx), elem_type))
  result = building_blocks.Call(
      value_impl.ValueImpl.get_comp(fn), building_blocks.Tuple(references))
  for ref in references[::-1]:
    result = building_blocks.Lambda(ref.name, ref.type_signature, result)
  return value_impl.ValueImpl(result,
                              value_impl.ValueImpl.get_context_stack(fn))


def ensure_federated_value(value, placement=None, label=None):
  """Ensures `value` is a federated value placed at `placement`.

  If `value` is not a `computation_types.FederatedType` but is a
  `computation_types.NamedTupleType` that can be converted via `federated_zip`
  to a `computation_types.FederatedType`, inserts the call to `federated_zip`
  and returns the result. If `value` cannot be converted, raises a TypeError.

  Args:
    value: A `value_impl.ValueImpl` to check and convert to a federated value if
      possible.
    placement: The expected placement. If None, any placement is allowed.
    label: An optional string label that describes `value`.

  Returns:
    The value as a federated value, automatically zipping if necessary.

  Raises:
    TypeError: if `value` is not a `FederatedType` and cannot be converted to
      a `FederatedType` with `federated_zip`.
  """
  py_typecheck.check_type(value, value_impl.ValueImpl)
  if label is not None:
    py_typecheck.check_type(label, str)

  if not value.type_signature.is_federated():
    comp = value_impl.ValueImpl.get_comp(value)
    context_stack = value_impl.ValueImpl.get_context_stack(value)
    try:
      zipped = building_block_factory.create_federated_zip(comp)
    except (TypeError, ValueError):
      raise TypeError(
          'The {l} must be a FederatedType or implicitly convertible '
          'to a FederatedType (got a {t}).'.format(
              l=label if label else 'value', t=comp.type_signature))
    value = value_impl.ValueImpl(zipped, context_stack)

  if placement and value.type_signature.placement is not placement:
    raise TypeError(
        'The {} should be placed at {}, but it is placed at {}.'.format(
            label if label else 'value', placement,
            value.type_signature.placement))

  return value
