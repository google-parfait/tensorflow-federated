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

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.context_stack import symbol_binding_context
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis


class FederatedComputationContext(
    symbol_binding_context.SymbolBindingContext[
        building_blocks.ComputationBuildingBlock,
        building_blocks.Reference,
    ]
):
  """The context for building federated computations.

  This context additionally holds a list of symbols which are bound to
  `building_block.ComputationBuildingBlocks` during construction of
  `tff.Values`, and which respect identical semantics to the binding of locals
  in `building_blocks.Blocks`.

  Any `tff.Value` constructed in this context may add such a symbol binding,
  and thereafter refer to the returned reference in place of the bound
  computation. It is then the responsibility of the installer of this context
  to ensure that the symbols bound during the `tff.Value` construction process
  are appropriately packaged in the result.
  """

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
      py_typecheck.check_type(suggested_name, str)
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
      name = '{}_{}'.format(suggested_name, name_count)
    self._context_stack = context_stack
    self._parent = parent
    self._name = name
    self._symbol_bindings = []
    self._next_symbol_val = 0

  @property
  def name(self):
    return self._name

  @property
  def parent(self):
    return self._parent

  def bind_computation_to_reference(
      self, comp: building_blocks.ComputationBuildingBlock
  ) -> building_blocks.Reference:
    """Binds a computation to a symbol, returns a reference to this binding."""
    name = 'fc_{name}_symbol_{val}'.format(
        name=self._name, val=self._next_symbol_val
    )
    self._next_symbol_val += 1
    self._symbol_bindings.append((name, comp))
    ref = building_blocks.Reference(name, comp.type_signature)
    return ref

  @property
  def symbol_bindings(
      self,
  ) -> list[tuple[str, building_blocks.ComputationBuildingBlock]]:
    return self._symbol_bindings

  def invoke(self, comp, arg):
    fn = value_impl.to_value(comp, type_spec=None)
    tys = fn.type_signature
    py_typecheck.check_type(tys, computation_types.FunctionType)
    if arg is not None:
      if tys.parameter is None:  # pytype: disable=attribute-error
        raise ValueError(
            'A computation of type {} does not expect any arguments, but got '
            'an argument {}.'.format(tys, arg)
        )
      arg = value_impl.to_value(
          arg,
          type_spec=tys.parameter,  # pytype: disable=attribute-error
          zip_if_needed=True,
      )
      type_analysis.check_type(arg, tys.parameter)  # pytype: disable=attribute-error
      ret_val = fn(arg)
    else:
      if tys.parameter is not None:  # pytype: disable=attribute-error
        raise ValueError(
            'A computation of type {} expects an argument of type {}, but got '
            ' no argument.'.format(tys, tys.parameter)  # pytype: disable=attribute-error
        )
      ret_val = fn()
    type_analysis.check_type(ret_val, tys.result)  # pytype: disable=attribute-error
    return ret_val
