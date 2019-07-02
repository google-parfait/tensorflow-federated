# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""An executor that understands lambda expressions and related abstractions."""

import asyncio

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_value_base
from tensorflow_federated.python.core.impl import transformations
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.core.impl import type_utils


class LambdaExecutorScope(object):
  """Represents a naming scope for computations in the lambda executor."""

  def __init__(self, symbols, parent=None):
    """Constructs a new scope.

    Args:
      symbols: A dict of symbols available in this scope, with keys being
        strings, and values being instances of `LambdaExecutorValue`.
      parent: The parent scope, or `None` if this is the root.
    """
    py_typecheck.check_type(symbols, dict)
    for k, v in symbols.items():
      py_typecheck.check_type(k, str)
      py_typecheck.check_type(v, LambdaExecutorValue)
    if parent is not None:
      py_typecheck.check_type(parent, LambdaExecutorScope)
    self._parent = parent
    self._symbols = {k: v for k, v in symbols.items()}

  def resolve_reference(self, name):
    """Resolves the given reference `name` in this scope.

    Args:
      name: The string name to resolve.

    Returns:
      An instance of `LambdaExecutorValue` corresponding to this name.

    Raises:
      ValueError: If the name cannot be resolved.
    """
    py_typecheck.check_type(name, str)
    value = self._symbols.get(str(name))
    if value is not None:
      return value
    elif self._parent is not None:
      return self._parent.resolve_reference(name)
    else:
      raise ValueError(
          'The name \'{}\' is not defined in this scope.'.format(name))


class LambdaExecutorValue(executor_value_base.ExecutorValue):
  """Represents a value embedded in the lambda executor."""

  def __init__(self, value, scope=None, type_spec=None):
    """Creates an instance of a value embedded in a lambda executor.

    The internal representation of a value can take one of the following
    supported forms:

    * An instance of `executor_value_base.ExecutorValue` that represents a
      value embedded in the target executor (functional or non-functional).

    * An as-yet unprocessed instance of `pb.Computation` that represents a
      function yet to be invoked (always a value of a functional type; any
      non-functional constructs should be processed on the fly).

    * A coroutine callable in Python that accepts a single argument that must
      be an instance of `LambdaExecutorValue` (or `None`), and that returns a
      result that is also an instance of `LambdaExecutorValue`. The associated
      type signature is always functional.

    * A single-level tuple (`anonymous_tuple.AnonymousTuple`) of instances
      of this class (of any of the supported forms listed here).

    Args:
      value: The internal representation of a value, as specified above.
      scope: An optional scope for computations. Only allowed if `value` is an
        unprocessed instance of `pb.Computation`, otherwise it must be `None`
        (the scope is meaningless in other cases).
      type_spec: An optional type signature, only allowed if `value` is a
        callable that represents a function (in which case it must be an
        instance of `computation_types.FunctionType`), otherwise it  must be
        `None` (the type is implied in other cases).
    """
    if isinstance(value, executor_value_base.ExecutorValue):
      py_typecheck.check_none(scope)
      py_typecheck.check_none(type_spec)
      type_spec = value.type_signature
    elif isinstance(value, pb.Computation):
      if scope is not None:
        py_typecheck.check_type(scope, LambdaExecutorScope)
      py_typecheck.check_none(type_spec)
      type_spec = type_utils.get_function_type(
          type_serialization.deserialize_type(value.type))
    elif callable(value):
      py_typecheck.check_none(scope)
      py_typecheck.check_type(type_spec, computation_types.FunctionType)
    else:
      py_typecheck.check_type(value, anonymous_tuple.AnonymousTuple)
      py_typecheck.check_none(scope)
      py_typecheck.check_none(type_spec)
      type_elements = []
      for k, v in anonymous_tuple.to_elements(value):
        py_typecheck.check_type(v, LambdaExecutorValue)
        type_elements.append((k, v.type_signature))
      type_spec = computation_types.NamedTupleType([
          (k, v) if k is not None else v for k, v in type_elements
      ])
    self._value = value
    self._scope = scope
    self._type_signature = type_spec

  @property
  def internal_representation(self):
    """Returns a representation of the value embedded in the executor.

    This property is only intended for use by the lambda executor and tests. Not
    for consumption by consumers of the executor interface.
    """
    return self._value

  @property
  def scope(self):
    return self._scope

  @property
  def type_signature(self):
    return self._type_signature

  async def compute(self):
    if isinstance(self._type_signature, computation_types.FunctionType):
      raise TypeError(
          'Materializing a computed value of a functional TFF type {} is not '
          'possible; only non-functional values can be materialized. Did you '
          'perhaps forget to __call__() a function you declared?'.format(
              str(self._type_signature)))
    elif isinstance(self._value, executor_value_base.ExecutorValue):
      return await self._value.compute()
    else:
      # An unprocessed computation or a callable would have had to declare a
      # functional type, so this is the only case left to handle.
      py_typecheck.check_type(self._value, anonymous_tuple.AnonymousTuple)
      elem = anonymous_tuple.to_elements(self._value)
      vals = asyncio.gather(*[v.compute() for _, v in elem])
      return anonymous_tuple.AnonymousTuple(zip([k for k, _ in elem], vals))


class LambdaExecutor(executor_base.Executor):
  """The lambda executor handles lambda expressions and related abstractions.

  WARNING: This executor is only partially implemented, and should not be used.

  This executor understands TFF computation compositional constructs, including
  lambdas, blocks, references, calls, tuples, and selections, and orchestrates
  the execution of these constructs, while delegating all the non-compositional
  constructs (tensorflow, intrinsics, data, or placement) to a target executor.

  NOTE: Not all lambda expressions are executed by this lambda executor. If the
  computation contains a call to an instrinsic that takes a functional argument,
  that functional argument is fed in its entirety to the target executor rather
  than being parsed by the lambda executor (since its execution needs to happen
  elsewhere).

  The arguments to be ingested can be either federated computations (those are
  natively interpreted), or whatever other form of arguments are understood by
  the target executor.
  """

  def __init__(self, target_executor):
    """Creates a lambda executor backed by a target executor.

    Args:
      target_executor: An instance of `executor_base.Executor` to which the
        lambda executor delegates all that it cannot handle by itself.
    """
    py_typecheck.check_type(target_executor, executor_base.Executor)
    self._target_executor = target_executor

  async def create_value(self, value, type_spec=None):
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, computation_impl.ComputationImpl):
      return await self.create_value(
          computation_impl.ComputationImpl.get_proto(value),
          type_utils.reconcile_value_with_type_spec(value, type_spec))
    elif isinstance(value, pb.Computation):
      result = LambdaExecutorValue(value)
      type_utils.reconcile_value_with_type_spec(result, type_spec)
      return result
    elif isinstance(type_spec, computation_types.NamedTupleType):
      v_el = anonymous_tuple.to_elements(anonymous_tuple.from_container(value))
      t_el = anonymous_tuple.to_elements(type_spec)
      vals = await asyncio.gather(
          *[self.create_value(v, t) for (_, v), (_, t) in zip(v_el, t_el)])
      return LambdaExecutorValue(
          anonymous_tuple.AnonymousTuple([
              (name, val) for (name, _), val in zip(v_el, vals)
          ]))
    else:
      return LambdaExecutorValue(await self._target_executor.create_value(
          value, type_spec))

  async def create_tuple(self, elements):
    return LambdaExecutorValue(anonymous_tuple.from_container(elements))

  async def create_selection(self, source, index=None, name=None):
    py_typecheck.check_type(source, LambdaExecutorValue)
    py_typecheck.check_type(source.type_signature,
                            computation_types.NamedTupleType)
    source_repr = source.internal_representation
    if isinstance(source_repr, executor_value_base.ExecutorValue):
      return LambdaExecutorValue(await self._target_executor.create_selection(
          source_repr, index=index, name=name))
    else:
      # Any unprocessed computation or a callable would have had to necessarily
      # declare a functional type signature (even if without an argument), so
      # an anonymous tuple is the only case left to handle.
      py_typecheck.check_type(source_repr, anonymous_tuple.AnonymousTuple)
      if index is not None:
        if name is not None:
          raise ValueError('Cannot specify both index and name for selection.')
        return source_repr[index]
      elif name is not None:
        return getattr(source_repr, name)
      else:
        raise ValueError('Either index or name must be present for selection.')

  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, LambdaExecutorValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    param_type = comp.type_signature.parameter
    if param_type is not None:
      py_typecheck.check_type(arg, LambdaExecutorValue)
      if not type_utils.is_assignable_from(param_type, arg.type_signature):
        arg_type = type_utils.get_argument_type(arg.type_signature)
        type_utils.check_assignable_from(param_type, arg_type)
        return await self.create_call(comp, await self.create_call(arg))
    else:
      py_typecheck.check_none(arg)
    comp_repr = comp.internal_representation
    if isinstance(comp_repr, executor_value_base.ExecutorValue):
      return LambdaExecutorValue(await self._target_executor.create_call(
          comp_repr, await self._delegate(arg) if arg is not None else None))
    elif callable(comp_repr):
      return await comp_repr(arg)
    else:
      # An anonymous tuple could not possibly have a functional type signature,
      # so this is the only case left to handle.
      py_typecheck.check_type(comp_repr, pb.Computation)
      eval_result = await self._evaluate(comp_repr, comp.scope)
      py_typecheck.check_type(eval_result, LambdaExecutorValue)
      if arg is not None:
        py_typecheck.check_type(eval_result.type_signature,
                                computation_types.FunctionType)
        type_utils.check_assignable_from(eval_result.type_signature.parameter,
                                         arg.type_signature)
        return await self.create_call(eval_result, arg)
      elif isinstance(eval_result.type_signature,
                      computation_types.FunctionType):
        return await self.create_call(eval_result, arg=None)
      else:
        return eval_result

  async def _delegate(self, value):
    """Delegates the entirety of `value` to the target executor.

    Args:
      value: An instance of `LambdaExecutorValue` (of any valid form except for
        a callable, which cannot be delegated, but that should only ever be
        constructed at the evaluation time).

    Returns:
      An instance of `executor_value_base.ExecutorValue` that represents a
      value embedded in the target executor.

    Raises:
      RuntimeError: Upon encountering a request to delegate a computation that
        is in a form that cannot be delegated.
    """
    py_typecheck.check_type(value, LambdaExecutorValue)
    value_repr = value.internal_representation
    if isinstance(value_repr, executor_value_base.ExecutorValue):
      return value_repr
    elif isinstance(value_repr, anonymous_tuple.AnonymousTuple):
      elem = anonymous_tuple.to_elements(value_repr)
      vals = await asyncio.gather(*[self._delegate(v) for _, v in elem])
      return await self._target_executor.create_tuple(
          anonymous_tuple.AnonymousTuple(list(zip([k for k, _ in elem], vals))))
    elif callable(value_repr):
      raise RuntimeError(
          'Cannot delegate a callable to a target executor; it appears that '
          'the internal computation structure has been evaluated too deeply '
          '(this is an internal error that represents a bug in the runtime).')
    else:
      py_typecheck.check_type(value_repr, pb.Computation)

      # TODO(b/134543154): This is the place to check for the computation we
      # are about to push down to the target executor making references to
      # something declared outside of its scope, in which case we'll have to
      # do a little bit more work to plumb things through.

      _check_no_unbound_references(value_repr)
      return await self._target_executor.create_value(value_repr,
                                                      value.type_signature)

  async def _evaluate(self, comp, scope=None):
    """Evaluates or partially evaluates `comp` in `scope`.

    Args:
      comp: An instance of `pb.Computation` to process.
      scope: An optional `LambdaExecutorScope` to process it in, or `None` if
        there's no surrounding scope (the computation is self-contained).

    Returns:
      An instance of `LambdaExecutorValue` that isn't unprocessed (i.e., the
      internal representation directly in it isn't simply `comp` or any
      other instance of`pb.Computation`). The result, however, does not have,
      and often won't be processed completely; it suffices for this function
      to make only partial progress.
    """
    py_typecheck.check_type(comp, pb.Computation)
    if scope is not None:
      py_typecheck.check_type(scope, LambdaExecutorScope)
    which_computation = comp.WhichOneof('computation')
    if which_computation in ['tensorflow', 'intrinsic', 'data', 'placement']:
      return LambdaExecutorValue(await self._target_executor.create_value(
          comp,
          type_utils.get_function_type(
              type_serialization.deserialize_type(comp.type))))
    elif which_computation == 'lambda':

      def _make_comp_fn(scope, result, name, type_spec):

        async def _comp_fn(arg):
          return await self._evaluate(result,
                                      LambdaExecutorScope({name: arg}, scope))

        return LambdaExecutorValue(_comp_fn, type_spec=type_spec)

      comp_lambda = getattr(comp, 'lambda')
      type_spec = type_utils.get_function_type(
          type_serialization.deserialize_type(comp.type))
      return _make_comp_fn(scope, comp_lambda.result,
                           comp_lambda.parameter_name, type_spec)
    elif which_computation == 'reference':
      return scope.resolve_reference(comp.reference.name)
    elif which_computation == 'call':
      if comp.call.argument.WhichOneof('computation') is not None:
        arg = LambdaExecutorValue(comp.call.argument, scope=scope)
      else:
        arg = None
      return await self.create_call(
          LambdaExecutorValue(comp.call.function, scope=scope), arg=arg)
    elif which_computation == 'selection':
      which_selection = comp.selection.WhichOneof('selection')
      return await self.create_selection(
          await self.create_call(
              LambdaExecutorValue(comp.selection.source, scope=scope)),
          **{which_selection: getattr(comp.selection, which_selection)})
    elif which_computation == 'tuple':
      names = [str(e.name) if e.name else None for e in comp.tuple.element]
      values = []
      for e in comp.tuple.element:
        val = LambdaExecutorValue(e.value, scope=scope)
        if (isinstance(val.type_signature, computation_types.FunctionType) and
            val.type_signature.parameter is None):
          val = self.create_call(val)
        else:

          async def _async_identity(x):
            return x

          val = _async_identity(val)
        values.append(val)
      values = await asyncio.gather(*values)
      return await self.create_tuple(
          anonymous_tuple.AnonymousTuple(list(zip(names, values))))
    elif which_computation == 'block':
      for loc in comp.block.local:
        value = LambdaExecutorValue(loc.value, scope)
        scope = LambdaExecutorScope({loc.name: value}, scope)
      return await self._evaluate(comp.block.result, scope)
    else:
      raise NotImplementedError(
          'Unsupported computation type "{}".'.format(which_computation))


def _check_no_unbound_references(comp):
  """Checks that `comp` has no unbound references.

  This is a temporary helper function, to be removed once we provide a more
  complete support.

  Args:
    comp: An instance of `pb.Computation` to check.

  Raises:
    ValueError: If `comp` has unbound references.
  """
  py_typecheck.check_type(comp, pb.Computation)
  blk = computation_building_blocks.ComputationBuildingBlock.from_proto(comp)
  unbound_map = transformations.get_map_of_unbound_references(blk)
  unbound_refs = unbound_map[blk]
  if unbound_refs:
    raise ValueError(
        'The computation contains unbound references: {}.'.format(unbound_refs))
