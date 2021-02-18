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
from typing import Set, Union

import cachetools

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import type_serialization


def _hash_proto(comp: pb.Computation) -> int:
  """Hash the `pb.Computation` for use as a cache key."""
  return hash(comp.SerializeToString())


class _UnboundRefChecker:
  """Callable implementing memoized checks for unbound references."""

  # TODO(b/149315253): This is the place to check for the computation we
  # are about to push down to the target executor making references to
  # something declared outside of its scope. Notice that it is
  # safe to keep this global cache around, since the check here is for a static
  # property of the proto.

  def __init__(self):
    super().__init__()
    # Note: this must be at least as large as the number of refs in any single
    # computation, otherwise the lookup after the update below might fail.
    ref_cache_size = 10000
    self._evaluated_comps = cachetools.LRUCache(ref_cache_size)

  def __call__(self, proto: pb.Computation) -> Set[str]:
    """Returns the names of any unbound references in `proto`."""
    py_typecheck.check_type(proto, pb.Computation)
    evaluated = self._evaluated_comps.get(_hash_proto(proto))
    if evaluated is not None:
      return evaluated
    tree = building_blocks.ComputationBuildingBlock.from_proto(proto)
    unbound_ref_map = transformation_utils.get_map_of_unbound_references(tree)
    self._evaluated_comps.update(
        {_hash_proto(k.proto): v for k, v in unbound_ref_map.items()})
    return unbound_ref_map[tree]


_unbound_refs = _UnboundRefChecker()


class ReferenceResolvingExecutorScope:
  """Represents a naming scope for computations in the lambda executor."""

  def __init__(self, symbols, parent=None):
    """Constructs a new scope.

    Args:
      symbols: A dict of symbols available in this scope, with keys being
        strings, and values being futures that resolve to instances of
        `ReferenceResolvingExecutorValue`.
      parent: The parent scope, or `None` if this is the root.
    """
    super().__init__()
    py_typecheck.check_type(symbols, dict)
    for k, v in symbols.items():
      py_typecheck.check_type(k, str)
      assert asyncio.isfuture(v)
    if parent is not None:
      py_typecheck.check_type(parent, ReferenceResolvingExecutorScope)
    self._parent = parent
    self._symbols = {k: v for k, v in symbols.items()}

  async def resolve_reference(self, name):
    """Resolves the given reference `name` in this scope.

    Args:
      name: The string name to resolve.

    Returns:
      An instance of `ReferenceResolvingExecutorValue` corresponding to this
      name.

    Raises:
      ValueError: If the name cannot be resolved.
    """
    py_typecheck.check_type(name, str)
    value = self._symbols.get(str(name))
    if value is not None:
      return await value
    elif self._parent is not None:
      return await self._parent.resolve_reference(name)
    else:
      raise ValueError(
          'The name \'{}\' is not defined in this scope.'.format(name))


class ScopedLambda():
  """Represents a lambda value with some attached scope.

  The scope is used to handle variables captured within the lambda.
  Note that lambdas which contain references to captured variables
  must be called by the lambda executor-- they cannot be passed down to
  child executors.

  This is because it isn't possible to replace `pb.Computation` `Reference`s
  with computed values created by the child executor, so we have no way to
  create a callable on the child containing the resolved values.
  """

  def __init__(
      self,
      comp: pb.Computation,
      scope: ReferenceResolvingExecutorScope,
  ):
    super().__init__()
    py_typecheck.check_type(comp, pb.Computation)
    py_typecheck.check_type(scope, ReferenceResolvingExecutorScope)
    self._comp = comp
    self._scope = scope

  @property
  def comp(self) -> pb.Computation:
    return self._comp

  async def invoke(
      self,
      executor: 'ReferenceResolvingExecutor',
      parameter_value: 'ReferenceResolvingExecutorValue',
  ) -> 'ReferenceResolvingExecutorValue':
    """Evaluates the lambda with the provided parameter."""
    scope = self._scope
    comp_lambda = getattr(self._comp, 'lambda')
    if parameter_value is None:
      new_scope = scope
    else:
      parameter_value_future = asyncio.Future()
      parameter_value_future.set_result(parameter_value)
      new_binding = {comp_lambda.parameter_name: parameter_value_future}
      new_scope = ReferenceResolvingExecutorScope(new_binding, scope)
    return await executor._evaluate(comp_lambda.result, new_scope)  # pylint: disable=protected-access


LambdaValueInner = Union[executor_value_base.ExecutorValue, ScopedLambda,
                         structure.Struct]


class ReferenceResolvingExecutorValue(executor_value_base.ExecutorValue):
  """Represents a value embedded in the lambda executor."""

  def __init__(self, value: LambdaValueInner, type_spec=None):
    """Creates an instance of a value embedded in a lambda executor.

    The internal representation of a value can take one of the following
    supported forms:

    * An instance of `executor_value_base.ExecutorValue` that represents a
      value embedded in the target executor (functional or non-functional).

    * A `ScopedLambda` (a `pb.Computation` lambda with some attached scope).

    * A single-level struct (`structure.Struct`) of instances
      of this class (of any of the supported forms listed here).

    Args:
      value: The internal representation of a value, as specified above.
      type_spec: An optional type signature, only allowed if `value` is a
        callable that represents a function (in which case it must be an
        instance of `computation_types.FunctionType`), otherwise it  must be
        `None` (the type is implied in other cases).
    """
    super().__init__()
    if isinstance(value, executor_value_base.ExecutorValue):
      py_typecheck.check_none(type_spec)
      type_spec = value.type_signature
    elif isinstance(value, ScopedLambda):
      py_typecheck.check_type(type_spec, computation_types.FunctionType)
    else:
      py_typecheck.check_type(value, structure.Struct)
      py_typecheck.check_none(type_spec)
      type_elements = []
      for k, v in structure.iter_elements(value):
        py_typecheck.check_type(v, ReferenceResolvingExecutorValue)
        type_elements.append((k, v.type_signature))
      type_spec = computation_types.StructType([
          (k, v) if k is not None else v for k, v in type_elements
      ])
    self._value = value
    self._type_signature = type_spec

  @property
  def internal_representation(self) -> LambdaValueInner:
    """Returns a representation of the value embedded in the executor.

    This property is only intended for use by the lambda executor and tests. Not
    for consumption by consumers of the executor interface.
    """
    return self._value

  @property
  def type_signature(self):
    return self._type_signature

  @tracing.trace
  async def compute(self):
    if self._type_signature.is_function():
      raise TypeError(
          'Materializing a computed value of a functional TFF type {} is not '
          'possible; only non-functional values can be materialized. Did you '
          'perhaps forget to __call__() a function you declared?'.format(
              str(self._type_signature)))
    elif isinstance(self._value, executor_value_base.ExecutorValue):
      return await self._value.compute()
    else:
      # `ScopedLambda` would have had to declare a functional type, so this is
      # the only case left to handle.
      py_typecheck.check_type(self._value, structure.Struct)
      elem = structure.to_elements(self._value)
      vals = await asyncio.gather(*[v.compute() for _, v in elem])
      return structure.Struct(zip([k for k, _ in elem], vals))


class ReferenceResolvingExecutor(executor_base.Executor):
  """The lambda executor handles lambda expressions and related abstractions.

  This executor understands TFF computation compositional constructs, including
  lambdas, blocks, references, calls, structs, and selections, and orchestrates
  the execution of these constructs, while delegating all the non-compositional
  constructs (tensorflow, intrinsics, data, or placement) to a target executor.

  Note: Not all lambda expressions are executed by this lambda executor. If the
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
    super().__init__()
    py_typecheck.check_type(target_executor, executor_base.Executor)
    self._target_executor = target_executor

  def close(self):
    self._target_executor.close()

  @tracing.trace(stats=False)
  async def create_value(self, value, type_spec=None):
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, computation_impl.ComputationImpl):
      return await self.create_value(
          computation_impl.ComputationImpl.get_proto(value),
          executor_utils.reconcile_value_with_type_spec(value, type_spec))
    elif isinstance(value, pb.Computation):
      return await self._evaluate(value)
    elif type_spec is not None and type_spec.is_struct():
      v_el = structure.to_elements(structure.from_container(value))
      vals = await asyncio.gather(
          *[self.create_value(val, t) for (_, val), t in zip(v_el, type_spec)])
      return ReferenceResolvingExecutorValue(
          structure.Struct((name, val) for (name, _), val in zip(v_el, vals)))
    else:
      return ReferenceResolvingExecutorValue(await
                                             self._target_executor.create_value(
                                                 value, type_spec))

  @tracing.trace
  async def create_struct(self, elements):
    return ReferenceResolvingExecutorValue(structure.from_container(elements))

  @tracing.trace
  async def create_selection(self, source, index=None, name=None):
    py_typecheck.check_type(source, ReferenceResolvingExecutorValue)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    source_repr = source.internal_representation
    if isinstance(source_repr, executor_value_base.ExecutorValue):
      return ReferenceResolvingExecutorValue(
          await self._target_executor.create_selection(
              source_repr, index=index, name=name))
    elif isinstance(source_repr, ScopedLambda):
      raise ValueError('Cannot index into a lambda.')
    else:
      py_typecheck.check_type(source_repr, structure.Struct)
      if index is not None:
        if name is not None:
          raise ValueError('Cannot specify both index and name for selection.')
        return source_repr[index]
      elif name is not None:
        return getattr(source_repr, name)
      else:
        raise ValueError('Either index or name must be present for selection.')

  @tracing.trace
  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, ReferenceResolvingExecutorValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    param_type = comp.type_signature.parameter
    if param_type is None:
      py_typecheck.check_none(arg)
    else:
      py_typecheck.check_type(arg, ReferenceResolvingExecutorValue)
      arg_type = arg.type_signature
      if not param_type.is_assignable_from(arg_type):
        raise TypeError('ReferenceResolvingExecutor asked to create call with '
                        'incompatible type specifications. Function '
                        'takes an argument of type {}, but was supplied '
                        'an argument of type {}'.format(param_type, arg_type))

    comp_repr = comp.internal_representation
    if isinstance(comp_repr, executor_value_base.ExecutorValue):
      # `comp` represents a function in the target executor, so we convert the
      # argument to a value inside the target executor and `create_call` on
      # the target executor.
      delegated_arg = await self._embed_value_in_target_exec(
          arg) if arg is not None else None
      return ReferenceResolvingExecutorValue(await
                                             self._target_executor.create_call(
                                                 comp_repr, delegated_arg))
    elif isinstance(comp_repr, ScopedLambda):
      return await comp_repr.invoke(self, arg)
    else:
      raise TypeError(
          'Unexpected type to ReferenceResolvingExecutor create_call: {}'
          .format(type(comp_repr)))

  @tracing.trace(stats=False)
  async def _embed_value_in_target_exec(
      self, value: ReferenceResolvingExecutorValue
  ) -> executor_value_base.ExecutorValue:
    """Inserts a value into the target executor.

    This function is called in order to prepare the argument being passed to a
    `self._target_executor.create_call`, which happens when a non-`Lambda`
    function is passed as the `comp` argument to `create_value` above.

    Args:
      value: An instance of `ReferenceResolvingExecutorValue`.

    Returns:
      An instance of `executor_value_base.ExecutorValue` that represents a value
      embedded in
      the target executor.

    Raises:
      RuntimeError: Upon encountering a request to delegate a computation that
        is in a form that cannot be delegated.
    """
    py_typecheck.check_type(value, ReferenceResolvingExecutorValue)
    value_repr = value.internal_representation
    if isinstance(value_repr, executor_value_base.ExecutorValue):
      return value_repr
    elif isinstance(value_repr, structure.Struct):
      vals = await asyncio.gather(
          *[self._embed_value_in_target_exec(v) for v in value_repr])
      return await self._target_executor.create_struct(
          structure.Struct(
              zip((k for k, _ in structure.iter_elements(value_repr)), vals)))
    else:
      py_typecheck.check_type(value_repr, ScopedLambda)
      # Pull `comp` out of the `ScopedLambda`, asserting that it doesn't
      # reference any scope variables. We don't have a way to replace the
      # references inside the lambda pb.Computation with actual computed values,
      # so we must throw an error in this case.
      unbound_refs = _unbound_refs(value_repr.comp)
      if len(unbound_refs) != 0:  # pylint: disable=g-explicit-length-test
        # Note: "passed to intrinsic" here is an assumption of what the user is
        # doing. Typechecking should reject a lambda passed to Tensorflow code,
        # and intrinsics are the only other functional construct in TFF.
        tree = building_blocks.ComputationBuildingBlock.from_proto(
            value_repr.comp)
        raise RuntimeError(
            'lambda passed to intrinsic contains references to captured '
            'variables. This is not currently supported. For more information, '
            'see b/148685415. '
            'Found references {} in computation {} with type {}'.format(
                unbound_refs, tree, tree.type_signature))
      return await self._target_executor.create_value(value_repr.comp,
                                                      value.type_signature)

  @tracing.trace(stats=False)
  async def _evaluate_to_delegate(
      self,
      comp: pb.Computation,
      scope: ReferenceResolvingExecutorScope,
  ) -> ReferenceResolvingExecutorValue:
    return ReferenceResolvingExecutorValue(
        await self._target_executor.create_value(
            comp, type_serialization.deserialize_type(comp.type)))

  @tracing.trace(stats=False)
  async def _evaluate_lambda(
      self,
      comp: pb.Computation,
      scope: ReferenceResolvingExecutorScope,
  ) -> ReferenceResolvingExecutorValue:
    type_spec = type_serialization.deserialize_type(comp.type)
    return ReferenceResolvingExecutorValue(
        ScopedLambda(comp, scope), type_spec=type_spec)

  @tracing.trace(stats=False)
  async def _evaluate_reference(
      self,
      comp: pb.Computation,
      scope: ReferenceResolvingExecutorScope,
  ) -> ReferenceResolvingExecutorValue:
    return await scope.resolve_reference(comp.reference.name)

  @tracing.trace(stats=False)
  async def _evaluate_call(
      self,
      comp: pb.Computation,
      scope: ReferenceResolvingExecutorScope,
  ) -> ReferenceResolvingExecutorValue:
    func = self._evaluate(comp.call.function, scope=scope)

    async def get_arg():
      if comp.call.argument.WhichOneof('computation') is not None:
        return await self._evaluate(comp.call.argument, scope=scope)
      else:
        return None

    func, arg = await asyncio.gather(func, get_arg())
    return await self.create_call(func, arg=arg)

  @tracing.trace(stats=False)
  async def _evaluate_selection(
      self,
      comp: pb.Computation,
      scope: ReferenceResolvingExecutorScope,
  ) -> ReferenceResolvingExecutorValue:
    which_selection = comp.selection.WhichOneof('selection')
    source = await self._evaluate(comp.selection.source, scope=scope)
    return await self.create_selection(
        source, **{which_selection: getattr(comp.selection, which_selection)})

  @tracing.trace(stats=False)
  async def _evaluate_struct(
      self,
      comp: pb.Computation,
      scope: ReferenceResolvingExecutorScope,
  ) -> ReferenceResolvingExecutorValue:
    names = [str(e.name) if e.name else None for e in comp.struct.element]
    values = [self._evaluate(e.value, scope=scope) for e in comp.struct.element]
    values = await asyncio.gather(*values)
    return await self.create_struct(structure.Struct(zip(names, values)))

  @tracing.trace(stats=False)
  async def _evaluate_block(
      self,
      comp: pb.Computation,
      scope: ReferenceResolvingExecutorScope,
  ) -> ReferenceResolvingExecutorValue:
    for loc in comp.block.local:
      value = asyncio.ensure_future(self._evaluate(loc.value, scope))
      scope = ReferenceResolvingExecutorScope({loc.name: value}, scope)
    return await self._evaluate(comp.block.result, scope)

  async def _evaluate(
      self,
      comp: pb.Computation,
      scope=ReferenceResolvingExecutorScope({}),
  ) -> ReferenceResolvingExecutorValue:
    """Transforms `pb.Computation` into a `ReferenceResolvingExecutorValue`.

    Args:
      comp: An instance of `pb.Computation` to process.
      scope: A `ReferenceResolvingExecutorScope` to process it in. If
        omitted,defaults to an empty scope.

    Returns:
      An instance of `ReferenceResolvingExecutorValue`.
    """
    py_typecheck.check_type(comp, pb.Computation)
    py_typecheck.check_type(scope, ReferenceResolvingExecutorScope)
    which_computation = comp.WhichOneof('computation')
    if which_computation in [
        'tensorflow', 'intrinsic', 'data', 'placement', 'xla'
    ]:
      # nothing interesting here-- forward the creation to the child executor
      return await self._evaluate_to_delegate(comp, scope)
    elif which_computation == 'lambda':
      return await self._evaluate_lambda(comp, scope)
    elif which_computation == 'reference':
      return await self._evaluate_reference(comp, scope)
    elif which_computation == 'call':
      return await self._evaluate_call(comp, scope)
    elif which_computation == 'selection':
      return await self._evaluate_selection(comp, scope)
    elif which_computation == 'struct':
      return await self._evaluate_struct(comp, scope)
    elif which_computation == 'block':
      return await self._evaluate_block(comp, scope)
    else:
      raise NotImplementedError(
          'Unsupported computation type "{}".'.format(which_computation))
