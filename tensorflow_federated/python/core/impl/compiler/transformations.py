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
"""A library of composite transformations.

A composite transformation is one that applies multiple atomic transformation to
an AST either pointwise or serially.
"""

import collections
from collections.abc import Collection, Sequence

import attr

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.types import computation_types


def to_call_dominant(
    comp: building_blocks.ComputationBuildingBlock,
) -> building_blocks.ComputationBuildingBlock:
  """Transforms local (non-federated) computations into call-dominant form.

  Args:
    comp: A computation to transform.

  Returns:
    A transformed but semantically-equivalent `comp`. The resulting `comp` will
    be in CDF (call-dominant form), as defined by the following CFG:

    External -> Intrinsic | Data | Compiled Computation |
                Reference(to top-level lambda parameter) |
                Reference(to value outside of `comp`)

    ExternalCall -> Call(External, CDFElem)

    CDFElem ->
       External
     | Reference to a bound call to an External
     | Selection(CDFElem, index)
     | Lambda(Block([bindings for ExternalCalls, CDF))

    CDF ->
       CDFElem
     | Struct(CDF, ...)
     | Lambda(CDF)
  """
  # Top-level comp must be a lambda to ensure that we create a set of bindings
  # immediately under it, as `_build` does for all lambdas.
  global_comp = comp
  name_generator = building_block_factory.unique_name_generator(comp)

  class _Scope:
    """Name resolution scopes which track the creation of new value bindings."""

    def __init__(self, parent=None, bind_to_parent=False):
      """Create a new scope.

      Args:
        parent: An optional parent scope.
        bind_to_parent: If true, `create_bindings` calls will be propagated to
          the parent scope, causing newly-created bindings to be visible at a
          higher level. If false, `create_bindings` will create a new binding in
          this scope. New bindings will be used as locals inside of
          `bindings_to_block_with_result`.
      """
      if parent is None and bind_to_parent:
        raise ValueError('Cannot `bind_to_parent` for `None` parent.')
      self._parent = parent
      self._newly_bound_values = None if bind_to_parent else []
      self._locals = {}

    def resolve(self, name: str):
      if name in self._locals:
        return self._locals[name]
      if self._parent is None:
        return None
      return self._parent.resolve(name)

    def add_local(self, name, value):
      self._locals[name] = value

    def create_binding(self, value):
      """Add a binding to the nearest binding scope."""
      if self._newly_bound_values is None:
        return self._parent.create_binding(value)
      else:
        name = next(name_generator)
        self._newly_bound_values.append((name, value))
        reference = building_blocks.Reference(name, value.type_signature)
        self._locals[name] = reference
        return reference

    def new_child(self):
      return _Scope(parent=self, bind_to_parent=True)

    def new_child_with_bindings(self):
      """Creates a child scope which will hold its own bindings."""
      # NOTE: should always be paired with a call to
      # `bindings_to_block_with_result`.
      return _Scope(parent=self, bind_to_parent=False)

    def bindings_to_block_with_result(self, result):
      # Don't create unnecessary blocks if there aren't any locals.
      if len(self._newly_bound_values) == 0:  # pylint: disable=g-explicit-length-test
        return result
      else:
        return building_blocks.Block(self._newly_bound_values, result)

  def _build(comp, scope):
    """Transforms `comp` to CDF, possibly adding bindings to `scope`."""
    # The structure returned by this function is a generalized version of
    # call-dominant form. This function may result in the patterns specified in
    # the top-level function's docstring.
    if comp.is_reference():
      result = scope.resolve(comp.name)
      if result is None:
        # If `comp.name` is only bound outside of `comp`, we can't resolve it.
        return comp
      return result
    elif comp.is_selection():
      source = _build(comp.source, scope)
      if source.is_struct():
        return source[comp.as_index()]
      return building_blocks.Selection(source, index=comp.as_index())
    elif comp.is_struct():
      elements = []
      for name, value in structure.iter_elements(comp):
        value = _build(value, scope)
        elements.append((name, value))
      return building_blocks.Struct(elements)
    elif comp.is_call():
      function = _build(comp.function, scope)
      argument = None if comp.argument is None else _build(comp.argument, scope)
      if function.is_lambda():
        if argument is not None:
          scope = scope.new_child()
          scope.add_local(function.parameter_name, argument)
        return _build(function.result, scope)
      else:
        return scope.create_binding(building_blocks.Call(function, argument))
    elif comp.is_lambda():
      scope = scope.new_child_with_bindings()
      if comp.parameter_name:
        scope.add_local(
            comp.parameter_name,
            building_blocks.Reference(comp.parameter_name, comp.parameter_type),
        )
      result = _build(comp.result, scope)
      block = scope.bindings_to_block_with_result(result)
      return building_blocks.Lambda(
          comp.parameter_name, comp.parameter_type, block
      )
    elif comp.is_block():
      scope = scope.new_child()
      for name, value in comp.locals:
        scope.add_local(name, _build(value, scope))
      return _build(comp.result, scope)
    elif (
        comp.is_intrinsic()
        or comp.is_data()
        or comp.is_compiled_computation()
        or comp.is_placement()
    ):
      return comp
    else:
      raise ValueError(
          f'Unrecognized computation kind\n{comp}\nin\n{global_comp}'
      )

  scope = _Scope()
  result = _build(comp, scope)
  comp = scope.bindings_to_block_with_result(result)
  for transform in [
      tree_transformations.uniquify_reference_names,
      tree_transformations.remove_unused_block_locals,
  ]:
    comp, _ = transform(comp)
  return comp


def get_normalized_call_dominant_lambda(
    comp: building_blocks.Lambda,
) -> building_blocks.Lambda:
  """Creates normalized call dominant form for a lambda computation.

  Args:
    comp: A computation to normalize.

  Returns:
    A transformed but semantically-equivalent `comp`. The result will be a
    lambda computation in CDF (call-dominant form) and the result component of
    the lambda is guaranteed to be a block.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)

  # Simplify the `comp` before transforming it to call-dominant form.
  comp, _ = tree_transformations.remove_mapped_or_applied_identity(comp)

  # Flatten `comp` to call-dominant form so that we're working with just a
  # linear list of intrinsic calls with no indirection via tupling, selection,
  # blocks, called lambdas, or references.
  comp = to_call_dominant(comp)

  # CDF can potentially return blocks if there are variables not dependent on
  # the top-level parameter. We normalize these away.
  if not comp.is_lambda():
    comp.check_block()
    comp.result.check_lambda()
    if comp.result.result.is_block():
      additional_locals = comp.result.result.locals
      result = comp.result.result.result
    else:
      additional_locals = []
      result = comp.result.result
    # Note: without uniqueness, a local in `comp.locals` could potentially
    # shadow `comp.result.parameter_name`. However, `to_call_dominant`
    # above ensure that names are unique, as it ends in a call to
    # `uniquify_reference_names`.
    comp = building_blocks.Lambda(
        comp.result.parameter_name,
        comp.result.parameter_type,
        building_blocks.Block(comp.locals + additional_locals, result),
    )
  comp.check_lambda()

  # Simple computations with no intrinsic calls won't have a block.
  # Normalize these as well.
  if not comp.result.is_block():
    comp = building_blocks.Lambda(
        comp.parameter_name,
        comp.parameter_type,
        building_blocks.Block([], comp.result),
    )
  comp.result.check_block()

  comp = tree_transformations.normalize_all_equal_bit(comp)
  tree_analysis.check_contains_no_unbound_references(comp)

  return comp


_NamedBinding = tuple[str, building_blocks.ComputationBuildingBlock]


@attr.s
class _IntrinsicDependencies:
  uri_to_locals: dict[str, list[_NamedBinding]] = attr.ib(
      factory=lambda: collections.defaultdict(list)
  )
  locals_dependent_on_intrinsics: list[_NamedBinding] = attr.ib(factory=list)
  locals_not_dependent_on_intrinsics: list[_NamedBinding] = attr.ib(
      factory=list
  )


class NonAlignableAlongIntrinsicError(ValueError):
  pass


def _compute_intrinsic_dependencies(
    intrinsic_uris: set[str],
    parameter_name: str,
    locals_list: list[_NamedBinding],
    comp_repr,
) -> _IntrinsicDependencies:
  """Computes which locals have dependencies on which called intrinsics."""
  result = _IntrinsicDependencies()
  intrinsic_dependencies_for_ref: dict[str, set[str]] = {}
  # Start by marking `comp.parameter_name` as independent of intrinsics.
  intrinsic_dependencies_for_ref[parameter_name] = set()
  for local_name, local_value in locals_list:
    intrinsic_dependencies = set()

    def record_dependencies(subvalue):
      if subvalue.is_reference():
        if subvalue.name not in intrinsic_dependencies_for_ref:
          names = [(n, v.compact_representation()) for n, v in locals_list]
          raise ValueError(
              f"Can't resolve {subvalue.name} in {names}\n"
              f'Current map {intrinsic_dependencies_for_ref}\n'
              f'Original comp: {comp_repr}\n'
          )
        intrinsic_dependencies.update(  # pylint: disable=cell-var-from-loop
            intrinsic_dependencies_for_ref[subvalue.name]
        )
      elif subvalue.is_lambda():
        # We treat the lambdas that appear in CDF (inside intrinsic invocations)
        # as though their parameters are independent of the rest of the
        # computation. Note that we're not careful about saving and then
        # restoring old variables here: this is okay because call-dominant form
        # guarantees unique variable names.
        intrinsic_dependencies_for_ref[subvalue.parameter_name] = set()
      elif subvalue.is_block():
        # Since we're in CDF, the only blocks inside the bodies of arguments
        # are within lambda arguments to intrinsics. We don't need to record
        # dependencies of these since they can't rely on the results of other
        # intrinsics.
        for subvalue_local_name, _ in subvalue.locals:
          intrinsic_dependencies_for_ref[subvalue_local_name] = set()

    tree_analysis.visit_preorder(local_value, record_dependencies)

    # All intrinsic calls are guaranteed to be top-level in call-dominant form.
    is_intrinsic_call = (
        local_value.is_call()
        and local_value.function.is_intrinsic()
        and local_value.function.uri in intrinsic_uris
    )
    if is_intrinsic_call:
      if intrinsic_dependencies:
        raise NonAlignableAlongIntrinsicError(
            'Cannot force-align intrinsics:\n'
            f'Call to intrinsic `{local_value.function.uri}` depends '
            f'on calls to intrinsics:\n`{intrinsic_dependencies}`.'
        )
      intrinsic_dependencies_for_ref[local_name] = set(
          [local_value.function.uri]
      )
      result.uri_to_locals[local_value.function.uri].append(
          (local_name, local_value)
      )
    else:
      intrinsic_dependencies_for_ref[local_name] = intrinsic_dependencies
      if intrinsic_dependencies:
        result.locals_dependent_on_intrinsics.append((local_name, local_value))
      else:
        result.locals_not_dependent_on_intrinsics.append(
            (local_name, local_value)
        )
  return result


@attr.s
class _MergedIntrinsic:
  uri: str = attr.ib()
  args: building_blocks.ComputationBuildingBlock = attr.ib()
  return_type: computation_types.Type = attr.ib()
  unpack_to_locals: list[str] = attr.ib()


def _compute_merged_intrinsics(
    intrinsic_defaults: list[building_blocks.Call],
    uri_to_locals: dict[str, list[_NamedBinding]],
    name_generator,
) -> list[_MergedIntrinsic]:
  """Computes a `_MergedIntrinsic` for each intrinsic in `intrinsic_defaults`.

  Args:
    intrinsic_defaults: A default call to each intrinsic URI to be merged. If no
      entry with this URI is present in `uri_to_locals`, the resulting
      `MergedIntrinsic` will describe only the provided default call.
    uri_to_locals: A mapping from intrinsic URI to locals (name + building_block
      pairs). The building block must be a `Call` to the intrinsic of the given
      URI. The name will be used to bind the result of the merged intrinsic via
      `_MergedIntrinsic.unpack_to_locals`.
    name_generator: A generator used to create unique names.

  Returns:
    A list of `_MergedIntrinsic`s describing, for each intrinsic, how to invoke
    the intrinsic exactly once and unpack the result.
  """
  results = []
  for default_call in intrinsic_defaults:
    uri = default_call.function.uri
    locals_for_uri = uri_to_locals[uri]
    if not locals_for_uri:
      results.append(
          _MergedIntrinsic(
              uri=uri,
              args=default_call.argument,
              return_type=default_call.type_signature,
              unpack_to_locals=[],
          )
      )
    else:
      calls = [local[1] for local in locals_for_uri]
      result_placement = calls[0].type_signature.placement
      result_all_equal = calls[0].type_signature.all_equal
      for call in calls:
        if call.type_signature.all_equal != result_all_equal:
          raise ValueError(
              'Encountered intrinsics to be merged with '
              f'mismatched all_equal bits. Intrinsic of URI {uri} '
              f'first call had all_equal bit {result_all_equal}, '
              'encountered call with all_equal value '
              f'{call.type_signature.all_equal}'
          )
      return_type = computation_types.FederatedType(
          computation_types.StructType(
              [(None, call.type_signature.member) for call in calls]
          ),
          placement=result_placement,
          all_equal=result_all_equal,
      )
      abstract_parameter_type = (
          default_call.function.intrinsic_def().type_signature.parameter
      )
      results.append(
          _MergedIntrinsic(
              uri=uri,
              args=_merge_args(
                  abstract_parameter_type,
                  [call.argument for call in calls],
                  name_generator,
              ),
              return_type=return_type,
              unpack_to_locals=[name for (name, _) in locals_for_uri],
          )
      )
  return results


def _merge_args(
    abstract_parameter_type,
    args: list[building_blocks.ComputationBuildingBlock],
    name_generator,
) -> building_blocks.ComputationBuildingBlock:
  """Merges the arguments of multiple function invocations into one.

  Args:
    abstract_parameter_type: The abstract parameter type specification for the
      function being invoked. This is used to determine whether any functional
      parameters accept multiple arguments.
    args: A list where each element contains the arguments to a single call.
    name_generator: A generator used to create unique names.

  Returns:
    A building block to use as the new (merged) argument.
  """
  if abstract_parameter_type.is_federated():
    zip_args = building_block_factory.create_federated_zip(
        building_blocks.Struct(args)
    )
    # `create_federated_zip` introduces repeated names.
    zip_args, _ = tree_transformations.uniquify_reference_names(
        zip_args, name_generator
    )
    return zip_args
  if (
      abstract_parameter_type.is_tensor()
      or abstract_parameter_type.is_abstract()
  ):
    return building_blocks.Struct([(None, arg) for arg in args])
  if abstract_parameter_type.is_function():
    # For functions, we must compose them differently depending on whether the
    # abstract function (from the intrinsic definition) takes more than one
    # parameter.
    #
    # If it does not, such as in the `fn` argument to `federated_map`, we can
    # simply select out the argument and call the result:
    # `(fn0(arg[0]), fn1(arg[1]), ..., fnN(arg[n]))`
    #
    # If it takes multiple arguments such as the `accumulate` argument to
    # `federated_aggregate`, we have to select out the individual arguments to
    # pass to each function:
    #
    # `(
    #   fn0(arg[0][0], arg[1][0]),
    #   fn1(arg[0][1], arg[1][1]),
    #   ...
    #   fnN(arg[0][n], arg[1][n]),
    # )`
    param_name = next(name_generator)
    if abstract_parameter_type.parameter.is_struct():
      num_args = len(abstract_parameter_type.parameter)
      parameter_types = [[] for i in range(num_args)]
      for arg in args:
        for i in range(num_args):
          parameter_types[i].append(arg.type_signature.parameter[i])
      param_type = computation_types.StructType(parameter_types)
      param_ref = building_blocks.Reference(param_name, param_type)
      calls = []
      for n, fn in enumerate(args):
        args_to_fn = []
        for i in range(num_args):
          args_to_fn.append(
              building_blocks.Selection(
                  building_blocks.Selection(param_ref, index=i), index=n
              )
          )
        calls.append(
            building_blocks.Call(
                fn, building_blocks.Struct([(None, arg) for arg in args_to_fn])
            )
        )
    else:
      param_type = computation_types.StructType(
          [arg.type_signature.parameter for arg in args]
      )
      param_ref = building_blocks.Reference(param_name, param_type)
      calls = [
          building_blocks.Call(
              fn, building_blocks.Selection(param_ref, index=n)
          )
          for (n, fn) in enumerate(args)
      ]
    return building_blocks.Lambda(
        parameter_name=param_name,
        parameter_type=param_type,
        result=building_blocks.Struct([(None, call) for call in calls]),
    )
  if abstract_parameter_type.is_struct():
    # Bind each argument to a name so that we can reference them multiple times.
    arg_locals = []
    arg_refs = []
    for arg in args:
      arg_name = next(name_generator)
      arg_locals.append((arg_name, arg))
      arg_refs.append(building_blocks.Reference(arg_name, arg.type_signature))
    merged_args = []
    for i in range(len(abstract_parameter_type)):
      ith_args = [building_blocks.Selection(ref, index=i) for ref in arg_refs]
      merged_args.append(
          _merge_args(abstract_parameter_type[i], ith_args, name_generator)
      )
    return building_blocks.Block(
        arg_locals, building_blocks.Struct([(None, arg) for arg in merged_args])
    )
  raise TypeError(f'Cannot merge args of type: {abstract_parameter_type}')


# TODO(b/266565233): Remove during MapReduceForm and BroadcastForm cleanup.
def force_align_and_split_by_intrinsics(
    comp: building_blocks.Lambda,
    intrinsic_defaults: list[building_blocks.Call],
) -> tuple[building_blocks.Lambda, building_blocks.Lambda]:
  """Divides `comp` into before-and-after of calls to one or more intrinsics.

  The input computation `comp` must have the following properties:

  1. The computation `comp` is completely self-contained, i.e., there are no
     references to arguments introduced in a scope external to `comp`.

  2. `comp`'s return value must not contain uncalled lambdas.

  3. None of the calls to intrinsics in `intrinsic_defaults` may be
     within a lambda passed to another external function (intrinsic or
     compiled computation).

  4. No argument passed to an intrinsic in `intrinsic_defaults` may be
     dependent on the result of a call to an intrinsic in
     `intrinsic_uris_and_defaults`.

  5. All intrinsics in `intrinsic_defaults` must have "merge-able" arguments.
     Structs will be merged element-wise, federated values will be zipped, and
     functions will be composed:
       `f = lambda f1_arg, f2_arg: (f1(f1_arg), f2(f2_arg))`

  6. All intrinsics in `intrinsic_defaults` must return a single federated value
     whose member is the merged result of any merged calls, i.e.:
       `f(merged_arg).member = (f1(f1_arg).member, f2(f2_arg).member)`

  Under these conditions, (and assuming `comp` is a computation with non-`None`
  argument), this function will return two `building_blocks.Lambda`s `before`
  and `after` such that `comp` is semantically equivalent to the following
  expression*:

  ```
  (arg -> (let
    x=before(arg),
    y=intrinsic1(x[0]),
    z=intrinsic2(x[1]),
    ...
   in after(<arg, <y,z,...>>)))
  ```

  If `comp` is a no-arg computation, the returned computations will be
  equivalent (in the same sense as above) to:
  ```
  ( -> (let
    x=before(),
    y=intrinsic1(x[0]),
    z=intrinsic2(x[1]),
    ...
   in after(<y,z,...>)))
  ```

  *Note that these expressions may not be entirely equivalent under
  nondeterminism since there is no way in this case to handle computations in
  which `before` creates a random variable that is then used in `after`, since
  the only way for state to pass from `before` to `after` is for it to travel
  through one of the intrinsics.

  In this expression, there is only a single call to `intrinsic` that results
  from consolidating all occurrences of this intrinsic in the original `comp`.
  All logic in `comp` that produced inputs to any these intrinsic calls is now
  consolidated and jointly encapsulated in `before`, which produces a combined
  argument to all the original calls. All the remaining logic in `comp`,
  including that which consumed the outputs of the intrinsic calls, must have
  been encapsulated into `after`.

  If the original computation `comp` had type `(T -> U)`, then `before` and
  `after` would be `(T -> X)` and `(<T,Y> -> U)`, respectively, where `X` is
  the type of the argument to the single combined intrinsic call above. Note
  that `after` takes the output of the call to the intrinsic as well as the
  original argument to `comp`, as it may be dependent on both.

  Args:
    comp: The instance of `building_blocks.Lambda` that serves as the input to
      this transformation, as described above.
    intrinsic_defaults: A list of intrinsics with which to split the
      computation, provided as a list of `Call`s to insert if no intrinsic with
      a matching URI is found. Intrinsics in this list will be merged, and
      `comp` will be split across them.

  Returns:
    A pair of the form `(before, after)`, where each of `before` and `after`
    is a `building_blocks.ComputationBuildingBlock` instance that represents a
    part of the result as specified above.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(intrinsic_defaults, list)
  comp_repr = comp.compact_representation()

  # Flatten `comp` to call-dominant form so that we're working with just a
  # linear list of intrinsic calls with no indirection via tupling, selection,
  # blocks, called lambdas, or references.
  comp = to_call_dominant(comp)

  # CDF can potentially return blocks if there are variables not dependent on
  # the top-level parameter. We normalize these away.
  if not comp.is_lambda():
    comp.check_block()
    comp.result.check_lambda()
    if comp.result.result.is_block():
      additional_locals = comp.result.result.locals
      result = comp.result.result.result
    else:
      additional_locals = []
      result = comp.result.result
    # Note: without uniqueness, a local in `comp.locals` could potentially
    # shadow `comp.result.parameter_name`. However, `to_call_dominant`
    # above ensure that names are unique, as it ends in a call to
    # `uniquify_reference_names`.
    comp = building_blocks.Lambda(
        comp.result.parameter_name,
        comp.result.parameter_type,
        building_blocks.Block(comp.locals + additional_locals, result),
    )
  comp.check_lambda()

  # Simple computations with no intrinsic calls won't have a block.
  # Normalize these as well.
  if not comp.result.is_block():
    comp = building_blocks.Lambda(
        comp.parameter_name,
        comp.parameter_type,
        building_blocks.Block([], comp.result),
    )
  comp.result.check_block()

  name_generator = building_block_factory.unique_name_generator(comp)

  intrinsic_uris = set(call.function.uri for call in intrinsic_defaults)
  deps = _compute_intrinsic_dependencies(
      intrinsic_uris, comp.parameter_name, comp.result.locals, comp_repr
  )
  merged_intrinsics = _compute_merged_intrinsics(
      intrinsic_defaults, deps.uri_to_locals, name_generator
  )

  # Note: the outputs are labeled as `{uri}_param for convenience, e.g.
  # `federated_secure_sum_param: ...`.
  before = building_blocks.Lambda(
      comp.parameter_name,
      comp.parameter_type,
      building_blocks.Block(
          deps.locals_not_dependent_on_intrinsics,
          building_blocks.Struct(
              [
                  (f'{merged.uri}_param', merged.args)
                  for merged in merged_intrinsics
              ]
          ),
      ),
  )

  after_param_name = next(name_generator)
  if comp.parameter_type is not None:
    # TODO(b/147499373): If None-arguments were uniformly represented as empty
    # tuples, we would be able to avoid this (and related) ugly casing.
    after_param_type = computation_types.StructType([
        ('original_arg', comp.parameter_type),
        (
            'intrinsic_results',
            computation_types.StructType(
                [
                    (f'{merged.uri}_result', merged.return_type)
                    for merged in merged_intrinsics
                ]
            ),
        ),
    ])
  else:
    after_param_type = computation_types.StructType(
        [
            (
                'intrinsic_results',
                computation_types.StructType(
                    [
                        (f'{merged.uri}_result', merged.return_type)
                        for merged in merged_intrinsics
                    ]
                ),
            ),
        ]
    )
  after_param_ref = building_blocks.Reference(
      after_param_name, after_param_type
  )
  if comp.parameter_type is not None:
    original_arg_bindings = [(
        comp.parameter_name,
        building_blocks.Selection(after_param_ref, name='original_arg'),
    )]
  else:
    original_arg_bindings = []

  unzip_bindings = []
  for merged in merged_intrinsics:
    if merged.unpack_to_locals:
      intrinsic_result = building_blocks.Selection(
          building_blocks.Selection(after_param_ref, name='intrinsic_results'),
          name=f'{merged.uri}_result',
      )
      select_param_type = intrinsic_result.type_signature.member
      for i, binding_name in enumerate(merged.unpack_to_locals):
        select_param_name = next(name_generator)
        select_param_ref = building_blocks.Reference(
            select_param_name, select_param_type
        )
        selected = building_block_factory.create_federated_map_or_apply(
            building_blocks.Lambda(
                select_param_name,
                select_param_type,
                building_blocks.Selection(select_param_ref, index=i),
            ),
            intrinsic_result,
        )
        unzip_bindings.append((binding_name, selected))
  after = building_blocks.Lambda(
      after_param_name,
      after_param_type,
      building_blocks.Block(
          original_arg_bindings +
          # Note that we must duplicate `locals_not_dependent_on_intrinsics`
          # across both the `before` and `after` computations since both can
          # rely on them, and there's no way to plumb results from `before`
          # through to `after` except via one of the intrinsics being split
          # upon. In MapReduceForm, this limitation is caused by the fact that
          # `prepare` has no output which serves as an input to `report`.
          deps.locals_not_dependent_on_intrinsics
          + unzip_bindings
          + deps.locals_dependent_on_intrinsics,
          comp.result.result,
      ),
  )
  try:
    tree_analysis.check_has_unique_names(before)
    tree_analysis.check_has_unique_names(after)
  except tree_analysis.NonuniqueNameError as e:
    raise ValueError(f'nonunique names in result of splitting\n{comp}') from e
  return before, after


def _augment_lambda_with_parameter_for_unbound_references(
    comp: building_blocks.Lambda, lambda_parameter_extension_name: str
) -> tuple[
    building_blocks.Lambda, list[building_blocks.ComputationBuildingBlock]
]:
  """Resolves unbound references in `comp` by extending the input parameter.

  This is a private helper method intended to be used only in constructing the
  DistributeAggregateForm for a computation.

  We attempt to re-write `comp` such that a minimal amount of information gets
  passed through the extended input parameter. In other words, if we encounter
  a selection such as unbound_arg[0] in `comp`, the extended input parameter
  will contain a new value that can be used in place of unbound_arg[0] rather
  than in place of unbound_arg. As another example, if we encounter a struct
  such as [unbound_arg, bound_arg], the extended input parameter will contain a
  new value that can be used in place of unbound_arg rather than in place of the
  entire struct. We do not go so far as replacing entire calls, however, so
  for something like federated_sum(unbound_arg[0]) we would just add a new value
  to the extended input parameter that can be used in place of the argument to
  the call.

  Args:
    comp: The lambda computation potentially containing unbound references. The
      lambda must be in CDF, the input parameter must be a struct, and the input
      parameter must always be used via a selection and never used directly.
    lambda_parameter_extension_name: The name of the new element in the input
      parameter struct.

  Returns:
    A tuple containing
      - The revised lambda computation, which is guaranteed to have 1 more
        element in the input parameter struct and no unbound references.
      - A list of the `ComputationBuildingBlock`s that were replaced by
        selections and whose values are now expected to be supplied to the new
        input.

  Raises:
    TypeError: If types do not match.
    ValueError: If the input parameter is used within the computation in ways
      that are unsupported.
  """

  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(
      comp.type_signature.parameter, computation_types.StructType
  )

  comp_parameter_name = comp.parameter_name

  # Helper function to check that the input parameter is always accessed via
  # a selection. Note that this function will be used while traversing the
  # computation in preorder fashion.
  def _check_input_parameter_used_via_selection(inner_comp):
    # Mark subtrees involving a selection of the input parameter as
    # "transformed" so that we skip traversal of the inner reference subtree
    # below.
    if (
        inner_comp.is_selection()
        and inner_comp.source.is_reference()
        and inner_comp.source.name == comp_parameter_name
    ):
      return inner_comp, True

    # If we encounter a reference to the input parameter, it must be accessed
    # without an immediately surrounding selection. This is not supported
    # because we will not be able to satisfy type signature requirements in a
    # later step when we attempt to replace the input parameter with an
    # augmented one.
    if inner_comp.is_reference() and inner_comp.name == comp_parameter_name:
      raise ValueError(
          'Computation accesses input parameter without an immediate selection.'
      )

    return inner_comp, False

  # Trace the computation to ensure that the input parameter is always used via
  # a selection and never used directly.
  transformation_utils.transform_preorder(
      comp, _check_input_parameter_used_via_selection
  )

  unbound_refs = transformation_utils.get_map_of_unbound_references(comp)
  top_level_unbound_refs = unbound_refs[comp]

  # Maintain a map where the keys are the computations that should be passed to
  # the extended input parameter and the values correspond to the index at which
  # the computation will be found in the extended input parameter.
  new_input_comps = collections.OrderedDict()

  def _is_replacement_candidate(inner_comp):
    # A replacement is needed if the subtree represents a reference to an top-
    # level unbound ref.
    if inner_comp.is_reference() and unbound_refs[inner_comp].issubset(
        top_level_unbound_refs
    ):
      return True

    # A replacement is also needed if the subtree represents a selection into
    # a top-level unbound ref. We trigger the replacement on selections at this
    # level so that we can pass the minimal amount of information possible
    # through the extended input parameter.
    if inner_comp.is_selection():
      return _is_replacement_candidate(inner_comp.source)

    return False

  # Helper function to determine the list of new input comps. Note that this
  # function will be used while traversing the computation in preorder fashion.
  def _compute_new_parameter_elements(inner_comp):
    # Specific unbound references indicate that a value must be passed in via
    # an input extension. Only add the computation to the list if there is no
    # equivalent computation already present.
    if _is_replacement_candidate(inner_comp):
      if inner_comp not in new_input_comps:
        new_input_comps[inner_comp] = len(new_input_comps)
      return inner_comp, True

    return inner_comp, False

  # Trace the computation to find the unbound references and construct the
  # list of new input comps. Use a preorder transformation since it is
  # important to replace larger subtrees when possible (e.g. replacing an entire
  # selection subtree vs just replacing the selection source subtree).
  transformation_utils.transform_preorder(comp, _compute_new_parameter_elements)

  # Update the comp parameter type to include the new extension.
  new_parameter_type = computation_types.StructType(
      structure.to_elements(comp.type_signature.parameter)
      + [(
          lambda_parameter_extension_name,
          [e.type_signature for e in new_input_comps.keys()],
      )]
  )

  # Helper function to update a computation to use the extended input parameter.
  def _rebind_unbound_references_to_new_parameter(inner_comp):
    # Replace a computation involving specific unbound references with a
    # selection into the list of new input comps.
    if _is_replacement_candidate(inner_comp):
      assert inner_comp in new_input_comps
      new_comp = building_blocks.Selection(
          building_blocks.Selection(
              building_blocks.Reference(
                  comp_parameter_name, new_parameter_type
              ),
              # The input param extension will be added at the end.
              index=len(new_parameter_type) - 1,
          ),
          index=new_input_comps[inner_comp],
      )
      return new_comp, True

    # Replace selections into the original input parameter with selections into
    # the extended input parameter to maintain type signature correctness.
    if (
        inner_comp.is_selection()
        and inner_comp.source.is_reference()
        and inner_comp.source.name == comp_parameter_name
    ):
      return (
          building_blocks.Selection(
              building_blocks.Reference(
                  comp_parameter_name, new_parameter_type
              ),
              # Use the same index as before.
              index=inner_comp.as_index(),
          ),
          True,
      )

    return inner_comp, False

  # Replace the unbound references with selections into the list of new input
  # comps and also update existing selections into the original input parameter.
  # Use a preorder transformation again to ensure that the new input comps are
  # used in the correct order.
  comp = building_blocks.Lambda(
      comp.parameter_name,
      new_parameter_type,
      transformation_utils.transform_preorder(
          comp.result, _rebind_unbound_references_to_new_parameter
      )[0],
  )

  return comp, list(new_input_comps.keys())


class UnavailableRequiredInputsError(ValueError):
  pass


# Helper function to replace references with a given name with a different
# computation.
def _replace_references(
    comp: building_blocks.ComputationBuildingBlock,
    ref_name: str,
    replacement: building_blocks.ComputationBuildingBlock,
) -> building_blocks.ComputationBuildingBlock:
  def _replace(comp):
    if comp.is_reference() and comp.name == ref_name:
      return replacement, True
    return comp, False

  return transformation_utils.transform_postorder(comp, _replace)[0]


def divisive_force_align_and_split_by_intrinsics(
    comp: building_blocks.Lambda,
    intrinsic_defs_to_split: Collection[intrinsic_defs.IntrinsicDef],
    before_comp_allowed_original_arg_subparameters: Sequence[Sequence[int]],
    intrinsic_comp_allowed_original_arg_subparameters: Sequence[Sequence[int]],
    after_comp_allowed_original_arg_subparameters: Sequence[Sequence[int]],
) -> tuple[
    building_blocks.Lambda, building_blocks.Lambda, building_blocks.Lambda
]:
  """Divides `comp` into three components (before, intrinsic, after).

  The input computation `comp` must have the following properties:

  1. It has a non-None input parameter. (This requirement allows the
    implementation to be slightly simpler, but it can be removed in the future
    if necessary.)

  2. The computation `comp` is completely self-contained, i.e., there are no
    unbound references in `comp`.

  3. None of the calls to intrinsics in `intrinsic_defs_to_split` may be within
    a lambda passed to another external function (intrinsic or compiled
    computation).

  4. No argument passed to an intrinsic in `intrinsic_defs_to_split` may be
    dependent on the result of another call to an intrinsic in
    `intrinsic_defs_to_split`.

  Under these conditions, this function will return three
  `building_blocks.Lambda`s `before`, `intrinsic`, and `after` such that
  `comp` is semantically equivalent to the following expression:
  ```
  (arg -> (let
    <"intrinsic_args_from_before_comp": x, "intermediate_state": temp> =
          before(<before_subparam_0(arg), before_subparam_1(arg), ...>)
    y = intrinsic(<intrinsic_subparam_0(arg), intrinsic_subparam_1(arg), ...,
          "intrinsic_args_from_before_comp": x>)
   in after(<after_subparam_0(arg), after_subparam_1(arg), ...,
          "intrinsic_results": y, "intermediate_state": temp>)
  ))
  ```

  The `*_allowed_original_arg_subparameters` arguments are used to specify which
  parts of the original computation input each of the split computatations is
  allowed to depend on. Each of these arguments takes a list of paths of integer
  indices that describe selections of the original computation input. For
  example, [(3,1,), (2,)] would mean that usage of arg[3][1] and arg[2] is
  permitted in the applicable output comp. An empty path list means that no
  part of the original computation input may be used, whereas [()] means that
  all parts of the original computation input may be used.

  The `before` computation will only use parts of `comp`'s input that are
  permitted by `before_comp_allowed_original_arg_subparameters`. It returns a
  tuple containing one of the inputs to the `intrinsic` computation and the
  minimal temporary state that is needed in the `after` computation.

  The `intrinsic` computation consists solely of calls to intrinsics in
  `intrinsic_defs_to_split`. Each intrinsic call will only use parts of `comp`'s
  input permitted by `intrinsic_comp_allowed_original_arg_subparameters` and/or
  the first part of the `before` result. The i-th entry of the output struct
  will be the result of the i-th intrinsic call. If the original computation
  contains calls to intrinsics that are not in `intrinsic_defs_to_split`, these
  calls will become part of either the `before` or `after` computations.

  The `after` computation takes as input the parts of the original `comp`
  argument allowed by `after_comp_allowed_original_arg_subparameters`, the
  result of `intrinsic`, and the temporary state produced by `before`. It
  returns the same output as the original computation.

  The input computation will be partitioned without repetition across the three
  output computations, meaning that no top-level call from the original input
  computation's CDF will appear in more than of the output computations. This
  is important since some calls may be non-deterministic and repeating them
  across the output computations may cause unexpected results to be produced
  when the output computations are executed. The link between the `before` and
  `after` comps via `intermediate_state` exists to ensure that this non-
  repetition guarantee can be met. The `intermediate_state` that is produced
  is not guaranteed to minimal, however.

  If it is not possible to split an input computation according to the allowed
  subparameters and the available channels between the three output
  computations, an error will be raised. If a valid split does exist, this
  function is guaranteed to find it.

  Args:
    comp: The instance of `building_blocks.Lambda` that serves as the input to
      this transformation, as described above.
    intrinsic_defs_to_split: A list of intrinsics with which to split the
      computation.
    before_comp_allowed_original_arg_subparameters: A list of paths describing
      the selections into the original comp input that can be used within the
      `before` comp.
    intrinsic_comp_allowed_original_arg_subparameters: A list of paths
      describing the selections into the original comp input that can be used
      within the `intrinsic` comp.
    after_comp_allowed_original_arg_subparameters: A list of paths describing
      the selections into the original comp input that can be used within the
      `after` comp.

  Returns:
    A tuple of the form `(before, intrinsic, after)`, where each of `before`,
    `intrinsic`, and `after` is a building_blocks.Lambda` instance with a
    `building_blocks.Block` result.

    Details about the inputs and outputs of the three computations as well as
    the contents of the `intrinsic` comp are specified above.

  Raises:
    TypeError: If arguments are not the documented types.
    NonAlignableAlongIntrinsicError: If the intrinsics to split on are not
      independent of each other.
    UnavailableRequiredInputsError: If the computation cannot be split due to
      inputs that are required but not available based on the provided
      subparameters.
  """

  # The following code attempts to construct a split using these steps:
  # 1. Check that the inputs are valid and categorize the locals in the original
  #    comp's result block.
  # 2. Produce a preliminary intrinsic comp that returns the outputs of the
  #    intrinsic calls and depends only on the allowed original input
  #    subparameters.
  # 3. Extend the input for the intrinsic comp to make it valid (i.e. resolve
  #    any unbound references).
  # 4. Produce a preliminary after comp that returns the original output and
  #    depends only on the allowed original input subparameters and the output
  #    from the intrinsic comp.
  # 5. Extend the input for the after comp to make it valid (i.e. resolve any
  #    unbound references).
  # 6. Attempt to produce an input comp that returns the two extensions required
  #    by steps 3 and 5 while only depending on the allowed original input
  #    subparameters. If this fails (i.e. unbound references exist), a split is
  #    not possible and an error is raised.
  # 7. Remove any repetition across the before and after comps.
  # 8. Normalize the output comps and check that their structure meets the
  #    promised guarantees.

  ############################### Step 1 ######################################
  if not comp.is_lambda():
    raise TypeError('Expected input computation to be a lambda computation.')

  if not comp.parameter_name or not comp.parameter_type:
    raise TypeError('Expected input lambda with a non-None input parameter.')

  py_typecheck.check_type(intrinsic_defs_to_split, list)

  # Normalize the input computation so that we are guaranteed to have a lambda
  # computation with a result block before attempting to split.
  comp = get_normalized_call_dominant_lambda(comp)

  # Identify which locals in the result block represent intrinsic calls, which
  # locals depend on intrinsic calls, and which locals are independent of
  # intrinsic calls.
  intrinsic_uris = set(
      intrinsic_def.uri for intrinsic_def in intrinsic_defs_to_split
  )
  deps = _compute_intrinsic_dependencies(
      intrinsic_uris,
      comp.parameter_name,
      comp.result.locals,
      comp.compact_representation(),
  )

  ############################### Step 2 ######################################
  # Generate a preliminary intrinsic comp.
  intrinsic_locals: list[tuple[str, building_blocks.Call]] = []
  for intrinsic_locals_for_uri in deps.uri_to_locals.values():
    intrinsic_locals.extend(intrinsic_locals_for_uri)
  intrinsic_results = [
      building_blocks.Reference(local_name, local_value.type_signature)
      for local_name, local_value in intrinsic_locals
  ]
  preliminary_intrinsic_comp = building_blocks.Lambda(
      comp.parameter_name,
      comp.parameter_type,
      building_blocks.Block(
          intrinsic_locals, building_blocks.Struct(intrinsic_results)
      ),
  )

  # Attempt to rewrite the preliminary intrinsic comp as a function of the
  # allowed subparameters.
  preliminary_intrinsic_comp = (
      tree_transformations.as_function_of_some_subparameters(
          preliminary_intrinsic_comp,
          intrinsic_comp_allowed_original_arg_subparameters,
      )
  )

  ############################### Step 3 ######################################
  # Determine the `intrinsic_args_from_before_comp` input extension that is
  # needed to make the intrinsic comp valid. The additional input values will
  # ultimately be returned by the before comp.
  intrinsic_comp, intrinsic_args_from_before_comp_values = (
      _augment_lambda_with_parameter_for_unbound_references(
          preliminary_intrinsic_comp,
          lambda_parameter_extension_name='intrinsic_args_from_before_comp',
      )
  )

  # There should be an additional element in the input of the resulting
  # intrinsic comp, which should also have no unbound references.
  assert (
      len(intrinsic_comp.parameter_type)
      == len(preliminary_intrinsic_comp.parameter_type) + 1
  )
  tree_analysis.check_contains_no_unbound_references(intrinsic_comp)

  ############################### Step 4 ######################################
  # Generate a preliminary after comp.
  name_generator = building_block_factory.unique_name_generator(comp)
  after_param_name = next(name_generator)
  after_param_type = [
      ('original_arg', comp.parameter_type),
      (
          'intrinsic_results',
          [local_value.type_signature for _, local_value in intrinsic_locals],
      ),
  ]
  original_arg_index = 0
  intrinsic_results_index = 1
  intrinsic_result_bindings: list[tuple[str, building_blocks.Selection]] = []
  for i, (local_name, _) in enumerate(intrinsic_locals):
    intrinsic_result_bindings.append((
        local_name,
        building_blocks.Selection(
            building_blocks.Selection(
                building_blocks.Reference(after_param_name, after_param_type),
                index=intrinsic_results_index,
            ),
            index=i,
        ),
    ))
  preliminary_after_comp = building_blocks.Lambda(
      after_param_name,
      after_param_type,
      _replace_references(
          building_blocks.Block(
              intrinsic_result_bindings
              + deps.locals_not_dependent_on_intrinsics
              + deps.locals_dependent_on_intrinsics,
              comp.result.result,
          ),
          comp.parameter_name,
          building_blocks.Selection(
              building_blocks.Reference(after_param_name, after_param_type),
              index=original_arg_index,
          ),
      ),
  )

  # Modify the allowed subparameters to accomodate the extra level of structure
  # for the original arg and to also allow access to the intrinsic results.
  original_arg_index = 0
  intrinsic_results_index = 1
  after_comp_allowed_subparameters = [
      (original_arg_index, *x)
      for x in after_comp_allowed_original_arg_subparameters
  ] + [(intrinsic_results_index,)]

  # Attempt to rewrite the preliminary after comp as a function of the allowed
  # subparameters. Any unneeded locals will be pruned but local names will
  # otherwise not be modified.
  preliminary_after_comp = (
      tree_transformations.as_function_of_some_subparameters(
          preliminary_after_comp, after_comp_allowed_subparameters
      )
  )

  ############################### Step 5 ######################################
  # Determine the `intermediate_state` input extension that is needed to make
  # the after comp valid. The additional input values will ultimately be
  # returned by the before comp. Before we do this, reverse the extra level of
  # indirection that was added to the original arg in the previous step in case
  # there are portions of the original arg that are now unbound and need to be
  # resolved via the new extension.
  preliminary_after_comp = tree_transformations.replace_selections(
      preliminary_after_comp,
      after_param_name,
      {
          (0,): building_blocks.Reference(
              comp.parameter_name, comp.parameter_type
          )
      },
  )

  preliminary_after_comp, intermediate_state = (
      _augment_lambda_with_parameter_for_unbound_references(
          preliminary_after_comp,
          lambda_parameter_extension_name='intermediate_state',
      )
  )

  # This next version of the after comp should have no unbound references.
  tree_analysis.check_contains_no_unbound_references(preliminary_after_comp)

  ############################### Step 6 ######################################
  # Generate a preliminary before comp that produces the values required by the
  # `intrinsic_args_from_before_comp` and `intermediate_state` extensions in
  # steps 3 and 5 above.
  before_result = [
      (
          'intrinsic_args_from_before_comp',
          building_blocks.Struct(intrinsic_args_from_before_comp_values),
      ),
      ('intermediate_state', building_blocks.Struct(intermediate_state)),
  ]
  preliminary_before_comp = building_blocks.Lambda(
      comp.parameter_name,
      comp.parameter_type,
      building_blocks.Block(
          deps.locals_not_dependent_on_intrinsics,
          building_blocks.Struct(before_result),
      ),
  )

  # Attempt to rewrite the preliminary before comp as a function of the allowed
  # subparameters. Any unneeded locals will be pruned but local names will
  # otherwise not be modified.
  preliminary_before_comp = (
      tree_transformations.as_function_of_some_subparameters(
          preliminary_before_comp,
          before_comp_allowed_original_arg_subparameters,
      )
  )

  # If the resulting comp is not valid (i.e. it contains no unbound
  # references), then a split is not possible and we throw an error.
  if not tree_analysis.contains_no_unbound_references(preliminary_before_comp):
    raise UnavailableRequiredInputsError(
        'The computation is not splittable given the allowed subparameters.'
    )

  ############################### Step 7 ######################################
  # Remove any duplication that exists between the locals in the result blocks
  # of the before comp and after comp by extending the `intermediate_state` link
  # between the before and after comps that was established in step 5.
  # Duplication may exist since locals_not_dependent_on_intrinsics was used in
  # constructing both the preliminary before and after comps. Note that all
  # unnecessary locals have already been pruned during the subparameter
  # transformations. For this step to correctly detect duplication, the
  # transformations applied in the previous steps must not have modified the
  # original local names.
  after_local_names = [
      local_name for local_name, _ in preliminary_after_comp.result.locals
  ]
  duplicated_locals = [
      (local_name, local_value)
      for local_name, local_value in preliminary_before_comp.result.locals
      if local_name in set(after_local_names)
  ]

  # Update the before comp result to produce the extended intermediate state.
  before_result_elements = structure.to_elements(
      preliminary_before_comp.result.result
  )
  intermediate_state_index_in_before_result = 1
  intermediate_state_name, intermediate_state_vals = before_result_elements[
      intermediate_state_index_in_before_result
  ]
  duplicate_intermediate_state_vals = [
      (None, building_blocks.Reference(local_name, local_value.type_signature))
      for local_name, local_value in duplicated_locals
  ]
  extended_intermediate_state_vals = building_blocks.Struct(
      structure.to_elements(intermediate_state_vals)
      + duplicate_intermediate_state_vals
  )
  before_result_elements[intermediate_state_index_in_before_result] = (
      intermediate_state_name,
      extended_intermediate_state_vals,
  )

  # Update the before comp to produce the result with the extended intermediate
  # state. If we have reached this stage, this latest before comp should have
  # no unbound references.
  before_comp = building_blocks.Lambda(
      preliminary_before_comp.parameter_name,
      preliminary_before_comp.parameter_type,
      building_blocks.Block(
          preliminary_before_comp.result.locals,
          building_blocks.Struct(before_result_elements),
      ),
  )
  tree_analysis.check_contains_no_unbound_references(before_comp)

  # Update the after comp parameter to represent the extended intermediate
  # state. Also restore the name associated with the intrinsic results portion
  # of the after comp parameter (it would have been lost by the earlier
  # subparams transformation).
  after_param_type_signature = structure.to_elements(
      preliminary_after_comp.parameter_type
  )
  intrinsic_results_index_in_after_comp_param = -2
  after_param_type_signature[intrinsic_results_index_in_after_comp_param] = (
      'intrinsic_results',
      after_param_type_signature[intrinsic_results_index_in_after_comp_param][
          1
      ],
  )
  intermediate_state_index_in_after_comp_param = -1
  after_param_type_signature[intermediate_state_index_in_after_comp_param] = (
      'intermediate_state',
      [e.type_signature for e in extended_intermediate_state_vals],
  )
  preliminary_after_comp = building_blocks.Lambda(
      preliminary_after_comp.parameter_name,
      after_param_type_signature,
      _replace_references(
          preliminary_after_comp.result,
          preliminary_after_comp.parameter_name,
          building_blocks.Reference(
              preliminary_after_comp.parameter_name, after_param_type_signature
          ),
      ),
  )

  # Replace the duplicated locals in the after comp with references to the
  # extended intermediate state. Note when indexing into the extended
  # intermediate state that it consists of the portion contructed in step 5
  # followed by the duplicated locals portion.
  block_locals: list[tuple[str, building_blocks.ComputationBuildingBlock]] = []
  duplicated_local_names = [local_name for local_name, _ in duplicated_locals]
  intermediate_state_index = (
      len(preliminary_after_comp.type_signature.parameter) - 1
  )
  duplicated_local_index = len(intermediate_state)
  for local_name, local_value in preliminary_after_comp.result.locals:
    if local_name in duplicated_local_names:
      block_locals.append((
          local_name,
          building_blocks.Selection(
              building_blocks.Selection(
                  building_blocks.Reference(
                      preliminary_after_comp.parameter_name,
                      preliminary_after_comp.parameter_type,
                  ),
                  index=intermediate_state_index,
              ),
              index=duplicated_local_index,
          ),
      ))
      duplicated_local_index += 1
    else:
      block_locals.append((local_name, local_value))

  # Update the after comp to use the de-duplicated block locals.
  after_comp = building_blocks.Lambda(
      preliminary_after_comp.parameter_name,
      preliminary_after_comp.parameter_type,
      building_blocks.Block(block_locals, preliminary_after_comp.result.result),
  )
  tree_analysis.check_contains_no_unbound_references(after_comp)

  ############################### Step 8 ######################################
  # Normalize all of the output computations.
  before_comp = get_normalized_call_dominant_lambda(before_comp)
  intrinsic_comp = get_normalized_call_dominant_lambda(intrinsic_comp)
  after_comp = get_normalized_call_dominant_lambda(after_comp)

  # Check that the intrinsic comp consists of a block containing locals that are
  # exclusively calls for the allowed intrinsics and that the results are
  # returned in the same order they are computed.
  expected_intrinsic_comp_result_names: list[str] = []
  for intrinsic_local, intrinsic_call in intrinsic_comp.result.locals:
    assert intrinsic_call.is_call()
    assert intrinsic_call.function.is_intrinsic()
    assert intrinsic_call.function.uri in intrinsic_uris
    expected_intrinsic_comp_result_names.append(intrinsic_local)
  actual_intrinsic_comp_result_names = [
      ref.name for ref in intrinsic_comp.result.result
  ]
  assert (
      expected_intrinsic_comp_result_names == actual_intrinsic_comp_result_names
  )

  return (before_comp, intrinsic_comp, after_comp)
