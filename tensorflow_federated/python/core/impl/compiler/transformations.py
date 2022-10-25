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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A library of composite transformations.

A composite transformation is one that applies multiple atomic transformation to
an AST either pointwise or serially.
"""

import collections

import attr

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
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

  class _Scope():
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
      for (name, value) in structure.iter_elements(comp):
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
            building_blocks.Reference(comp.parameter_name, comp.parameter_type))
      result = _build(comp.result, scope)
      block = scope.bindings_to_block_with_result(result)
      return building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                    block)
    elif comp.is_block():
      scope = scope.new_child()
      for (name, value) in comp.locals:
        scope.add_local(name, _build(value, scope))
      return _build(comp.result, scope)
    elif (comp.is_intrinsic() or comp.is_data() or
          comp.is_compiled_computation() or comp.is_placement()):
      return comp
    else:
      raise ValueError(
          f'Unrecognized computation kind\n{comp}\nin\n{global_comp}')

  scope = _Scope()
  result = _build(comp, scope)
  comp = scope.bindings_to_block_with_result(result)
  for transform in [
      tree_transformations.uniquify_reference_names,
      tree_transformations.remove_unused_block_locals,
  ]:
    comp, _ = transform(comp)
  return comp


_NamedBinding = tuple[str, building_blocks.ComputationBuildingBlock]


@attr.s
class _IntrinsicDependencies:
  uri_to_locals: dict[str, list[_NamedBinding]] = attr.ib(
      factory=lambda: collections.defaultdict(list))
  locals_dependent_on_intrinsics: list[_NamedBinding] = attr.ib(factory=list)
  locals_not_dependent_on_intrinsics: list[_NamedBinding] = attr.ib(
      factory=list)


class _NonAlignableAlongIntrinsicError(ValueError):
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
          raise ValueError(
              f'Can\'t resolve {subvalue.name} in {[(name, value.compact_representation()) for name, value in locals_list]}\n'
              f'Current map {intrinsic_dependencies_for_ref}\n'
              f'Original comp: {comp_repr}\n')
        intrinsic_dependencies.update(  # pylint: disable=cell-var-from-loop
            intrinsic_dependencies_for_ref[subvalue.name])
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
        local_value.is_call() and local_value.function.is_intrinsic() and
        local_value.function.uri in intrinsic_uris)
    if is_intrinsic_call:
      if intrinsic_dependencies:
        raise _NonAlignableAlongIntrinsicError(
            'Cannot force-align intrinsics:\n'
            f'Call to intrinsic `{local_value.function.uri}` depends '
            f'on calls to intrinsics:\n`{intrinsic_dependencies}`.')
      intrinsic_dependencies_for_ref[local_name] = set(
          [local_value.function.uri])
      result.uri_to_locals[local_value.function.uri].append(
          (local_name, local_value))
    else:
      intrinsic_dependencies_for_ref[local_name] = intrinsic_dependencies
      if intrinsic_dependencies:
        result.locals_dependent_on_intrinsics.append((local_name, local_value))
      else:
        result.locals_not_dependent_on_intrinsics.append(
            (local_name, local_value))
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
              unpack_to_locals=[]))
    else:
      calls = [local[1] for local in locals_for_uri]
      result_placement = calls[0].type_signature.placement
      result_all_equal = calls[0].type_signature.all_equal
      for call in calls:
        if call.type_signature.all_equal != result_all_equal:
          raise ValueError('Encountered intrinsics to be merged with '
                           f'mismatched all_equal bits. Intrinsic of URI {uri} '
                           f'first call had all_equal bit {result_all_equal}, '
                           'encountered call with all_equal value '
                           f'{call.type_signature.all_equal}')
      return_type = computation_types.FederatedType(
          computation_types.StructType([
              (None, call.type_signature.member) for call in calls
          ]),
          placement=result_placement,
          all_equal=result_all_equal)
      abstract_parameter_type = default_call.function.intrinsic_def(
      ).type_signature.parameter
      results.append(
          _MergedIntrinsic(
              uri=uri,
              args=_merge_args(abstract_parameter_type,
                               [call.argument for call in calls],
                               name_generator),
              return_type=return_type,
              unpack_to_locals=[name for (name, _) in locals_for_uri],
          ))
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
        building_blocks.Struct(args))
    # `create_federated_zip` introduces repeated names.
    zip_args, _ = tree_transformations.uniquify_reference_names(
        zip_args, name_generator)
    return zip_args
  if (abstract_parameter_type.is_tensor() or
      abstract_parameter_type.is_abstract()):
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
      for (n, fn) in enumerate(args):
        args_to_fn = []
        for i in range(num_args):
          args_to_fn.append(
              building_blocks.Selection(
                  building_blocks.Selection(param_ref, index=i), index=n))
        calls.append(
            building_blocks.Call(
                fn,
                building_blocks.Struct([(None, arg) for arg in args_to_fn])))
    else:
      param_type = computation_types.StructType(
          [arg.type_signature.parameter for arg in args])
      param_ref = building_blocks.Reference(param_name, param_type)
      calls = [
          building_blocks.Call(fn,
                               building_blocks.Selection(param_ref, index=n))
          for (n, fn) in enumerate(args)
      ]
    return building_blocks.Lambda(
        parameter_name=param_name,
        parameter_type=param_type,
        result=building_blocks.Struct([(None, call) for call in calls]))
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
          _merge_args(abstract_parameter_type[i], ith_args, name_generator))
    return building_blocks.Block(
        arg_locals,
        building_blocks.Struct([(None, arg) for arg in merged_args]))
  raise TypeError(f'Cannot merge args of type: {abstract_parameter_type}')


def force_align_and_split_by_intrinsics(
    comp: building_blocks.Lambda,
    intrinsic_defaults: list[building_blocks.Call],
) -> tuple[building_blocks.Lambda, building_blocks.Lambda]:
  """Divides `comp` into before-and-after of calls to one ore more intrinsics.

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
        comp.result.parameter_name, comp.result.parameter_type,
        building_blocks.Block(comp.locals + additional_locals, result))
  comp.check_lambda()

  # Simple computations with no intrinsic calls won't have a block.
  # Normalize these as well.
  if not comp.result.is_block():
    comp = building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                  building_blocks.Block([], comp.result))
  comp.result.check_block()

  name_generator = building_block_factory.unique_name_generator(comp)

  intrinsic_uris = set(call.function.uri for call in intrinsic_defaults)
  deps = _compute_intrinsic_dependencies(intrinsic_uris, comp.parameter_name,
                                         comp.result.locals, comp_repr)
  merged_intrinsics = _compute_merged_intrinsics(intrinsic_defaults,
                                                 deps.uri_to_locals,
                                                 name_generator)

  # Note: the outputs are labeled as `{uri}_param for convenience, e.g.
  # `federated_secure_sum_param: ...`.
  before = building_blocks.Lambda(
      comp.parameter_name, comp.parameter_type,
      building_blocks.Block(
          deps.locals_not_dependent_on_intrinsics,
          building_blocks.Struct([(f'{merged.uri}_param', merged.args)
                                  for merged in merged_intrinsics])))

  after_param_name = next(name_generator)
  if comp.parameter_type is not None:
    # TODO(b/147499373): If None-arguments were uniformly represented as empty
    # tuples, we would be able to avoid this (and related) ugly casing.
    after_param_type = computation_types.StructType([
        ('original_arg', comp.parameter_type),
        ('intrinsic_results',
         computation_types.StructType([(f'{merged.uri}_result',
                                        merged.return_type)
                                       for merged in merged_intrinsics])),
    ])
  else:
    after_param_type = computation_types.StructType([
        ('intrinsic_results',
         computation_types.StructType([(f'{merged.uri}_result',
                                        merged.return_type)
                                       for merged in merged_intrinsics])),
    ])
  after_param_ref = building_blocks.Reference(after_param_name,
                                              after_param_type)
  if comp.parameter_type is not None:
    original_arg_bindings = [
        (comp.parameter_name,
         building_blocks.Selection(after_param_ref, name='original_arg'))
    ]
  else:
    original_arg_bindings = []

  unzip_bindings = []
  for merged in merged_intrinsics:
    if merged.unpack_to_locals:
      intrinsic_result = building_blocks.Selection(
          building_blocks.Selection(after_param_ref, name='intrinsic_results'),
          name=f'{merged.uri}_result')
      select_param_type = intrinsic_result.type_signature.member
      for i, binding_name in enumerate(merged.unpack_to_locals):
        select_param_name = next(name_generator)
        select_param_ref = building_blocks.Reference(select_param_name,
                                                     select_param_type)
        selected = building_block_factory.create_federated_map_or_apply(
            building_blocks.Lambda(
                select_param_name, select_param_type,
                building_blocks.Selection(select_param_ref, index=i)),
            intrinsic_result)
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
          deps.locals_not_dependent_on_intrinsics + unzip_bindings +
          deps.locals_dependent_on_intrinsics,
          comp.result.result))
  try:
    tree_analysis.check_has_unique_names(before)
    tree_analysis.check_has_unique_names(after)
  except tree_analysis.NonuniqueNameError as e:
    raise ValueError(f'nonunique names in result of splitting\n{comp}') from e
  return before, after
