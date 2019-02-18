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
"""A library of transformations that can be applied to a computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import six

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import transformation_utils


def replace_compiled_computations_names_with_unique_names(comp):
  """Replaces all the compiled computations names in `comp` with unique names.

  This transform traverses `comp` postorder and replaces the name of all the
  comiled computations found in `comp` with a unique name.

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  name_generator = itertools.count(start=1)

  def _should_transform(comp):
    return isinstance(comp, computation_building_blocks.CompiledComputation)

  def _transform(comp):
    if not _should_transform(comp):
      return comp
    return computation_building_blocks.CompiledComputation(
        comp.proto, str(six.next(name_generator)))

  return transformation_utils.transform_postorder(comp, _transform)


def replace_intrinsic_with_callable(comp, uri, body, context_stack):
  """Replaces all the intrinsics with the given `uri` with a callable.

  This transform traverses `comp` postorder and replaces all the intrinsics with
  the given `uri` with a polymorphic callable that represents the body of the
  implementation of the intrinsic; i.e., one that given the parameter of the
  intrinsic constructs the intended result. This will typically be a Python
  function decorated with `@federated_computation` to make it into a polymorphic
  callable.

  Args:
    comp: The computation building block in which to perform the replacements.
    uri: The URI of the intrinsic to replace.
    body: A polymorphic callable.
    context_stack: The context stack to use.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(uri, six.string_types)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  if not callable(body):
    raise TypeError('The body of the intrinsic must be a callable.')

  def _should_transform(comp):
    return (isinstance(comp, computation_building_blocks.Intrinsic) and
            comp.uri == uri and
            isinstance(comp.type_signature, computation_types.FunctionType))

  def _transform(comp):
    if not _should_transform(comp):
      return comp
    # We need 'wrapped_body' to accept exactly one argument.
    wrapped_body = lambda x: body(x)  # pylint: disable=unnecessary-lambda
    return federated_computation_utils.zero_or_one_arg_fn_to_building_block(
        wrapped_body, 'arg', comp.type_signature.parameter, context_stack, uri)

  return transformation_utils.transform_postorder(comp, _transform)


def replace_called_lambda_with_block(comp):
  r"""Replaces all the called lambdas in `comp` with a block.

  This transform traverses `comp` postorder, matches the following pattern `*`,
  and replaces the following computation containing a called lambda:

            *Call
            /    \
  *Lambda(x)      Comp(y)
            \
             Comp(z)

  (x -> z)(y)

  with the following computation containing a block:

            Block
           /     \
  x=Comp(y)       Comp(z)

  let x=y in z

  The functional computation `b` and the argument `c` are retained; the other
  computations are replaced. This transformation is used to facilitate the
  merging of TFF orchestration logic, in particular to remove unnecessary lambda
  expressions and as a stepping stone for merging Blocks together.

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    return (isinstance(comp, computation_building_blocks.Call) and
            isinstance(comp.function, computation_building_blocks.Lambda))

  def _transform(comp):
    if not _should_transform(comp):
      return comp
    return computation_building_blocks.Block(
        [(comp.function.parameter_name, comp.argument)], comp.function.result)

  return transformation_utils.transform_postorder(comp, _transform)


def remove_mapped_or_applied_identity(comp):
  r"""Removes all the mapped or applied identity functions in `comp`.

  This transform traverses `comp` postorder, matches the following pattern `*`,
  and removes all the mapped or applied identity fucntions by replacing the
  following computation:

            *Call
            /    \
  *Intrinsic     *Tuple
                  |
                  [*Lambda(x), Comp(y)]
                             \
                             *Ref(x)

  Intrinsic(<(x -> x), y>)

  with its argument:

  Comp(y)

  y

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    """Returns `True` if `comp` is a mapped or applied identity function."""
    if (isinstance(comp, computation_building_blocks.Call) and
        isinstance(comp.function, computation_building_blocks.Intrinsic) and
        comp.function.uri in (
            intrinsic_defs.FEDERATED_MAP.uri,
            intrinsic_defs.FEDERATED_APPLY.uri,
            intrinsic_defs.SEQUENCE_MAP.uri,
        )):
      called_function = comp.argument[0]
      if _is_identity_function(called_function):
        return True
    return False

  def _transform(comp):
    if not _should_transform(comp):
      return comp
    called_arg = comp.argument[1]
    return called_arg

  return transformation_utils.transform_postorder(comp, _transform)


def replace_chained_federated_maps_with_federated_map(comp):
  r"""Replaces all the chained federated maps in `comp` with one federated map.

  This transform traverses `comp` postorder, matches the following pattern `*`,
  and replaces the following computation containing two federated map
  intrinsics:

            *Call
            /    \
  *Intrinsic     *Tuple
                  |
                  [Comp(x), *Call]
                            /    \
                  *Intrinsic     *Tuple
                                  |
                                  [Comp(y), Comp(z)]

  federated_map(<x, federated_map(<y, z>)>)

  with the following computation containing one federated map intrinsic:

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Lambda(arg), Comp(z)]
                             \
                              Call
                             /    \
                      Comp(x)      Call
                                  /    \
                           Comp(y)      Ref(arg)

  federated_map(<(arg -> x(y(arg))), z>)

  The functional computations `x` and `y`, and the argument `z` are retained;
  the other computations are replaced.

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    """Returns `True` if `comp` is a chained federated map computation."""
    uri = intrinsic_defs.FEDERATED_MAP.uri
    if _is_called_intrinsic(comp, uri):
      outer_arg = comp.argument[1]
      if _is_called_intrinsic(outer_arg, uri):
        return True
    return False

  def _transform(comp):
    """Returns a new transformed computation or `comp`."""
    if not _should_transform(comp):
      return comp
    map_arg = comp.argument[1].argument[1]
    inner_arg = computation_building_blocks.Reference(
        'arg', map_arg.type_signature.member)
    inner_fn = comp.argument[1].argument[0]
    inner_call = computation_building_blocks.Call(inner_fn, inner_arg)
    outer_fn = comp.argument[0]
    outer_call = computation_building_blocks.Call(outer_fn, inner_call)
    map_lambda = computation_building_blocks.Lambda(inner_arg.name,
                                                    inner_arg.type_signature,
                                                    outer_call)
    map_tuple = computation_building_blocks.Tuple([map_lambda, map_arg])
    map_intrinsic_type = computation_types.FunctionType(
        map_tuple.type_signature, comp.function.type_signature.result)
    map_intrinsic = computation_building_blocks.Intrinsic(
        comp.function.uri, map_intrinsic_type)
    return computation_building_blocks.Call(map_intrinsic, map_tuple)

  return transformation_utils.transform_postorder(comp, _transform)


def replace_tuple_intrinsics_with_intrinsic(comp):
  r"""Replaces all the tuples of intrinsics in `comp` with one intrinsic.

  This transform traverses `comp` postorder, matches the following pattern `*`,
  and replaces the following computation containing a tuple of called intrinsics
  all represeting the same operation:

           *Tuple
            |
           *[Call,                        Call, ...]
            /    \                       /    \
  *Intrinsic      Tuple        *Intrinsic      Tuple
                  |                            |
         [Comp(f1), Comp(v1), ...]    [Comp(f2), Comp(v2), ...]

  <Intrinsic(<f1, v1>), Intrinsic(<f2, v2>)>

  with the following computation containing one called intrinsic:

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Block,               Tuple, ...]
                 /     \               |
         fn=Tuple       Lambda(arg)    [Comp(y), Comp(y), ...]
            |                      \
   [Comp(f1), Comp(f2), ...]        Tuple
                                    |
                               [Call,                  Call, ...]
                               /    \                 /    \
                         Sel(0)      Sel(0)     Sel(1)      Sel(1)
                        /           /          /           /
                 Ref(fn)    Ref(arg)    Ref(fn)    Ref(arg)

  Intrinsic(<
    (let fn=<f1, f2> in (arg -> <fn[0](arg[0]), fn[1](arg[1])>)),
    <v1, v2>,
  >)

  The functional computations `f1`, `f2`, etc..., and the computations `v1`,
  `v2`, etc... are retained; the other computations are replaced.

  NOTE: This is just an example of what this transformation would look like when
  applied to a tuple of federated maps. The components `f1`, `f2`, `v1`, and
  `v2` and the number of those components are not important.

  NOTE: This transformation is implemented to match the following intrinsics:

  * intrinsic_defs.FEDERATED_MAP.uri
  * intrinsic_defs.FEDERATED_AGGREGATE.uri

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    uri = (
        intrinsic_defs.FEDERATED_MAP.uri,
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
    )
    return (isinstance(comp, computation_building_blocks.Tuple) and
            _is_called_intrinsic(comp[0], uri) and all(
                _is_called_intrinsic(element, comp[0].function.uri)
                for element in comp))

  def _transform(comp):
    """Returns a new transformed computation or `comp`."""
    if not _should_transform(comp):
      return comp

    def _get_comp_elements(comp):
      """Constructs a 2 dimentional Python list of (name, computation) pairs.

      Args:
        comp: A `computation_building_blocks.Tuple` containing `n` called
          intrinsics with `m` arguments.

      Returns:
        A 2 dimentional Python list of (name, computation) pairs.
      """
      first_call = comp[0]
      comps = [[] for _ in range(len(first_call.argument))]
      for name, call in anonymous_tuple.to_elements(comp):
        for index, arg in enumerate(call.argument):
          comps[index].append((name, arg))
      return comps

    def _create_transformed_arg_from_functional_comps(comps):
      r"""Constructs a transformed computation from `comps`.

      Given the "original" computation containing `n` called intrinsics
      with `m` arguments, this function constructs the following computation:

                     Block
                    /     \
            fn=Tuple       Lambda(arg)
               |                      \
      [Comp(f1), Comp(f2), ...]        Tuple
                                       |
                                  [Call,                  Call, ...]
                                  /    \                 /    \
                            Sel(0)      Sel(0)     Sel(1)      Sel(1)
                           /           /          /           /
                    Ref(fn)    Ref(arg)    Ref(fn)    Ref(arg)

      with one `computation_building_blocks.Call` for each `n`. This computation
      represents one of `m` arguments that should be passed to the call of the
      "transformed" computation.

      Args:
        comps: A Python list of functional computations.

      Returns:
        A `computation_building_blocks.Block` from the Python list of functional
        computations `comps`.
      """
      elements = []
      arg_types = []
      for name, fn in comps:
        elements.append((name, fn))
        arg_types.append(fn.type_signature.parameter)
      functions = computation_building_blocks.Tuple(elements)
      fn = computation_building_blocks.Reference('fn', functions.type_signature)
      arg = computation_building_blocks.Reference('arg', arg_types)
      elements = []
      for index, (name, _) in enumerate(comps):
        sel_fn = computation_building_blocks.Selection(fn, index=index)
        sel_arg = computation_building_blocks.Selection(arg, index=index)
        call = computation_building_blocks.Call(sel_fn, sel_arg)
        elements.append((name, call))
      calls = computation_building_blocks.Tuple(elements)
      lam = computation_building_blocks.Lambda(arg.name, arg.type_signature,
                                               calls)
      return computation_building_blocks.Block([('fn', functions)], lam)

    def _create_transformed_args(elements):
      """Constructs a Python list of transformed computations.

      Given the "original" computation containing `n` called intrinsics
      with `m` arguments, this function constructs the following Python list
      of computations:

      [Block, Tuple, ...]

      with one `computation_building_blocks.Block` for each functional
      computation in `m` and one `computation_building_blocks.Tuple` for each
      non-functional computation in `m`. This list of computations represent the
      arguments that should be passed to the `computation_building_blocks.Call`
      of the "transformed" computation.

      Args:
        elements: A 2 dimentional Python list of (name, computation) pairs.

      Returns:
        A Python list of computations.
      """
      args = []
      for comps in elements:
        _, first_comp = comps[0]
        if isinstance(first_comp.type_signature,
                      computation_types.FunctionType):
          arg = _create_transformed_arg_from_functional_comps(comps)
        else:
          arg = computation_building_blocks.Tuple(comps)
        args.append(arg)
      return args

    elements = _get_comp_elements(comp)
    args = _create_transformed_args(elements)
    arg = computation_building_blocks.Tuple(args)
    parameter_type = computation_types.to_type(arg.type_signature)
    elements = anonymous_tuple.to_elements(comp)
    result_type = [(n, c.function.type_signature.result) for n, c in elements]
    intrinsic_type = computation_types.FunctionType(parameter_type, result_type)
    intrinsic = computation_building_blocks.Intrinsic(comp[0].function.uri,
                                                      intrinsic_type)
    return computation_building_blocks.Call(intrinsic, arg)

  return transformation_utils.transform_postorder(comp, _transform)


def inline_blocks_with_n_referenced_locals(comp, inlining_threshold=1):
  """Replaces locals referenced few times in `comp` with bound values.

  Args:
    comp: The computation building block in which to inline the locals which
      occur only `inlining_threshold` times in the result computation.
    inlining_threshold: The threshhold below which to inline computations. E.g.
      if `inlining_threshold` is 1, locals which are referenced exactly once
      will be inlined, but locals which are referenced twice or more will not.

  Returns:
    A modified version of `comp` for which all occurrences of
    `computation_building_blocks.Block`s with locals which
      are referenced `inlining_threshold` or fewer times inlined with the value
      of the local.
  """

  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  count_snapshot = transformation_utils.scope_count_snapshot(comp)
  op = transformation_utils.InlineReferences(inlining_threshold, count_snapshot,
                                             comp)
  return transformation_utils.transform_postorder(comp, op)


def _is_called_intrinsic(comp, uri):
  """Returns `True` if `comp` is a called intrinsic with the `uri` or `uri`s.

            Call
           /
  Intrinsic

  Args:
    comp: The computation building block in which to test.
    uri: A uri or a list, tuple, or set of uri.
  """
  if isinstance(uri, six.string_types):
    uri = (uri,)
  py_typecheck.check_type(uri, (list, tuple, set))
  return (isinstance(comp, computation_building_blocks.Call) and
          isinstance(comp.function, computation_building_blocks.Intrinsic) and
          comp.function.uri in uri)


def _is_identity_function(comp):
  """Returns `True` if `comp` is an identity function."""
  return (isinstance(comp, computation_building_blocks.Lambda) and
          isinstance(comp.result, computation_building_blocks.Reference) and
          comp.parameter_name == comp.result.name)


def inline_locals_referenced_once(comp):
  """Inlines any local variable in a `Block` which is referenced only once.

  Removes any unreferenced locals in `Block` definitions.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      representing the root of the AST whose block locals we want to check for
      inlining.

  Returns:
    Returns a possibly modified version of `comp`.
  """

  def transform(comp, symbol_tree):
    """Inlines references and updates block locals declarations.

    Replaces instances of `computation_building_blocks.Reference` with their
    bound value if referenced once.

    Args:
      comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
        whose block locals we wish to inline.
      symbol_tree: Instance of `transformation_utils.SymbolTree` with nodes of
        class `transformation_utils.ReferenceCounter`, providing counts of all
        bound variables in the scope available to `comp`.

    Returns:
      A possibly modified version of `comp`.
    """
    if isinstance(comp, computation_building_blocks.Reference):
      payload = symbol_tree.get_payload_with_name(comp.name)
      if payload.count <= 1 and payload.value is not None:
        return payload.value
    return comp

  def locals_filter(name, symbol_tree):
    """Discards locals referenced less than twice in the block."""
    payload = symbol_tree.get_payload_with_name(name)
    if payload.count <= 1:
      return False
    return True

  count_of_references = transformation_utils.get_count_of_references_to_variables(
      comp)
  inlined = transformation_utils.transform_postorder_with_symbol_bindings(
      comp, transform, count_of_references, locals_filter)

  return inlined
