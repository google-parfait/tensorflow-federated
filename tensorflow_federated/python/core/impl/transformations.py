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
from six.moves import range

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import transformation_utils


def extract_intrinsics(comp):
  r"""Extracts intrinsics to the scope which binds any variable it depends on.

  This transform traverses `comp` postorder, matches the following pattern, and
  replaces the following computation containing a called intrinsic:

        ...
           \
            Call
           /    \
  Intrinsic      ...

  with the following computation containing a block with the extracted called
  intrinsic:

                  Block
                 /     \
         [x=Call]       ...
           /    \          \
  Intrinsic      ...        Ref(x)

  The called intrinsics are extracted to the scope which binds any variable the
  called intrinsic depends. If the called intrinsic is not bound by any
  computation in `comp` it will be extracted to the root. Both the
  `parameter_name` of a `computation_building_blocks.Lambda` and the name of any
  variable defined by a `computation_building_blocks.Block` can affect the scope
  in which a reference in called intrinsic is bound.

  NOTE: This function will also extract blocks to the scope in which they are
  bound because block variables can restrict the scope in which intrinsics are
  bound.

  Args:
    comp: The computation building block in which to perform the extraction. The
      names of lambda parameters and locals in blocks in `comp` must be unique.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
    ValueError: If `comp` contains a reference named `name`.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  _check_has_unique_names(comp)
  name_generator = computation_constructing_utils.unique_name_generator(comp)
  unbound_references = _get_unbound_references(comp)

  def _contains_unbound_reference(comp, names):
    """Returns `True` if `comp` contains unbound references to `names`.

    This function will update the non-local `unbound_references` captured from
    the parent context if `comp` is not contained in that collection. This can
    happen when new computations are created and added to the AST.

    Args:
      comp: The computation building block to test.
      names: A Python string or a list, tuple, or set of Python strings.
    """
    if isinstance(names, six.string_types):
      names = (names,)
    if comp not in unbound_references:
      references = _get_unbound_references(comp)
      unbound_references.update(references)
    return any(n in unbound_references[comp] for n in names)

  def _is_called_intrinsic_or_block(comp):
    """Returns `True` if `comp` is a called intrinsic or a block."""
    return (_is_called_intrinsic(comp) or
            isinstance(comp, computation_building_blocks.Block))

  def _should_transform(comp):
    """Returns `True` if `comp` should be transformed.

    The following `_extract_intrinsic_*` methods all depend on being invoked
    after `_should_transform` evaluates to `True` for a given `comp`. Because of
    this certain assumptions are made:

    * transformation functions will transform a given `comp`
    * block variables are guaranteed to not be empty

    Args:
      comp: The computation building block in which to test.
    """
    if isinstance(comp, computation_building_blocks.Block):
      return (_is_called_intrinsic_or_block(comp.result) or any(
          isinstance(e, computation_building_blocks.Block)
          for _, e in comp.locals))
    elif isinstance(comp, computation_building_blocks.Call):
      return _is_called_intrinsic_or_block(comp.argument)
    elif isinstance(comp, computation_building_blocks.Lambda):
      if _is_called_intrinsic(comp.result):
        return True
      if isinstance(comp.result, computation_building_blocks.Block):
        for index, (_, variable) in enumerate(comp.result.locals):
          names = [n for n, _ in comp.result.locals[:index]]
          if (not _contains_unbound_reference(variable, comp.parameter_name) and
              not _contains_unbound_reference(variable, names)):
            return True
    elif isinstance(comp, computation_building_blocks.Selection):
      return _is_called_intrinsic_or_block(comp.source)
    elif isinstance(comp, computation_building_blocks.Tuple):
      return any(_is_called_intrinsic_or_block(e) for e in comp)
    return False

  def _extract_from_block(comp):
    """Returns a new computation with all intrinsics extracted."""
    if _is_called_intrinsic(comp.result):
      called_intrinsic = comp.result
      name = six.next(name_generator)
      variables = comp.locals
      variables.append((name, called_intrinsic))
      result = computation_building_blocks.Reference(
          name, called_intrinsic.type_signature)
      return computation_building_blocks.Block(variables, result)
    elif isinstance(comp.result, computation_building_blocks.Block):
      return computation_building_blocks.Block(comp.locals + comp.result.locals,
                                               comp.result.result)
    else:
      variables = []
      for name, variable in comp.locals:
        if isinstance(variable, computation_building_blocks.Block):
          variables.extend(variable.locals)
          variables.append((name, variable.result))
        else:
          variables.append((name, variable))
      return computation_building_blocks.Block(variables, comp.result)

  def _extract_from_call(comp):
    """Returns a new computation with all intrinsics extracted."""
    if _is_called_intrinsic(comp.argument):
      called_intrinsic = comp.argument
      name = six.next(name_generator)
      variables = ((name, called_intrinsic),)
      result = computation_building_blocks.Reference(
          name, called_intrinsic.type_signature)
    else:
      block = comp.argument
      variables = block.locals
      result = block.result
    call = computation_building_blocks.Call(comp.function, result)
    block = computation_building_blocks.Block(variables, call)
    return _extract_from_block(block)

  def _extract_from_lambda(comp):
    """Returns a new computation with all intrinsics extracted."""
    if _is_called_intrinsic(comp.result):
      called_intrinsic = comp.result
      name = six.next(name_generator)
      variables = ((name, called_intrinsic),)
      ref = computation_building_blocks.Reference(
          name, called_intrinsic.type_signature)
      if not _contains_unbound_reference(comp.result, comp.parameter_name):
        fn = computation_building_blocks.Lambda(comp.parameter_name,
                                                comp.parameter_type, ref)
        return computation_building_blocks.Block(variables, fn)
      else:
        block = computation_building_blocks.Block(variables, ref)
        return computation_building_blocks.Lambda(comp.parameter_name,
                                                  comp.parameter_type, block)
    else:
      block = comp.result
      extracted_variables = []
      retained_variables = []
      for name, variable in block.locals:
        names = [n for n, _ in retained_variables]
        if (not _contains_unbound_reference(variable, comp.parameter_name) and
            not _contains_unbound_reference(variable, names)):
          extracted_variables.append((name, variable))
        else:
          retained_variables.append((name, variable))
      if retained_variables:
        result = computation_building_blocks.Block(retained_variables,
                                                   block.result)
      else:
        result = block.result
      fn = computation_building_blocks.Lambda(comp.parameter_name,
                                              comp.parameter_type, result)
      block = computation_building_blocks.Block(extracted_variables, fn)
      return _extract_from_block(block)

  def _extract_from_selection(comp):
    """Returns a new computation with all intrinsics extracted."""
    if _is_called_intrinsic(comp.source):
      called_intrinsic = comp.source
      name = six.next(name_generator)
      variables = ((name, called_intrinsic),)
      result = computation_building_blocks.Reference(
          name, called_intrinsic.type_signature)
    else:
      block = comp.source
      variables = block.locals
      result = block.result
    selection = computation_building_blocks.Selection(
        result, name=comp.name, index=comp.index)
    block = computation_building_blocks.Block(variables, selection)
    return _extract_from_block(block)

  def _extract_from_tuple(comp):
    """Returns a new computation with all intrinsics extracted."""
    variables = []
    elements = []
    for name, element in anonymous_tuple.to_elements(comp):
      if _is_called_intrinsic_or_block(element):
        variable_name = six.next(name_generator)
        variables.append((variable_name, element))
        ref = computation_building_blocks.Reference(variable_name,
                                                    element.type_signature)
        elements.append((name, ref))
      else:
        elements.append((name, element))
    tup = computation_building_blocks.Tuple(elements)
    block = computation_building_blocks.Block(variables, tup)
    return _extract_from_block(block)

  def _transform(comp):
    """Returns a new transformed computation or `comp`."""
    if not _should_transform(comp):
      return comp, False
    if isinstance(comp, computation_building_blocks.Block):
      comp = _extract_from_block(comp)
    elif isinstance(comp, computation_building_blocks.Call):
      comp = _extract_from_call(comp)
    elif isinstance(comp, computation_building_blocks.Lambda):
      comp = _extract_from_lambda(comp)
    elif isinstance(comp, computation_building_blocks.Selection):
      comp = _extract_from_selection(comp)
    elif isinstance(comp, computation_building_blocks.Tuple):
      comp = _extract_from_tuple(comp)
    return comp, True

  return transformation_utils.transform_postorder(comp, _transform)


def inline_block_locals(comp):
  """Inlines all block local variables.

  Since this transform is not necessarily safe, it should only be calles if
  all references under `comp` have unique names.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      whose blocks we wish to inline.

  Returns:
    A possibly different `computation_building_blocks.ComputationBuildingBlock`
    containing the same logic as `comp`, but with all blocks inlined.

  Raises:
    ValueError: If `comp` has variables with non-unique names.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  _check_has_unique_names(comp)

  def _transform(comp, symbol_tree):
    """Inline transform function."""
    if isinstance(comp, computation_building_blocks.Reference):
      value = symbol_tree.get_payload_with_name(comp.name).value
      # This identifies a variable bound by a Block as opposed to a Lambda.
      if value is not None:
        return value, True
      else:
        return comp, False
    elif isinstance(comp, computation_building_blocks.Block):
      return comp.result, True
    return comp, False

  symbol_tree = transformation_utils.SymbolTree(
      transformation_utils.ReferenceCounter)
  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, _transform, symbol_tree)


def merge_chained_blocks(comp):
  r"""Merges all the chained blocks in `comp` into one block.

  Looks for occurrences of the following pattern:

        Block
       /     \
  [...]       Block
             /     \
        [...]       Comp(x)

  And merges them to

        Block
       /     \
  [...]       Comp(x)

  Preserving the relative ordering of any locals declarations in a postorder
  walk, which therefore preserves scoping rules.

  Notice that because TFF Block constructs bind their variables in sequence, it
  is completely safe to add the locals lists together in this implementation,

  Args:
    comp: The computation building block in which to perform the merges.

  Returns:
    Transformed version of `comp` with its neighboring blocks merged.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    return (isinstance(comp, computation_building_blocks.Block) and
            isinstance(comp.result, computation_building_blocks.Block))

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    transformed_comp = computation_building_blocks.Block(
        comp.locals + comp.result.locals, comp.result.result)
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


def merge_chained_federated_maps_or_applys(comp):
  r"""Merges all the chained federated maps or federated apply in `comp`.

  This transform traverses `comp` postorder, matches the following pattern, and
  replaces the following computation containing two federated map intrinsics:

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Comp(x), Call]
                          /    \
                 Intrinsic      Tuple
                                |
                                [Comp(y), Comp(z)]

  intrinsic(<x, intrinsic(<y, z>)>)

  with the following computation containing one federated map or apply
  intrinsic:


            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Block, Comp(z)]
                 /     \
       [fn=Tuple]       Lambda(arg)
           |                       \
   [Comp(y), Comp(x)]               Call
                                   /    \
                             Sel(1)      Call
                            /           /    \
                     Ref(fn)      Sel(0)      Ref(arg)
                                 /
                          Ref(fn)

  intrinsic(<(let fn=<y, x> in (arg -> fn[1](fn[0](arg)))), z>)

  The functional computations `x` and `y`, and the argument `z` are retained;
  the other computations are replaced.

  Args:
    comp: The computation building block in which to perform the merges.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    """Returns `True` if `comp` is a chained federated map."""
    if _is_called_intrinsic(comp, (
        intrinsic_defs.FEDERATED_APPLY.uri,
        intrinsic_defs.FEDERATED_MAP.uri,
    )):
      outer_arg = comp.argument[1]
      if _is_called_intrinsic(outer_arg, comp.function.uri):
        return True
    return False

  def _transform(comp):
    """Returns a new transformed computation or `comp`."""
    if not _should_transform(comp):
      return comp, False

    def _create_block_to_chained_calls(comps):
      r"""Constructs a transformed block computation from `comps`.

                     Block
                    /     \
          [fn=Tuple]       Lambda(arg)
              |                       \
      [Comp(y), Comp(x)]               Call
                                      /    \
                                Sel(1)      Call
                               /           /    \
                        Ref(fn)      Sel(0)      Ref(arg)
                                    /
                             Ref(fn)

      (let fn=<y, x> in (arg -> fn[1](fn[0](arg)))

      Args:
        comps: A Python list of computations.

      Returns:
        A `computation_building_blocks.Block`.
      """
      functions = computation_building_blocks.Tuple(comps)
      fn_ref = computation_building_blocks.Reference('fn',
                                                     functions.type_signature)
      arg_type = comps[0].type_signature.parameter
      arg_ref = computation_building_blocks.Reference('arg', arg_type)
      arg = arg_ref
      for index, _ in enumerate(comps):
        fn_sel = computation_building_blocks.Selection(fn_ref, index=index)
        call = computation_building_blocks.Call(fn_sel, arg)
        arg = call
      lam = computation_building_blocks.Lambda(arg_ref.name,
                                               arg_ref.type_signature, call)
      return computation_building_blocks.Block([('fn', functions)], lam)

    block = _create_block_to_chained_calls((
        comp.argument[1].argument[0],
        comp.argument[0],
    ))
    arg = computation_building_blocks.Tuple([
        block,
        comp.argument[1].argument[1],
    ])
    intrinsic_type = computation_types.FunctionType(
        arg.type_signature, comp.function.type_signature.result)
    intrinsic = computation_building_blocks.Intrinsic(comp.function.uri,
                                                      intrinsic_type)
    transformed_comp = computation_building_blocks.Call(intrinsic, arg)
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


def merge_tuple_intrinsics(comp):
  r"""Merges all the tuples of intrinsics in `comp` into one intrinsic.

  This transform traverses `comp` postorder, matches the following pattern, and
  replaces the following computation containing a tuple of called intrinsics all
  represeting the same operation:

           Tuple
           |
           [Call,                        Call, ...]
           /    \                       /    \
  Intrinsic      Tuple         Intrinsic      Tuple
                 |                            |
        [Comp(f1), Comp(v1), ...]    [Comp(f2), Comp(v2), ...]

  <Intrinsic(<f1, v1>), Intrinsic(<f2, v2>)>

  with the following computation containing one called intrinsic:

  federated_unzip(Call)
                 /    \
        Intrinsic      Tuple
                       |
                       [Block,    federated_zip(Tuple), ...]
                       /     \                  |
               fn=Tuple       Lambda(arg)       [Comp(v1), Comp(v2), ...]
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
    comp: The computation building block in which to perform the merges.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    uri = (
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
        intrinsic_defs.FEDERATED_MAP.uri,
    )
    return (isinstance(comp, computation_building_blocks.Tuple) and
            _is_called_intrinsic(comp[0], uri) and all(
                _is_called_intrinsic(element, comp[0].function.uri)
                for element in comp))

  def _get_comps(comp):
    """Constructs a 2 dimentional Python list of computations.

    Args:
      comp: A `computation_building_blocks.Tuple` containing `n` called
        intrinsics with `m` arguments.

    Returns:
      A 2 dimentional Python list of computations.
    """
    first_call = comp[0]
    comps = [[] for _ in range(len(first_call.argument))]
    for _, call in anonymous_tuple.to_elements(comp):
      for index, arg in enumerate(call.argument):
        comps[index].append(arg)
    return comps

  def _transform_functional_args(comps):
    r"""Transforms the functional computations `comps`.

    Given a computation containing `n` called intrinsics with `m` arguments,
    this function constructs the following computation from the functional
    arguments to the called intrinsic:

                    Block
                   /     \
         [fn=Tuple]       Lambda(arg)
             |                       \
    [Comp(f1), Comp(f2), ...]         Tuple
                                      |
                                 [Call,                  Call, ...]
                                 /    \                 /    \
                           Sel(0)      Sel(0)     Sel(1)      Sel(1)
                          /           /          /           /
                   Ref(fn)    Ref(arg)    Ref(fn)    Ref(arg)

    with one `computation_building_blocks.Call` for each `n`. This computation
    represents one of `m` arguments that should be passed to the call of the
    transformed computation.

    Args:
      comps: a Python list of computations.

    Returns:
      A `computation_building_blocks.Block`.
    """
    functions = computation_building_blocks.Tuple(comps)
    fn = computation_building_blocks.Reference('fn', functions.type_signature)
    arg_type = [element.type_signature.parameter for element in comps]
    arg = computation_building_blocks.Reference('arg', arg_type)
    elements = []
    for index in range(len(comps)):
      sel_fn = computation_building_blocks.Selection(fn, index=index)
      sel_arg = computation_building_blocks.Selection(arg, index=index)
      call = computation_building_blocks.Call(sel_fn, sel_arg)
      elements.append(call)
    calls = computation_building_blocks.Tuple(elements)
    lam = computation_building_blocks.Lambda(arg.name, arg.type_signature,
                                             calls)
    return computation_building_blocks.Block([('fn', functions)], lam)

  def _transform_non_functional_args(comps):
    r"""Transforms the non-functional computations `comps`.

    Given a computation containing `n` called intrinsics with `m` arguments,
    this function constructs the following computation from the non-functional
    arguments to the called intrinsic:

    federated_zip(Tuple)
                  |
                  [Comp, Comp, ...]

    with one `computation_building_blocks.ComputationBuildignBlock` for each
    `n`. This computation represents one of `m` arguments that should be passed
    to the call of the transformed computation.

    Args:
      comps: A Python list of computations.

    Returns:
      A `computation_building_blocks.Block`.
    """
    values = computation_building_blocks.Tuple(comps)
    first_comp = comps[0]
    if isinstance(first_comp.type_signature, computation_types.FederatedType):
      return computation_constructing_utils.create_federated_zip(values)
    else:
      return values

  def _transform_comps(elements):
    """Constructs a Python list of transformed computations.

    Given a computation containing `n` called intrinsics with `m` arguments,
    this function constructs the following Python list of computations:

    [Block, federated_zip(Tuple), ...]

    with one `computation_building_blocks.Block` for each functional computation
    in `m` and one called federated zip for each non-functional computation in
    `m`. This list of computations represent the `m` arguments that should be
    passed
    to the call of the transformed computation.

    Args:
      elements: A 2 dimentional Python list of computations.

    Returns:
      A Python list of computations.
    """
    args = []
    for comps in elements:
      if isinstance(comps[0].type_signature, computation_types.FunctionType):
        arg = _transform_functional_args(comps)
      else:
        arg = _transform_non_functional_args(comps)
      args.append(arg)
    return args

  def _transform(comp):
    """Returns a new transformed computation or `comp`."""
    if not _should_transform(comp):
      return comp, False
    named_comps = anonymous_tuple.to_elements(comp)
    elements = _get_comps(comp)
    comps = _transform_comps(elements)
    arg = computation_building_blocks.Tuple(comps)
    first_comp = comp[0]
    parameter_type = computation_types.to_type(arg.type_signature)
    type_signature = [call.type_signature.member for _, call in named_comps]
    result_type = computation_types.FederatedType(
        type_signature, first_comp.type_signature.placement)
    intrinsic_type = computation_types.FunctionType(parameter_type, result_type)
    intrinsic = computation_building_blocks.Intrinsic(first_comp.function.uri,
                                                      intrinsic_type)
    call = computation_building_blocks.Call(intrinsic, arg)
    tup = computation_constructing_utils.create_federated_unzip(call)
    names = [name for name, _ in named_comps]
    transformed_comp = computation_constructing_utils.create_named_tuple(
        tup, names)
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


def remove_mapped_or_applied_identity(comp):
  r"""Removes all the mapped or applied identity functions in `comp`.

  This transform traverses `comp` postorder, matches the following pattern, and
  removes all the mapped or applied identity fucntions by replacing the
  following computation:

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Lambda(x), Comp(y)]
                           \
                            Ref(x)

  Intrinsic(<(x -> x), y>)

  with its argument:

  Comp(y)

  y

  Args:
    comp: The computation building block in which to perform the removals.

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
      return comp, False
    transformed_comp = comp.argument[1]
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


def replace_called_lambda_with_block(comp):
  r"""Replaces all the called lambdas in `comp` with a block.

  This transform traverses `comp` postorder, matches the following pattern, and
  replaces the following computation containing a called lambda:

            Call
           /    \
  Lambda(x)      Comp(y)
           \
            Comp(z)

  (x -> z)(y)

  with the following computation containing a block:

             Block
            /     \
  [x=Comp(y)]       Comp(z)

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
      return comp, False
    transformed_comp = computation_building_blocks.Block(
        [(comp.function.parameter_name, comp.argument)], comp.function.result)
    return transformed_comp, True

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
      return comp, False
    # We need 'wrapped_body' to accept exactly one argument.
    wrapped_body = lambda x: body(x)  # pylint: disable=unnecessary-lambda
    transformed_comp = federated_computation_utils.zero_or_one_arg_fn_to_building_block(
        wrapped_body, 'arg', comp.type_signature.parameter, context_stack, uri)
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


def replace_selection_from_tuple_with_element(comp):
  r"""Replaces any selection from a tuple with the underlying tuple element.

  Replaces any occurences of:

  Selection
           \
            Tuple
            |
            [Comp, Comp, ...]

  with the appropriate Comp, as determined by the `index` or `name` of the
  `Selection`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock` to
      transform.

  Returns:
    A possibly modified version of comp, without any occurrences of selections
    from tuples.

  Raises:
    TypeError: If `comp` is not an instance of
      `computation_building_blocks.ComputationBuildingBlock`.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    return (isinstance(comp, computation_building_blocks.Selection) and
            isinstance(comp.source, computation_building_blocks.Tuple))

  def _get_index_from_name(selection_name, tuple_type_signature):
    named_type_signatures = anonymous_tuple.to_elements(tuple_type_signature)
    return [x[0] for x in named_type_signatures].index(selection_name)

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    if comp.name is not None:
      index = _get_index_from_name(comp.name, comp.source.type_signature)
    else:
      index = comp.index
    return comp.source[index], True

  return transformation_utils.transform_postorder(comp, _transform)


def uniquify_compiled_computation_names(comp):
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
  name_generator = computation_constructing_utils.unique_name_generator(
      None, prefix='')

  def _should_transform(comp):
    return isinstance(comp, computation_building_blocks.CompiledComputation)

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    transformed_comp = computation_building_blocks.CompiledComputation(
        comp.proto, six.next(name_generator))
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


def uniquify_reference_names(comp):
  """Replaces all the reference names in `comp` with unique names.

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    Returns a transformed version of comp inside of which all variable names
      are guaranteed to be unique.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  name_generator = computation_constructing_utils.unique_name_generator(None)

  class _RenameNode(transformation_utils.BoundVariableTracker):
    """transformation_utils.SymbolTree node for renaming References in ASTs."""

    def __init__(self, name, value):
      super(_RenameNode, self).__init__(name, value)
      py_typecheck.check_type(name, str)
      self.new_name = six.next(name_generator)

    def __str__(self):
      return 'Value: {}, name: {}, new_name: {}'.format(self.value, self.name,
                                                        self.new_name)

  def _transform(comp, context_tree):
    """Renames References in `comp` to unique names."""
    if isinstance(comp, computation_building_blocks.Reference):
      new_name = context_tree.get_payload_with_name(comp.name).new_name
      return computation_building_blocks.Reference(new_name,
                                                   comp.type_signature,
                                                   comp.context), True
    elif isinstance(comp, computation_building_blocks.Block):
      new_locals = []
      for name, val in comp.locals:
        context_tree.walk_down_one_variable_binding()
        new_name = context_tree.get_payload_with_name(name).new_name
        new_locals.append((new_name, val))
      return computation_building_blocks.Block(new_locals, comp.result), True
    elif isinstance(comp, computation_building_blocks.Lambda):
      context_tree.walk_down_one_variable_binding()
      new_name = context_tree.get_payload_with_name(
          comp.parameter_name).new_name
      return computation_building_blocks.Lambda(new_name, comp.parameter_type,
                                                comp.result), True
    return comp, False

  symbol_tree = transformation_utils.SymbolTree(_RenameNode)
  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, _transform, symbol_tree)


def _is_called_intrinsic(comp, uri=None):
  """Returns `True` if `comp` is a called intrinsic with the `uri` or `uri`s.

            Call
           /
  Intrinsic

  Args:
    comp: The computation building block to test.
    uri: A uri or a list, tuple, or set of uri.
  """
  if isinstance(uri, six.string_types):
    uri = (uri,)
  if uri is not None:
    py_typecheck.check_type(uri, (list, tuple, set))
  return (isinstance(comp, computation_building_blocks.Call) and
          isinstance(comp.function, computation_building_blocks.Intrinsic) and
          (uri is None or comp.function.uri in uri))


def _is_identity_function(comp):
  """Returns `True` if `comp` is an identity function."""
  return (isinstance(comp, computation_building_blocks.Lambda) and
          isinstance(comp.result, computation_building_blocks.Reference) and
          comp.parameter_name == comp.result.name)


def _check_has_unique_names(comp):
  if not transformation_utils.has_unique_names(comp):
    raise ValueError(
        'This transform should only be called after we have uniquified all '
        '`computation_building_blocks.Reference` names, since we may be moving '
        'computations with unbound references under constructs which bind '
        'those references.')


def _get_unbound_references(comp):
  """Gets a Python `dict` of the unbound references in `comp`.

  Compuations that are equal will have the same collections of unbounded
  references, so it is safe to use `comp` as the key for this `dict` even though
  a given compuation may appear in many positions in the AST.

  Args:
    comp: The computation building block to parse.

  Returns:
    A Python `dict` of elements where keys are the compuations in `comp` and
    values are a Python `set` of the names of the unbound references in the
    subtree of that compuation.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  references = {}

  def _update(comp):
    """Updates the Python dict of references."""
    if isinstance(comp, computation_building_blocks.Reference):
      references[comp] = set((comp.name,))
    elif isinstance(comp, computation_building_blocks.Block):
      references[comp] = set()
      names = []
      for name, variable in comp.locals:
        elements = references[variable]
        references[comp].update([e for e in elements if e not in names])
        names.append(name)
      elements = references[comp.result]
      references[comp].update([e for e in elements if e not in names])
    elif isinstance(comp, computation_building_blocks.Call):
      elements = references[comp.function]
      if comp.argument is not None:
        elements.update(references[comp.argument])
      references[comp] = elements
    elif isinstance(comp, computation_building_blocks.Lambda):
      elements = references[comp.result]
      references[comp] = set([e for e in elements if e != comp.parameter_name])
    elif isinstance(comp, computation_building_blocks.Selection):
      references[comp] = references[comp.source]
    elif isinstance(comp, computation_building_blocks.Tuple):
      elements = [references[e] for e in comp]
      references[comp] = set(itertools.chain.from_iterable(elements))
    else:
      references[comp] = set()
    return comp, False

  transformation_utils.transform_postorder(comp, _update)
  return references
