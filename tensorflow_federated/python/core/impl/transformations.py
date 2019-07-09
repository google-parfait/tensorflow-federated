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
from six.moves import zip

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import compiled_computation_transforms
from tensorflow_federated.python.core.impl import computation_building_block_utils
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl import intrinsic_bodies
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import transformation_utils
from tensorflow_federated.python.core.impl import type_utils


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
    comp: The computation building block in which to perform the extractions.
      The names of lambda parameters and block variables in `comp` must be
      unique.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
    ValueError: If `comp` contains variables with non-unique names.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  check_has_unique_names(comp)
  name_generator = computation_constructing_utils.unique_name_generator(comp)
  unbound_references = get_map_of_unbound_references(comp)

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
      references = get_map_of_unbound_references(comp)
      unbound_references.update(references)
    return any(n in unbound_references[comp] for n in names)

  def _is_called_intrinsic_or_block(comp):
    """Returns `True` if `comp` is a called intrinsic or a block."""
    return (computation_building_block_utils.is_called_intrinsic(comp) or
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
      if computation_building_block_utils.is_called_intrinsic(comp.result):
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
    if computation_building_block_utils.is_called_intrinsic(comp.result):
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
    if computation_building_block_utils.is_called_intrinsic(comp.argument):
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
    if computation_building_block_utils.is_called_intrinsic(comp.result):
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
    if computation_building_block_utils.is_called_intrinsic(comp.source):
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


def inline_block_locals(comp, variable_names=None):
  """Inlines the block variables in `comp` whitelisted by `variable_names`.

  Args:
    comp: The computation building block in which to perform the extractions.
      The names of lambda parameters and block variables in `comp` must be
      unique.
    variable_names: A Python list, tuple, or set representing the whitelist of
      variable names to inline; or None if all variables should be inlined.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    ValueError: If `comp` contains variables with non-unique names.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  check_has_unique_names(comp)
  if variable_names is not None:
    py_typecheck.check_type(variable_names, (list, tuple, set))

  def _should_inline_variable(name):
    return variable_names is None or name in variable_names

  def _should_transform(comp):
    return ((isinstance(comp, computation_building_blocks.Reference) and
             _should_inline_variable(comp.name)) or
            (isinstance(comp, computation_building_blocks.Block) and
             any(_should_inline_variable(name) for name, _ in comp.locals)))

  def _transform(comp, symbol_tree):
    """Returns a new transformed computation or `comp`."""
    if not _should_transform(comp):
      return comp, False
    if isinstance(comp, computation_building_blocks.Reference):
      try:
        value = symbol_tree.get_payload_with_name(comp.name).value
      except NameError:
        # This reference is unbound
        value = None
      # This identifies a variable bound by a Block as opposed to a Lambda.
      if value is not None:
        return value, True
      return comp, False
    elif isinstance(comp, computation_building_blocks.Block):
      variables = [(name, value)
                   for name, value in comp.locals
                   if not _should_inline_variable(name)]
      if not variables:
        comp = comp.result
      else:
        comp = computation_building_blocks.Block(variables, comp.result)
      return comp, True
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
  name_generator = computation_constructing_utils.unique_name_generator(comp)

  def _should_transform(comp):
    """Returns `True` if `comp` is a chained federated map."""
    if computation_building_block_utils.is_called_intrinsic(
        comp, (
            intrinsic_defs.FEDERATED_APPLY.uri,
            intrinsic_defs.FEDERATED_MAP.uri,
        )):
      outer_arg = comp.argument[1]
      if computation_building_block_utils.is_called_intrinsic(
          outer_arg, comp.function.uri):
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
      functions_name = six.next(name_generator)
      functions_ref = computation_building_blocks.Reference(
          functions_name, functions.type_signature)
      arg_name = six.next(name_generator)
      arg_type = comps[0].type_signature.parameter
      arg_ref = computation_building_blocks.Reference(arg_name, arg_type)
      arg = arg_ref
      for index, _ in enumerate(comps):
        fn_sel = computation_building_blocks.Selection(
            functions_ref, index=index)
        call = computation_building_blocks.Call(fn_sel, arg)
        arg = call
      fn = computation_building_blocks.Lambda(arg_ref.name,
                                              arg_ref.type_signature, call)
      return computation_building_blocks.Block(
          ((functions_ref.name, functions),), fn)

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


def merge_tuple_intrinsics(comp, uri):
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
             [fn=Tuple]       Lambda(arg)       [Comp(v1), Comp(v2), ...]
                 |                       \
        [Comp(f1), Comp(f2), ...]         Tuple
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

  This transformation is implemented to match the following intrinsics:

  * intrinsic_defs.FEDERATED_AGGREGATE.uri
  * intrinsic_defs.FEDERATED_APPLY.uri
  * intrinsic_defs.FEDERATED_BROADCAST.uri
  * intrinsic_defs.FEDERATED_MAP.uri

  Args:
    comp: The computation building block in which to perform the merges.
    uri: The URI of the intrinsic to merge.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(uri, six.string_types)
  expected_uri = (
      intrinsic_defs.FEDERATED_AGGREGATE.uri,
      intrinsic_defs.FEDERATED_APPLY.uri,
      intrinsic_defs.FEDERATED_BROADCAST.uri,
      intrinsic_defs.FEDERATED_MAP.uri,
  )
  if uri not in expected_uri:
    raise ValueError(
        'The value of `uri` is expected to be on of {}, found {}'.format(
            expected_uri, uri))
  name_generator = computation_constructing_utils.unique_name_generator(comp)

  def _should_transform(comp):
    return (isinstance(comp, computation_building_blocks.Tuple) and
            comp and computation_building_block_utils.is_called_intrinsic(
                comp[0], uri) and all(
                    computation_building_block_utils.is_called_intrinsic(
                        element, comp[0].function.uri) for element in comp))

  def _transform_args_with_type(comps, type_signature):
    """Transforms a Python `list` of computations.

    Given a computation containing `n` called intrinsics with `m` arguments,
    this function takes a Python `list` of computations `comps` containing the
    `m`-th argument from each computation `n` and creates a new computation
    representing th `m`-th arguments that should be passed to the called
    intrinsic of the transformed computation.

    Args:
      comps: A Python list of computations.
      type_signature: The type to use when determining how to transform the
        computations.

    Returns:
      A `computation_building_blocks.Block`.
    """
    if isinstance(type_signature, computation_types.FederatedType):
      return _transform_args_with_federated_types(comps, type_signature)
    elif isinstance(type_signature, computation_types.FunctionType):
      return _transform_args_with_functional_types(comps, type_signature)
    elif isinstance(type_signature, computation_types.AbstractType):
      return _transform_args_with_abstract_types(comps, type_signature)
    else:
      raise TypeError(
          'Expected a FederatedType, FunctionalType, or an AbstractType, '
          'found: {}'.format(type(type_signature)))

  def _transform_args_with_abstract_types(comps, type_signature):
    r"""Transforms a Python `list` of computations with abstract types.

    Tuple
    |
    [Comp, Comp, ...]

    Args:
      comps: A Python list of computations.
      type_signature: The type to use when determining how to transform the
        computations.

    Returns:
      A `computation_building_blocks.Tuple`.
    """
    del type_signature  # Unused
    return computation_building_blocks.Tuple(comps)

  def _transform_args_with_federated_types(comps, type_signature):
    r"""Transforms a Python `list` of computations with federated types.

    federated_zip(Tuple)
                  |
                  [Comp, Comp, ...]

    Args:
      comps: A Python list of computations.
      type_signature: The type to use when determining how to transform the
        computations.

    Returns:
      A `computation_building_blocks.Block`.
    """
    del type_signature  # Unused
    values = computation_building_blocks.Tuple(comps)
    return computation_constructing_utils.create_federated_zip(values)

  def _transform_args_with_functional_types(comps, type_signature):
    r"""Transforms a Python `list` of computations with functional types.

                    Block
                   /     \
         [fn=Tuple]       Lambda(arg)
             |                       \
    [Comp(f1), Comp(f2), ...]         Tuple
                                      |
                                 [Call,                   Call, ...]
                                 /    \                  /    \
                           Sel(0)      Sel(0)      Sel(1)     Sel(1)
                           |           |           |          |
                           Ref(fn)     Ref(arg)    Ref(fn)    Ref(arg)

    Args:
      comps: a Python list of computations.
      type_signature: The type to use when determining how to transform the
        computations.

    Returns:
      A `computation_building_blocks.Block`.
    """
    functions = computation_building_blocks.Tuple(comps)
    fn_name = six.next(name_generator)
    fn_ref = computation_building_blocks.Reference(fn_name,
                                                   functions.type_signature)
    if isinstance(type_signature.parameter, computation_types.NamedTupleType):
      arg_type = [[] for _ in range(len(type_signature.parameter))]
      for functional_comp in comps:
        named_type_signatures = anonymous_tuple.to_elements(
            functional_comp.type_signature.parameter)
        for index, (_, concrete_type) in enumerate(named_type_signatures):
          arg_type[index].append(concrete_type)
    else:
      arg_type = [e.type_signature.parameter for e in comps]
    arg_name = six.next(name_generator)
    arg_ref = computation_building_blocks.Reference(arg_name, arg_type)
    if isinstance(type_signature.parameter, computation_types.NamedTupleType):
      arg = computation_constructing_utils.create_zip(arg_ref)
    else:
      arg = arg_ref
    elements = []
    for index, functional_comp in enumerate(comps):
      sel_fn = computation_building_blocks.Selection(fn_ref, index=index)
      sel_arg = computation_building_blocks.Selection(arg, index=index)
      call = computation_building_blocks.Call(sel_fn, sel_arg)
      elements.append(call)
    calls = computation_building_blocks.Tuple(elements)
    result = computation_building_blocks.Lambda(arg_ref.name,
                                                arg_ref.type_signature, calls)
    return computation_building_blocks.Block(((fn_ref.name, functions),),
                                             result)

  def _transform_args(comp, type_signature):
    """Transforms the arguments from `comp`.

    Given a computation containing a tuple of intrinsics that can be merged,
    this function creates arguments that should be passed to the call of the
    transformed computation.

    Args:
      comp: The computation building block in which to perform the transform.
      type_signature: The type to use when determining how to transform the
        computations.

    Returns:
      A `computation_building_blocks.ComputationBuildingBlock` representing the
      transformed arguments from `comp`.
    """
    if isinstance(type_signature, computation_types.NamedTupleType):
      comps = [[] for _ in range(len(type_signature))]
      for _, call in anonymous_tuple.to_elements(comp):
        for index, arg in enumerate(call.argument):
          comps[index].append(arg)
      transformed_args = []
      for args, arg_type in zip(comps, type_signature):
        transformed_arg = _transform_args_with_type(args, arg_type)
        transformed_args.append(transformed_arg)
      return computation_building_blocks.Tuple(transformed_args)
    else:
      args = []
      for _, call in anonymous_tuple.to_elements(comp):
        args.append(call.argument)
      return _transform_args_with_type(args, type_signature)

  def _transform(comp):
    """Returns a new transformed computation or `comp`."""
    if not _should_transform(comp):
      return comp, False
    intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(uri)
    arg = _transform_args(comp, intrinsic_def.type_signature.parameter)
    named_comps = anonymous_tuple.to_elements(comp)
    parameter_type = computation_types.to_type(arg.type_signature)
    type_signature = [call.type_signature.member for _, call in named_comps]
    result_type = computation_types.FederatedType(
        type_signature, intrinsic_def.type_signature.result.placement,
        intrinsic_def.type_signature.result.all_equal)
    intrinsic_type = computation_types.FunctionType(parameter_type, result_type)
    intrinsic = computation_building_blocks.Intrinsic(uri, intrinsic_type)
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
            intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
            intrinsic_defs.FEDERATED_APPLY.uri,
            intrinsic_defs.SEQUENCE_MAP.uri,
        )):
      called_function = comp.argument[0]
      return computation_building_block_utils.is_identity_function(
          called_function)
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
    comp: The computation building block in which to perform the replacements.

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
  """Replaces all the bound reference names in `comp` with unique names.

  Notice that `uniquify_reference_names` simply leaves alone any reference
  which is unbound under `comp`.

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
      try:
        new_name = context_tree.get_payload_with_name(comp.name).new_name
        return computation_building_blocks.Reference(new_name,
                                                     comp.type_signature,
                                                     comp.context), True
      except NameError:
        return comp, False
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


class TFParser(object):
  """Callable taking subset of TFF AST constructs to CompiledComputations.

  When this function is applied via `transformation_utils.transform_postorder`
  to a TFF AST node satisfying its assumptions,  the tree under this node will
  be reduced to a single instance of
  `computation_building_blocks.CompiledComputation` representing the same
  logic.

  Notice that this function is designed to be applied to what is essentially
  a subtree of a larger TFF AST; once the processing on a single device has
  been aligned at the AST level, and placement separated from the logic of
  this processing, we should be left with a function wrapped via
  `federated_map` or `federated_apply` to a federated argument. It is this
  function which we need to reduce to TensorFlow, and it is to the root
  node of this function which we are looking to apply `TFParser`. Because of
  this, we assume that there is a lambda expression at the top of the AST
  we are looking to parse, as well as the rest of the assumptions below.

  1. All called lambdas have been converted to blocks.
  2. All blocks have been inlined; that is, there are no block/LET constructs
     remaining.
  3. All compiled computations are called.
  4. No compiled computations have been partially called; we believe this
     should be handled correctly today but we haven't reasoned explicitly about
     this possibility.
  5. The only leaf nodes present under `comp` are compiled computations and
     references to the argument of the top-level lambda which we are hoping to
     replace with a compiled computation. Further, every leaf node which is a
     reference has as its parent a `computation_building_blocks.Call`, whose
     associated function is a TF graph. This prevents us from needing to
     deal with arbitrary nesting of references and TF graphs, and significantly
     clarifies the reasoning. This can be accomplished by "decorating" the
     appropriate leaves with called identity TF graphs, the construction of
     which is provided by a utility module.
  6. There is only a single lambda binding any references present in the AST,
     and it is placed at the root of the AST to which we apply `TFParser`.
  7. There are no intrinsics present in the AST.
  """

  # TODO(b/133328350): Allow for this to take in multiple selections from a
  # single argument.

  def __init__(self):
    """Populates the parser library with mutually exclusive options."""
    self._parse_library = [
        compiled_computation_transforms.SelectionFromCalledTensorFlowBlock(),
        compiled_computation_transforms.LambdaWrappingGraph(),
        compiled_computation_transforms.TupleCalledGraphs(),
        compiled_computation_transforms.CalledCompositionOfTensorFlowBlocks(),
    ]

  def __call__(self, comp):
    """Transforms `comp` by checking all elements of the parser library.

    This function is roughly performing intermediate-code generation, taking
    TFF and generating TF. Calling this function is essentially checking the
    stack and selecting a semantic action based on its contents, and *only one*
    of these actions should be selected for a given computation.

    Notice that since the parser library contains mutually exclusive options,
    it is safe to return early.

    Args:
      comp: The `computation_building_blocks.ComputationBuildingBlock` to check
        for possibility of reduction according to the parsing library.

    Returns:
      A tuple whose first element is a possibly transformed version of `comp`,
      and whose second is a Boolean indicating whether or not `comp` was
      transformed. This is in conforming to the conventions of
      `transformation_utils.transform_postorder`.
    """
    py_typecheck.check_type(
        comp, computation_building_blocks.ComputationBuildingBlock)
    for option in self._parse_library:
      if option.should_transform(comp):
        transformed, ind = option.transform(comp)
        return transformed, ind
    return comp, False


def insert_called_tf_identity_at_leaves(comp):
  r"""Inserts an identity TF graph called on References under `comp`.

  For ease of reasoning about and proving completeness of TFF-to-TF
  translation capabilities, we will maintain the invariant that
  we constantly pass up the AST instances of the pattern:

                                    Call
                                  /      \
              CompiledComputation         Reference

  Any block of TFF reducible to TensorFlow must have a functional type
  signature without nested functions, and therefore we may assume there is
  a single Reference in the code we are parsing to TF. We continually push logic
  into the compiled computation as we make our way up the AST, preserving the
  pattern above; when we hit the lambda that binds this reference, we simply
  unwrap the call.

  To perform this process, we must begin with this pattern; otherwise there
  may be some arbitrary TFF constructs present between any occurrences of TF
  and the arguments to which they are applied, e.g. arbitrary selections from
  and nesting of tuples containing references.

  `insert_called_tf_identity_at_leaves` ensures that the pattern above is
  present at the leaves of any portion of the TFF AST which is destined to be
  reduced to TF.

  We detect such a destiny by checking for the existence of a
  `computation_building_blocks.Lambda` whose parameter and result type
  can both be bound into TensorFlow. This pattern is enforced here as
  parameter validation on `comp`.

  Args:
    comp: Instance of `computation_building_blocks.Lambda` whose AST we will
      traverse, replacing appropriate instances of
      `computation_building_blocks.Reference` with graphs representing thei
      identity function of the appropriate type called on the same reference.
      `comp` must declare a parameter and result type which are both able to be
      stamped in to a TensorFlow graph.

  Returns:
    A possibly modified  version of `comp`, where any references now have a
    parent of type `computation_building_blocks.Call` with function an instance
    of `computation_building_blocks.CompiledComputation`.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  if isinstance(comp, computation_building_blocks.CompiledComputation):
    return comp, False

  if not (isinstance(comp, computation_building_blocks.Lambda) and
          type_utils.is_tensorflow_compatible_type(comp.result.type_signature)
          and type_utils.is_tensorflow_compatible_type(comp.parameter_type)):
    raise ValueError(
        '`insert_called_tf_identity_at_leaves` should only be '
        'called on instances of '
        '`computation_building_blocks.Lambda` whose parameter '
        'and result types can both be stamped into TensorFlow '
        'graphs. You have called in on a {} of type signature {}.'.format(
            computation_building_blocks.compact_representation(comp),
            comp.type_signature))

  def _should_decorate(comp):
    return (isinstance(comp, computation_building_blocks.Reference) and
            type_utils.is_tensorflow_compatible_type(comp.type_signature))

  def _decorate(comp):
    identity_function = computation_constructing_utils.construct_compiled_identity(
        comp.type_signature)
    return computation_building_blocks.Call(identity_function, comp)

  def _decorate_if_reference_without_graph(comp):
    """Decorates references under `comp` if necessary."""
    if (isinstance(comp, computation_building_blocks.Tuple) and
        any(_should_decorate(x) for x in comp)):
      elems = []
      for x in anonymous_tuple.to_elements(comp):
        if _should_decorate(x[1]):
          elems.append((x[0], _decorate(x[1])))
        else:
          elems.append((x[0], x[1]))
      return computation_building_blocks.Tuple(elems), True
    elif (isinstance(comp, computation_building_blocks.Call) and not isinstance(
        comp.function, computation_building_blocks.CompiledComputation) and
          _should_decorate(comp.argument)):
      arg = _decorate(comp.argument)
      return computation_building_blocks.Call(comp.function, arg), True
    elif (isinstance(comp, computation_building_blocks.Selection) and
          _should_decorate(comp.source)):
      return computation_building_blocks.Selection(
          _decorate(comp.source), name=comp.name, index=comp.index), True
    elif (isinstance(comp, computation_building_blocks.Lambda) and
          _should_decorate(comp.result)):
      return computation_building_blocks.Lambda(comp.parameter_name,
                                                comp.parameter_type,
                                                _decorate(comp.result)), True
    elif isinstance(comp, computation_building_blocks.Block) and (
        any(_should_decorate(x[1]) for x in comp.locals) or
        _should_decorate(comp.result)):
      new_locals = []
      for x in comp.locals:
        if _should_decorate(x[1]):
          new_locals.append((x[0], _decorate(x[1])))
        else:
          new_locals.append((x[0], x[1]))
      new_result = comp.result
      if _should_decorate(comp.result):
        new_result = _decorate(comp.result)
      return computation_building_blocks.Block(new_locals, new_result), True
    return comp, False

  return transformation_utils.transform_postorder(
      comp, _decorate_if_reference_without_graph)


def check_has_single_placement(comp, single_placement):
  """Checks that the AST of `comp` contains only `single_placement`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`.
    single_placement: Instance of `placement_literals.PlacementLiteral` which
      should be the only placement present under `comp`.

  Raises:
    ValueError: If the AST under `comp` contains any
    `computation_types.FederatedType` other than `single_placement`.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(single_placement, placement_literals.PlacementLiteral)

  def _check_single_placement(comp):
    """Checks that the placement in `type_spec` matches `single_placement`."""
    if (isinstance(comp.type_signature, computation_types.FederatedType) and
        comp.type_signature.placement != single_placement):
      raise ValueError(
          'Comp contains a placement other than {}; '
          'placement {} on comp {} inside the structure. '.format(
              single_placement, comp.type_signature.placement,
              computation_building_blocks.compact_representation(comp)))
    return comp, False

  transformation_utils.transform_postorder(comp, _check_single_placement)


def unwrap_placement(comp):
  """Strips `comp`'s placement, returning a single call to map, apply or value.

  For this purpose it is necessary to assume that all processing under `comp`
  is happening at a single placement.

  The other assumptions on inputs of `unwrap_placement` are enumerated as
  follows:

  1. There is at most one unbound reference under `comp`, which is of federated
     type.
  2. The only intrinsics present here are apply or map, zip,
     and federated_value_at_*.
  3. The type signature of `comp` is federated.
  4. There are no instances of `computation_building_blocks.Data` of federated
     type under `comp`; how these would be handled by a function such as this
     is not entirely clear.

  Under these conditions, `unwrap_placement` will produce a single call to
  federated_map, federated_apply or federated_value, depending on the placement
  and type signature of `comp`. Other than this single map or apply, no
  intrinsics will remain under `comp`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      satisfying the assumptions above.

  Returns:
    A modified version of `comp`, whose root is a single called
    intrinsic, and containing no other intrinsics. Equivalent
    to `comp`.

  Raises:
    TypeError: If the lone unbound reference under `comp` is not of federated
    type, `comp` itself is not of federated type, or `comp` is not a building
    block.
    ValueError: If we encounter a placement other than the one declared by
    `comp.type_signature`, an intrinsic not present in the whitelist above, or
    `comp` contains more than one unbound reference.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  single_placement = comp.type_signature.placement

  check_has_single_placement(comp, single_placement)

  name_generator = computation_constructing_utils.unique_name_generator(comp)

  all_unbound_references = get_map_of_unbound_references(comp)
  root_unbound_references = all_unbound_references[comp]

  if len(root_unbound_references) > 1:
    raise ValueError(
        '`unwrap_placement` can only handle computations with at most a single '
        'unbound reference; you have passed in the computation {} with {} '
        'unbound references.'.format(
            computation_building_blocks.compact_representation(comp),
            len(root_unbound_references)))

  if len(root_unbound_references) == 1:
    unbound_reference_name = root_unbound_references.pop()
  else:
    unbound_reference_name = None

  def _rename_unbound_variable(comp, unbound_variable_name):
    """Reads info about the unbound variable, and renames it uniquely.

    The unique rename is simply to preserve uniqueness of names if this
    property is present in the argument to `unwrap_placement`, since we will
    eventually be binding a new reference of non-federated type in place
    of this federated unbound reference.

    Args:
      comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
        with at most a single unbound reference.
      unbound_variable_name: The name of the lone unbound variable present under
        `comp`.

    Returns:
      A tuple, whose first element a is possibly transformed version of `comp`
      with its unbound variable renamed to a name which is globally unique
      within `comp`, and its second element a tuple containing the new name
      given to the unbound reference, and the type of this unbound reference.
    """
    unbound_reference_name_and_type_pair = [(None, None)]

    class _UnboundVariableIdentifier(transformation_utils.BoundVariableTracker):
      """transformation_utils.SymbolTree node for tracking unbound variables."""

      def __init__(self, name, value):
        super(_UnboundVariableIdentifier, self).__init__(name, value)
        self.unbound = False

      def __str__(self):
        return ''

      def update(self, x):
        del x  # Unused
        self.unbound = True

    symbol_tree = transformation_utils.SymbolTree(_UnboundVariableIdentifier)
    symbol_tree.ingest_variable_binding(unbound_variable_name, None)
    symbol_tree.update_payload_tracking_reference(
        computation_building_blocks.Reference(
            unbound_variable_name, computation_types.AbstractType('T')))

    def _should_transform(comp, symbol_tree):
      return (isinstance(comp, computation_building_blocks.Reference) and
              comp.name == unbound_variable_name and
              symbol_tree.get_payload_with_name(comp.name).unbound)

    def _rename_unbound_variable(comp, symbol_tree):
      """Updates the nonlocal tracker, and renames the unbound variable."""
      if not _should_transform(comp, symbol_tree):
        return comp, False
      if unbound_reference_name_and_type_pair[0][1] is None:
        name = six.next(name_generator)
        unbound_reference_name_and_type_pair[0] = (name, comp.type_signature)
      else:
        name = unbound_reference_name_and_type_pair[0][0]
      return computation_building_blocks.Reference(name,
                                                   comp.type_signature), True

    renamed_comp, _ = transformation_utils.transform_postorder_with_symbol_bindings(
        comp, _rename_unbound_variable, symbol_tree)
    return renamed_comp, unbound_reference_name_and_type_pair[0]

  def _remove_placement(comp):
    """Unwraps placement from `comp`.

    `_remove_placement` embodies the main transform logic in
    `unwrap_placement`, performing a pure AST transformation to replace
    any nodes of federated type with equivalent non-federated versions.
    Whether or not it is safe to do this is left to `unwrap_placement` to
    handle.

    One note on the implementation: the four cases in the internal `_transform`
    switch here exactly case for the building blocks which explicitly take type
    signatures as arguments to their constructors.

    Args:
      comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
        from which we wish to remove placement.

    Returns:
      A transformed version of comp with its placements removed.

    Raises:
      NotImplementedError: In case a node of type
        `computation_building_blocks.Data` is encountered in the AST, as
        handling of data objects is not yet implemented in TFF and so it is
        unclear what this function should do in that case.
    """

    def _remove_placement_from_type(type_spec):
      if isinstance(type_spec, computation_types.FederatedType):
        return type_spec.member, True
      else:
        return type_spec, False

    def _remove_reference_placement(comp):
      """Unwraps placement from references and updates unbound reference info."""
      new_type, _ = type_utils.transform_type_postorder(
          comp.type_signature, _remove_placement_from_type)
      return computation_building_blocks.Reference(comp.name, new_type)

    def _replace_intrinsics_with_functions(comp):
      """Helper to remove intrinsics from the AST."""
      if (comp.uri == intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri or
          comp.uri == intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri or
          comp.uri == intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri or
          comp.uri == intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri):
        arg_name = six.next(name_generator)
        arg_type = comp.type_signature.result.member
        val = computation_building_blocks.Reference(arg_name, arg_type)
        lam = computation_building_blocks.Lambda(arg_name, arg_type, val)
        return lam
      elif comp.uri not in (intrinsic_defs.FEDERATED_MAP.uri,
                            intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
                            intrinsic_defs.FEDERATED_APPLY.uri):
        raise ValueError('Disallowed intrinsic: {}'.format(comp))
      arg_name = six.next(name_generator)
      tuple_ref = computation_building_blocks.Reference(arg_name, [
          comp.type_signature.parameter[0],
          comp.type_signature.parameter[1].member,
      ])
      fn = computation_building_blocks.Selection(tuple_ref, index=0)
      arg = computation_building_blocks.Selection(tuple_ref, index=1)
      called_fn = computation_building_blocks.Call(fn, arg)
      return computation_building_blocks.Lambda(arg_name,
                                                tuple_ref.type_signature,
                                                called_fn)

    def _remove_lambda_placement(comp):
      """Removes placement from Lambda's parameter."""
      new_parameter_type, _ = type_utils.transform_type_postorder(
          comp.parameter_type, _remove_placement_from_type)
      return computation_building_blocks.Lambda(comp.parameter_name,
                                                new_parameter_type, comp.result)

    def _transform(comp):
      """Dispatches to helpers above."""
      if isinstance(comp, computation_building_blocks.Reference):
        return _remove_reference_placement(comp), True
      elif isinstance(comp, computation_building_blocks.Intrinsic):
        return _replace_intrinsics_with_functions(comp), True
      elif isinstance(comp, computation_building_blocks.Lambda):
        return _remove_lambda_placement(comp), True
      elif (isinstance(comp, computation_building_blocks.Data) and
            isinstance(comp.type_signature, computation_types.FederatedType)):
        # TODO(b/135126947): Design and implement Data constructs.
        raise NotImplementedError
      return comp, False

    return transformation_utils.transform_postorder(comp, _transform)

  if unbound_reference_name is None:
    unbound_variable_renamed = comp
    unbound_reference_name = None
    unbound_reference_type = None
  else:
    (unbound_variable_renamed,
     unbound_reference_info) = _rename_unbound_variable(comp,
                                                        unbound_reference_name)
    (new_reference_name, unbound_reference_type) = unbound_reference_info

    if not isinstance(unbound_reference_type, computation_types.FederatedType):
      raise TypeError('The lone unbound reference is not of federated type; '
                      'this is disallowed. '
                      'The unbound type is {}'.format(unbound_reference_type))

  placement_removed, _ = _remove_placement(unbound_variable_renamed)

  if unbound_reference_name is None:
    return computation_constructing_utils.create_federated_value(
        placement_removed, single_placement), True

  ref_to_fed_arg = computation_building_blocks.Reference(
      unbound_reference_name, unbound_reference_type)

  lambda_wrapping_placement_removal = computation_building_blocks.Lambda(
      new_reference_name, unbound_reference_type.member, placement_removed)

  called_intrinsic = computation_constructing_utils.create_federated_map_or_apply(
      lambda_wrapping_placement_removal, ref_to_fed_arg)

  return called_intrinsic, True


def replace_all_intrinsics_with_bodies(comp, context_stack):
  """Iterates over all intrinsic bodies, inlining the intrinsics in `comp`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock` in
      which we wish to replace all intrinsics with their bodies.
    context_stack: Instance of `context_stack_base.ContextStack`, the context
      stack to use for the bodies of the intrinsics.

  Returns:
    Instance of `computation_building_blocks.ComputationBuildingBlock` with all
    the intrinsics from `intrinsic_bodies.py` inlined with their bodies, along
    with a Boolean indicating whether there was any inlining in fact done.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  bodies = intrinsic_bodies.get_intrinsic_bodies(context_stack)
  transformed = False
  for uri, body in six.iteritems(bodies):
    comp, uri_found = replace_intrinsic_with_callable(comp, uri, body,
                                                      context_stack)
    transformed = transformed or uri_found
  return comp, transformed


def check_has_unique_names(comp):
  if not transformation_utils.has_unique_names(comp):
    raise ValueError(
        'This transform should only be called after we have uniquified all '
        '`computation_building_blocks.Reference` names, since we may be moving '
        'computations with unbound references under constructs which bind '
        'those references.')


def get_map_of_unbound_references(comp):
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


def check_intrinsics_whitelisted_for_reduction(comp):
  """Checks whitelist of intrinsics reducible to aggregate or broadcast.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock` to
      check for presence of intrinsics not currently immediately reducible to
      `FEDERATED_AGGREGATE` or `FEDERATED_BROADCAST`, or local processing.

  Raises:
    ValueError: If we encounter an intrinsic under `comp` that is not
    whitelisted as currently reducible.
  """
  # TODO(b/135930668): Factor this and other non-transforms (e.g.
  # `check_has_unique_names` out of this file into a structure specified for
  # static analysis of ASTs.
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  uri_whitelist = (
      intrinsic_defs.FEDERATED_AGGREGATE.uri,
      intrinsic_defs.FEDERATED_APPLY.uri,
      intrinsic_defs.FEDERATED_BROADCAST.uri,
      intrinsic_defs.FEDERATED_MAP.uri,
      intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
      intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri,
      intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri,
      intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri,
      intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri,
  )

  def _check_whitelisted(comp):
    if isinstance(comp, computation_building_blocks.Intrinsic
                 ) and comp.uri not in uri_whitelist:
      raise ValueError(
          'Encountered an Intrinsic not currently reducible to aggregate or '
          'broadcast, the intrinsic {}'.format(
              computation_building_blocks.compact_representation(comp)))
    return comp, False

  transformation_utils.transform_postorder(comp, _check_whitelisted)
