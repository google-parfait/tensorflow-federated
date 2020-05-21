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
"""A library of transformation functions for ASTs."""

import typing

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_transformations


class TransformationError(Exception):
  """Raised when a transformation fails."""


def _apply_transforms(comp, transforms):
  """Applies all `transforms` in a single walk of `comp`.

  This function is private for a reason; TFF does not intend to expose the
  capability to chain arbitrary transformations in this way, since the
  application of one transformation may cause the resulting AST to violate the
  assumptions of another. This function should be used quite selectively and
  considered extensively in order to avoid such subtle issues.

  Args:
    comp: An instance of `building_blocks.ComputationBuildingBlock` to transform
      with all elements of `transforms`.
    transforms: An instance of `transformation_utils.TransformSpec` or iterable
      thereof, the transformations to apply to `comp`.

  Returns:
    A transformed version of `comp`, with all transformations in `transforms`
    applied.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  if isinstance(transforms, transformation_utils.TransformSpec):
    transforms = [transforms]
  else:
    for transform in transforms:
      py_typecheck.check_type(transform, transformation_utils.TransformSpec)

  def _transform(comp):
    modified = False
    for transform in transforms:
      comp, transform_modified = transform.transform(comp)
      modified = modified or transform_modified
    return comp, modified

  return transformation_utils.transform_postorder(comp, _transform)


class ExtractComputation(transformation_utils.TransformSpec):
  """Extracts a computation if all referenced variables are free (unbound).

  This transforms a computation which matches the `predicate` or is a Block, and
  replaces the computations with a LET construct if it doesn't reference any
  variables bound by the current scope.

  Both the `parameter_name` of a `building_blocks.Lambda` and the name of
  any variable defined by a `building_blocks.Block` can affect the scope in
  which a reference in computation is bound.

  Note: This function extracts `computation_building_block.Block` because block
  variables can restrict the scope in which computations are bound.
  """

  def __init__(self, comp, predicate):
    """Constructs a new instance.

    Args:
      comp: The computation building block in which to perform the extractions.
        The names of lambda parameters and block variables in `comp` must be
        unique.
      predicate: A function that takes a single computation building block as a
        argument and returns `True` if the computation should be extracted and
        `False` otherwise.

    Raises:
      TypeError: If types do not match.
      ValueError: If `comp` contains variables with non-unique names.
    """
    super().__init__()
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    tree_analysis.check_has_unique_names(comp)
    self._name_generator = building_block_factory.unique_name_generator(comp)
    self._predicate = predicate
    self._unbound_references = transformation_utils.get_map_of_unbound_references(
        comp)

  def _contains_unbound_reference(self, comp, names):
    """Returns `True` if `comp` contains unbound references to `names`.

    This function will update the non-local `_unbound_references` captured from
    the parent context if `comp` is not contained in that collection. This can
    happen when new computations are created and added to the AST.

    Args:
      comp: The computation building block to test.
      names: A Python string or a list, tuple, or set of Python strings.
    """
    if isinstance(names, str):
      names = (names,)
    if comp not in self._unbound_references:
      references = transformation_utils.get_map_of_unbound_references(comp)
      self._unbound_references.update(references)
    return any(n in self._unbound_references[comp] for n in names)

  def _passes_test_or_block(self, comp):
    """Returns `True` if `comp` matches the `predicate` or is a block."""
    return self._predicate(comp) or isinstance(comp, building_blocks.Block)

  def should_transform(self, comp):
    """Returns `True` if `comp` should be transformed.

    The following `_extract_intrinsic_*` methods all depend on being invoked
    after `should_transform` evaluates to `True` for a given `comp`. Because of
    this certain assumptions are made:

    * transformation functions will transform a given `comp`
    * block variables are guaranteed to not be empty

    Args:
      comp: The computation building block in which to test.
    """
    if isinstance(comp, building_blocks.Block):
      return (self._passes_test_or_block(comp.result) or
              any(isinstance(e, building_blocks.Block) for _, e in comp.locals))
    elif isinstance(comp, building_blocks.Call):
      return (self._passes_test_or_block(comp.function) or
              self._passes_test_or_block(comp.argument))
    elif isinstance(comp, building_blocks.Lambda):
      if self._predicate(comp.result):
        return True
      if isinstance(comp.result, building_blocks.Block):
        for index, (_, variable) in enumerate(comp.result.locals):
          names = [n for n, _ in comp.result.locals[:index]]
          if (not self._contains_unbound_reference(variable,
                                                   comp.parameter_name) and
              not self._contains_unbound_reference(variable, names)):
            return True
    elif isinstance(comp, building_blocks.Selection):
      return self._passes_test_or_block(comp.source)
    elif isinstance(comp, building_blocks.Tuple):
      return any(self._passes_test_or_block(e) for e in comp)
    return False

  def _extract_from_block(self, comp):
    """Returns a new computation with all intrinsics extracted."""
    if self._predicate(comp.result):
      name = next(self._name_generator)
      variables = comp.locals
      variables.append((name, comp.result))
      result = building_blocks.Reference(name, comp.result.type_signature)
    elif isinstance(comp.result, building_blocks.Block):
      variables = comp.locals + comp.result.locals
      result = comp.result.result
    else:
      variables = comp.locals
      result = comp.result

    def _remove_blocks_from_variables(variables):
      new_variables = []
      for name, variable in variables:
        if isinstance(variable, building_blocks.Block):
          new_variables.extend(variable.locals)
          new_variables.append((name, variable.result))
        else:
          new_variables.append((name, variable))
      return new_variables

    variables = _remove_blocks_from_variables(variables)
    return building_blocks.Block(variables, result)

  def _extract_from_call(self, comp):
    """Returns a new computation with all intrinsics extracted."""
    variables = []
    if self._predicate(comp.function):
      name = next(self._name_generator)
      variables.append((name, comp.function))
      function = building_blocks.Reference(name, comp.function.type_signature)
    elif isinstance(comp.function, building_blocks.Block):
      block = comp.function
      variables.extend(block.locals)
      function = block.result
    else:
      function = comp.function
    if comp.argument is not None:
      if self._predicate(comp.argument):
        name = next(self._name_generator)
        variables.append((name, comp.argument))
        argument = building_blocks.Reference(name, comp.argument.type_signature)
      elif isinstance(comp.argument, building_blocks.Block):
        block = comp.argument
        variables.extend(block.locals)
        argument = block.result
      else:
        argument = comp.argument
    else:
      argument = None
    call = building_blocks.Call(function, argument)
    block = building_blocks.Block(variables, call)
    return self._extract_from_block(block)

  def _extract_from_lambda(self, comp):
    """Returns a new computation with all intrinsics extracted."""
    if self._predicate(comp.result):
      name = next(self._name_generator)
      variables = [(name, comp.result)]
      result = building_blocks.Reference(name, comp.result.type_signature)
      if not self._contains_unbound_reference(comp.result, comp.parameter_name):
        fn = building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                    result)
        block = building_blocks.Block(variables, fn)
        return self._extract_from_block(block)
      else:
        block = building_blocks.Block(variables, result)
        block = self._extract_from_block(block)
        return building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                      block)
    else:
      block = comp.result
      extracted_variables = []
      retained_variables = []
      for name, variable in block.locals:
        names = [n for n, _ in retained_variables]
        if (not self._contains_unbound_reference(variable, comp.parameter_name)
            and not self._contains_unbound_reference(variable, names)):
          extracted_variables.append((name, variable))
        else:
          retained_variables.append((name, variable))
      if retained_variables:
        result = building_blocks.Block(retained_variables, block.result)
      else:
        result = block.result
      fn = building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                  result)
      block = building_blocks.Block(extracted_variables, fn)
      return self._extract_from_block(block)

  def _extract_from_selection(self, comp):
    """Returns a new computation with all intrinsics extracted."""
    if self._predicate(comp.source):
      name = next(self._name_generator)
      variables = [(name, comp.source)]
      source = building_blocks.Reference(name, comp.source.type_signature)
    else:
      block = comp.source
      variables = block.locals
      source = block.result
    selection = building_blocks.Selection(
        source, name=comp.name, index=comp.index)
    block = building_blocks.Block(variables, selection)
    return self._extract_from_block(block)

  def _extract_from_tuple(self, comp):
    """Returns a new computation with all intrinsics extracted."""
    variables = []
    elements = []
    for name, element in anonymous_tuple.iter_elements(comp):
      if self._passes_test_or_block(element):
        variable_name = next(self._name_generator)
        variables.append((variable_name, element))
        ref = building_blocks.Reference(variable_name, element.type_signature)
        elements.append((name, ref))
      else:
        elements.append((name, element))
    tup = building_blocks.Tuple(elements)
    block = building_blocks.Block(variables, tup)
    return self._extract_from_block(block)

  def transform(self, comp):
    """Returns a new transformed computation or `comp`."""
    if not self.should_transform(comp):
      return comp, False
    if isinstance(comp, building_blocks.Block):
      comp = self._extract_from_block(comp)
    elif isinstance(comp, building_blocks.Call):
      comp = self._extract_from_call(comp)
    elif isinstance(comp, building_blocks.Lambda):
      comp = self._extract_from_lambda(comp)
    elif isinstance(comp, building_blocks.Selection):
      comp = self._extract_from_selection(comp)
    elif isinstance(comp, building_blocks.Tuple):
      comp = self._extract_from_tuple(comp)
    return comp, True


def extract_computations(comp):
  """Extracts subcomputations to a binding in the outermost-possible scope.

  Subcomputations that reference only free variables (variables bound outside
  `comp`) will be extracted to root-level. Computations which reference bound
  variables will be extracted to the scope in which the innermost variable
  they reference is bound.

  Args:
    comp: The computation building block on which to perform the transformation.

  Returns:
    A computation representing `comp` with the transformation applied.
  """

  def _predicate(comp):
    return not isinstance(comp, building_blocks.Reference)

  return _apply_transforms(comp, ExtractComputation(comp, _predicate))


def extract_intrinsics(comp):
  """Extracts intrinsics to a binding in the outermost-possible scope.

  Intrinsics that reference only free variables (variables bound outside
  `comp`) will be extracted to root-level. Intrinsics which reference bound
  variables will be extracted to the scope in which the innermost variable
  they depend on is bound.

  Args:
    comp: The computation building block in which to perform the transformation.

  Returns:
    A new computation representing `comp` with the transformation applied.
  """

  def _predicate(comp):
    return building_block_analysis.is_called_intrinsic(comp)

  return _apply_transforms(comp, ExtractComputation(comp, _predicate))


class InlineBlock(transformation_utils.TransformSpec):
  """Inlines the block variables in `comp` whitelisted by `variable_names`.

  Each invocation of the `transform` method checks for presence of a
  block-bound `building_blocks.Reference`, and inlines this
  reference with its appropriate value.
  """

  def __init__(self, comp, variable_names=None):
    """Initializes the block inliner.

    Checks that `comp` has unique names, and that `variable_names` is an
    iterable of string types.

    Args:
      comp: The top-level computation to inline.
      variable_names: The variable names to inline. If `None`, inlines all
        variables.

    Raises:
      ValueError: If `comp` contains variables with non-unique names.
      TypeError: If `variable_names` is a non-`list`, `set` or `tuple`, or
        contains anything other than strings.
    """
    super().__init__(global_transform=True)
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    tree_analysis.check_has_unique_names(comp)
    if variable_names is not None:
      py_typecheck.check_type(variable_names, (list, tuple, set))
      for name in variable_names:
        py_typecheck.check_type(name, str)
    self._variable_names = variable_names

  def _should_inline_variable(self, name):
    return self._variable_names is None or name in self._variable_names

  def should_transform(self, comp):
    return ((isinstance(comp, building_blocks.Reference) and
             self._should_inline_variable(comp.name)) or
            (isinstance(comp, building_blocks.Block) and any(
                self._should_inline_variable(name) for name, _ in comp.locals)))

  def transform(self, comp, symbol_tree):
    if not self.should_transform(comp):
      return comp, False
    if isinstance(comp, building_blocks.Reference):
      payload = symbol_tree.get_payload_with_name(comp.name)
      if payload is None:
        value = None
      else:
        value = payload.value
      # This identifies a variable bound by a Block as opposed to a Lambda.
      if value is not None:
        return value, True
      return comp, False
    elif isinstance(comp, building_blocks.Block):
      variables = [(name, value)
                   for name, value in comp.locals
                   if not self._should_inline_variable(name)]
      if not variables:
        comp = comp.result
      else:
        comp = building_blocks.Block(variables, comp.result)
      return comp, True
    return comp, False


def inline_block_locals(comp, variable_names=None):
  """Inlines the block variables in `comp` whitelisted by `variable_names`."""
  symbol_tree = transformation_utils.SymbolTree(
      transformation_utils.ReferenceCounter)
  transform_spec = InlineBlock(comp, variable_names)
  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, transform_spec.transform, symbol_tree)


class InlineSelectionsFromTuples(transformation_utils.TransformSpec):
  """Inlines all Tuple-bound variables which are referenced via Selections.

  This should be used as a preprocessing and optimization step, since this can
  allow for large tuples which are only referenced as selections to be removed
  in another pass. Notice this transform makes no effort to remove these tuples,
  as it does nothing to guarantee that the tuples it inlines are not referenced
  elsewhere.

  This is implemented as a stanadlone transform, as opposed to simply extending
  `ReplaceSelectionFromTuple`, in order to preserve ease of chaining that
  transformation together with inlining all or some block locals.
  """

  def __init__(self):
    super().__init__(global_transform=True)

  def should_transform(self, comp, symbol_tree):
    if isinstance(comp, building_blocks.Selection) and isinstance(
        comp.source, building_blocks.Tuple):
      return True
    elif (isinstance(comp, building_blocks.Selection) and
          isinstance(comp.source, building_blocks.Reference)):
      resolved = symbol_tree.get_payload_with_name(comp.source.name)
      return resolved is not None and isinstance(resolved.value,
                                                 building_blocks.Tuple)
    return False

  def transform(self, comp, symbol_tree):
    if not self.should_transform(comp, symbol_tree):
      return comp, False
    if isinstance(comp.source, building_blocks.Tuple):
      tup = comp.source
    else:
      tup = symbol_tree.get_payload_with_name(comp.source.name).value
    if comp.index is None:
      element_to_inline = getattr(tup, comp.name)
    else:
      element_to_inline = tup[comp.index]
    return element_to_inline, True


def inline_selections_from_tuple(comp):
  symbol_tree = transformation_utils.SymbolTree(
      transformation_utils.TrackRemovedReferences)
  transform_spec = InlineSelectionsFromTuples()
  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, transform_spec.transform, symbol_tree)


class MergeChainedBlocks(transformation_utils.TransformSpec):
  r"""Merges chained blocks into one block.

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

  Preserving the relative ordering of any locals declarations, which preserves
  scoping rules.

  Notice that because TFF Block constructs bind their variables in sequence, it
  is completely safe to add the locals lists together in this implementation.
  """

  def should_transform(self, comp):
    """Returns `True` if `comp` is a block and its result is a block."""
    return (isinstance(comp, building_blocks.Block) and
            isinstance(comp.result, building_blocks.Block))

  def transform(self, comp):
    """Returns a new transformed computation or `comp`."""
    if not self.should_transform(comp):
      return comp, False
    comp = building_blocks.Block(comp.locals + comp.result.locals,
                                 comp.result.result)
    return comp, True


def merge_chained_blocks(comp):
  """Merges chained blocks into one block."""
  return _apply_transforms(comp, MergeChainedBlocks())


class MergeChainedFederatedMapsOrApplys(transformation_utils.TransformSpec):
  r"""Merges chained federated maps or federated apply into one structure.

  This transform matches the following pattern, and replaces the following
  computation containing two federated map intrinsics:

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
  """

  def __init__(self, comp):
    """Constructs a new instance.

    Args:
      comp: The computation building block in which to perform the merges.

    Raises:
      TypeError: If types do not match.
    """
    super().__init__()
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    self._name_generator = building_block_factory.unique_name_generator(comp)

  def should_transform(self, comp):
    """Returns `True` if `comp` is a chained federated map."""
    if building_block_analysis.is_called_intrinsic(comp, (
        intrinsic_defs.FEDERATED_APPLY.uri,
        intrinsic_defs.FEDERATED_MAP.uri,
    )):
      outer_arg = comp.argument[1]
      if building_block_analysis.is_called_intrinsic(outer_arg,
                                                     comp.function.uri):
        return True
    return False

  def _create_block_to_chained_calls(self, comps):
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
      A `building_blocks.Block`.
    """
    functions = building_blocks.Tuple(comps)
    functions_name = next(self._name_generator)
    functions_ref = building_blocks.Reference(functions_name,
                                              functions.type_signature)
    arg_name = next(self._name_generator)
    arg_type = comps[0].type_signature.parameter
    arg_ref = building_blocks.Reference(arg_name, arg_type)
    arg = arg_ref
    for index, _ in enumerate(comps):
      fn_sel = building_blocks.Selection(functions_ref, index=index)
      call = building_blocks.Call(fn_sel, arg)
      arg = call
    fn = building_blocks.Lambda(arg_ref.name, arg_ref.type_signature, call)
    return building_blocks.Block(((functions_ref.name, functions),), fn)

  def transform(self, comp):
    """Returns a new transformed computation or `comp`."""
    if not self.should_transform(comp):
      return comp, False
    block = self._create_block_to_chained_calls((
        comp.argument[1].argument[0],
        comp.argument[0],
    ))
    arg = building_blocks.Tuple([
        block,
        comp.argument[1].argument[1],
    ])
    intrinsic_type = computation_types.FunctionType(
        arg.type_signature, comp.function.type_signature.result)
    intrinsic = building_blocks.Intrinsic(comp.function.uri, intrinsic_type)
    comp = building_blocks.Call(intrinsic, arg)
    return comp, True


def remove_duplicate_building_blocks(comp):
  """Composite transformation to remove duplicated building blocks."""
  mutated = False
  for transform in [
      replace_called_lambda_with_block,
      remove_mapped_or_applied_identity,
      uniquify_reference_names,
      extract_computations,
      remove_duplicate_block_locals,
  ]:
    comp, comp_mutated = transform(comp)
    mutated = mutated or comp_mutated
  return comp, mutated


def merge_chained_federated_maps_or_applys(comp):
  """Merges chained federated maps or federated apply into one structure."""
  return _apply_transforms(comp, MergeChainedFederatedMapsOrApplys(comp))


class MergeTupleIntrinsics(transformation_utils.TransformSpec):
  r"""Merges a tuple of called intrinsics into one called intrinsic.

  This transform matches the following pattern, and replaces the following
  computation containing a tuple of called intrinsics all representing the same
  operation:

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

  Note: This is just an example of what this transformation would look like when
  applied to a tuple of federated maps. The components `f1`, `f2`, `v1`, and
  `v2` and the number of those components are not important.

  This transformation is implemented to match the following intrinsics:

  * intrinsic_defs.FEDERATED_AGGREGATE.uri
  * intrinsic_defs.FEDERATED_APPLY.uri
  * intrinsic_defs.FEDERATED_BROADCAST.uri
  * intrinsic_defs.FEDERATED_MAP.uri
  * intrinsic_defs.FEDERATED_SECURE_SUM.uri
  * intrinsic_defs.FEDERATED_SUM.uri
  """

  def __init__(self, comp, uri):
    """Constructs a new instance.

    Args:
      comp: The computation building block in which to perform the merges.
      uri: The URI of the intrinsic to merge.

    Raises:
      TypeError: If types do not match.
      ValueError: If the `uri` has an unexpected value.
    """
    super().__init__()
    py_typecheck.check_type(uri, str)
    self._name_generator = building_block_factory.unique_name_generator(comp)
    expected_uri = (
        intrinsic_defs.FEDERATED_AGGREGATE.uri,
        intrinsic_defs.FEDERATED_APPLY.uri,
        intrinsic_defs.FEDERATED_BROADCAST.uri,
        intrinsic_defs.FEDERATED_MAP.uri,
        intrinsic_defs.FEDERATED_SECURE_SUM.uri,
        intrinsic_defs.FEDERATED_SUM.uri,
    )
    if uri not in expected_uri:
      raise ValueError(
          'The value of `uri` is expected to be on of {}, found {}'.format(
              expected_uri, uri))
    self._uri = uri

  def should_transform(self, comp):
    return (isinstance(comp, building_blocks.Tuple) and comp and
            building_block_analysis.is_called_intrinsic(comp[0], self._uri) and
            all(
                building_block_analysis.is_called_intrinsic(
                    element, comp[0].function.uri) for element in comp))

  def _transform_args_with_type(self, comps, type_signature):
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
      A `building_blocks.Block`.
    """
    if isinstance(type_signature, computation_types.FederatedType):
      return self._transform_args_with_federated_types(comps, type_signature)
    elif isinstance(type_signature, computation_types.FunctionType):
      return self._transform_args_with_functional_types(comps, type_signature)
    elif isinstance(type_signature, computation_types.AbstractType):
      return self._transform_args_with_abstract_types(comps, type_signature)
    else:
      raise TypeError(
          'Expected a FederatedType, FunctionalType, or an AbstractType, '
          'found: {}'.format(type(type_signature)))

  def _transform_args_with_abstract_types(self, comps, type_signature):
    r"""Transforms a Python `list` of computations with abstract types.

    Tuple
    |
    [Comp, Comp, ...]

    Args:
      comps: A Python list of computations.
      type_signature: The type to use when determining how to transform the
        computations.

    Returns:
      A `building_blocks.Tuple`.
    """
    del type_signature  # Unused
    return building_blocks.Tuple(comps)

  def _transform_args_with_federated_types(self, comps, type_signature):
    r"""Transforms a Python `list` of computations with federated types.

    federated_zip(Tuple)
                  |
                  [Comp, Comp, ...]

    Args:
      comps: A Python list of computations.
      type_signature: The type to use when determining how to transform the
        computations.

    Returns:
      A `building_blocks.Block`.
    """
    del type_signature  # Unused
    values = building_blocks.Tuple(comps)
    return building_block_factory.create_federated_zip(values)

  def _transform_args_with_functional_types(self, comps, type_signature):
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
      A `building_blocks.Block`.
    """
    functions = building_blocks.Tuple(comps)
    fn_name = next(self._name_generator)
    fn_ref = building_blocks.Reference(fn_name, functions.type_signature)
    if isinstance(type_signature.parameter, computation_types.NamedTupleType):
      arg_type = [[] for _ in range(len(type_signature.parameter))]
      for functional_comp in comps:
        named_type_signatures = anonymous_tuple.to_elements(
            functional_comp.type_signature.parameter)
        for index, (_, concrete_type) in enumerate(named_type_signatures):
          arg_type[index].append(concrete_type)
    else:
      arg_type = [e.type_signature.parameter for e in comps]
    arg_name = next(self._name_generator)
    arg_ref = building_blocks.Reference(arg_name, arg_type)
    if isinstance(type_signature.parameter, computation_types.NamedTupleType):
      arg = building_block_factory.create_zip(arg_ref)
    else:
      arg = arg_ref
    elements = []
    for index, functional_comp in enumerate(comps):
      sel_fn = building_blocks.Selection(fn_ref, index=index)
      sel_arg = building_blocks.Selection(arg, index=index)
      call = building_blocks.Call(sel_fn, sel_arg)
      elements.append(call)
    calls = building_blocks.Tuple(elements)
    result = building_blocks.Lambda(arg_ref.name, arg_ref.type_signature, calls)
    return building_blocks.Block(((fn_ref.name, functions),), result)

  def _transform_args(self, comp, type_signature):
    """Transforms the arguments from `comp`.

    Given a computation containing a tuple of intrinsics that can be merged,
    this function creates arguments that should be passed to the call of the
    transformed computation.

    Args:
      comp: The computation building block in which to perform the transform.
      type_signature: The type to use when determining how to transform the
        computations.

    Returns:
      A `building_blocks.ComputationBuildingBlock` representing the
      transformed arguments from `comp`.
    """
    if isinstance(type_signature, computation_types.NamedTupleType):
      comps = [[] for _ in range(len(type_signature))]
      for _, call in anonymous_tuple.iter_elements(comp):
        for index, arg in enumerate(call.argument):
          comps[index].append(arg)
      transformed_args = []
      for args, arg_type in zip(comps, type_signature):
        transformed_arg = self._transform_args_with_type(args, arg_type)
        transformed_args.append(transformed_arg)
      return building_blocks.Tuple(transformed_args)
    else:
      args = []
      for _, call in anonymous_tuple.iter_elements(comp):
        args.append(call.argument)
      return self._transform_args_with_type(args, type_signature)

  def transform(self, comp):
    """Returns a new transformed computation or `comp`."""
    if not self.should_transform(comp):
      return comp, False
    intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(self._uri)
    arg = self._transform_args(comp, intrinsic_def.type_signature.parameter)
    named_comps = anonymous_tuple.to_elements(comp)
    parameter_type = arg.type_signature
    type_signature = [call.type_signature.member for _, call in named_comps]
    result_type = computation_types.FederatedType(
        type_signature, intrinsic_def.type_signature.result.placement,
        intrinsic_def.type_signature.result.all_equal)
    intrinsic_type = computation_types.FunctionType(parameter_type, result_type)
    intrinsic = building_blocks.Intrinsic(self._uri, intrinsic_type)
    call = building_blocks.Call(intrinsic, arg)
    tup = building_block_factory.create_federated_unzip(call)
    names = [name for name, _ in named_comps]
    transformed_comp = building_block_factory.create_named_tuple(tup, names)
    return transformed_comp, True


def merge_tuple_intrinsics(comp, uri):
  r"""Merges tuples of called intrinsics into one called intrinsic."""
  return _apply_transforms(comp, MergeTupleIntrinsics(comp, uri))


def remove_duplicate_block_locals(comp):
  r"""Removes duplicated computations from Block locals in `comp`.

  This transform traverses `comp` postorder and removes duplicated computation
  building blocks from Block locals in `comp`. Additionally, Blocks variables
  whose value is a Reference and References pointing to References are removed.

  Args:
    comp: The computation building block in which to perform the removals.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  tree_analysis.check_has_unique_names(comp)

  def _should_transform(comp):
    """Returns `True` if `comp` should be transformed."""
    return (isinstance(comp, building_blocks.Block) or
            isinstance(comp, building_blocks.Reference))

  def _transform(comp, symbol_tree):
    """Returns a new transformed computation or `comp`."""
    if not _should_transform(comp):
      return comp, False
    if isinstance(comp, building_blocks.Block):
      variables = []
      for name, value in comp.locals:
        symbol_tree.walk_down_one_variable_binding()
        payload = symbol_tree.get_payload_with_name(name)
        if (not payload.removed and
            not isinstance(value, building_blocks.Reference)):
          variables.append((name, value))
      if not variables:
        comp = comp.result
      else:
        comp = building_blocks.Block(variables, comp.result)
      return comp, True
    elif isinstance(comp, building_blocks.Reference):
      value = symbol_tree.get_payload_with_name(comp.name).value
      if value is None:
        return comp, False
      while isinstance(value, building_blocks.Reference):
        new_value = symbol_tree.get_payload_with_name(value.name).value
        if new_value is None:
          comp = building_blocks.Reference(value.name, value.type_signature)
          return comp, True
        else:
          value = new_value
      payloads_with_value = symbol_tree.get_all_payloads_with_value(
          value, tree_analysis.trees_equal)
      if payloads_with_value:
        highest_payload = payloads_with_value[-1]
        lower_payloads = payloads_with_value[:-1]
        for payload in lower_payloads:
          symbol_tree.update_payload_with_name(payload.name)
        comp = building_blocks.Reference(highest_payload.name,
                                         highest_payload.value.type_signature)
        return comp, True
    return comp, False

  symbol_tree = transformation_utils.SymbolTree(
      transformation_utils.TrackRemovedReferences)
  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, _transform, symbol_tree)


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
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    """Returns `True` if `comp` is a mapped or applied identity function."""
    if (isinstance(comp, building_blocks.Call) and
        isinstance(comp.function, building_blocks.Intrinsic) and
        comp.function.uri in (
            intrinsic_defs.FEDERATED_MAP.uri,
            intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
            intrinsic_defs.FEDERATED_APPLY.uri,
            intrinsic_defs.SEQUENCE_MAP.uri,
        )):
      called_function = comp.argument[0]
      return building_block_analysis.is_identity_function(called_function)
    return False

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    transformed_comp = comp.argument[1]
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


class RemoveUnusedBlockLocals(transformation_utils.TransformSpec):
  """Removes block local variables which are not used in the result."""

  def should_transform(self, comp):
    return isinstance(comp, building_blocks.Block)

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    unbound_ref_set = transformation_utils.get_map_of_unbound_references(
        comp.result)[comp.result]
    if (not unbound_ref_set) or (not comp.locals):
      return comp.result, True
    new_locals = []
    for name, val in reversed(comp.locals):
      if name in unbound_ref_set:
        new_locals.append((name, val))
        unbound_ref_set = unbound_ref_set.union(
            transformation_utils.get_map_of_unbound_references(val)[val])
        unbound_ref_set.discard(name)
    if len(new_locals) == len(comp.locals):
      return comp, False
    elif not new_locals:
      return comp.result, True
    return building_blocks.Block(reversed(new_locals), comp.result), True


def remove_unused_block_locals(comp):
  return _apply_transforms(comp, RemoveUnusedBlockLocals())


class ReplaceCalledLambdaWithBlock(transformation_utils.TransformSpec):
  r"""Replaces all the called lambdas in `comp` with a block.

  This transform replaces the following computation containing a called lambda:

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
  """

  def __init__(self):
    super().__init__(global_transform=True)

  def should_transform(self, comp, referred):
    return (isinstance(comp, building_blocks.Call) and
            (isinstance(comp.function, building_blocks.Lambda) or
             isinstance(referred, building_blocks.Lambda))) or isinstance(
                 comp, building_blocks.Block)

  def _resolve_reference_to_concrete_value(self, ref, symbol_tree):
    while isinstance(ref, building_blocks.Reference):
      referred_payload = symbol_tree.get_payload_with_name(ref.name)
      ref = referred_payload.value
    return ref

  def transform(self, comp, symbol_tree):
    referred = None
    if isinstance(comp, building_blocks.Call) and isinstance(
        comp.function, building_blocks.Reference):
      node = symbol_tree.get_payload_with_name(comp.function.name)
      if node is not None:
        referred = node.value
        if isinstance(referred, building_blocks.Reference):
          referred = self._resolve_reference_to_concrete_value(
              referred, symbol_tree)
    if not self.should_transform(comp, referred):
      return comp, False
    if isinstance(comp, building_blocks.Block):
      new_locals = []
      for name, value in comp.locals:
        symbol_tree.walk_down_one_variable_binding()
        if not symbol_tree.get_payload_with_name(name).removed:
          new_locals.append((name, value))
      if not new_locals:
        return comp.result, True
      elif len(new_locals) == len(comp.locals):
        return comp, False
      return building_blocks.Block(new_locals, comp.result), True
    elif isinstance(referred, building_blocks.Lambda):
      referred = typing.cast(building_blocks.Lambda, referred)
      transformed_comp = building_blocks.Block(
          [(referred.parameter_name, comp.argument)], referred.result)
      symbol_tree.update_payload_with_name(comp.function.name)
    else:
      if comp.function.parameter_type is not None:
        transformed_comp = building_blocks.Block(
            [(comp.function.parameter_name, comp.argument)],
            comp.function.result)
      else:
        transformed_comp = comp.function.result
    return transformed_comp, True


def replace_called_lambda_with_block(comp):
  """Replaces all the called lambdas in `comp` with a block."""
  lambda_replacer = ReplaceCalledLambdaWithBlock()

  def _transform_fn(comp, symbol_tree):
    return lambda_replacer.transform(comp, symbol_tree)

  symbol_tree = transformation_utils.SymbolTree(
      transformation_utils.TrackRemovedReferences)
  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, _transform_fn, symbol_tree)


class ReplaceSelectionFromTuple(transformation_utils.TransformSpec):
  r"""Replaces any selection from a tuple with the underlying tuple element.

  Invocations of `transform` replace any occurences of:

  Selection
           \
            Tuple
            |
            [Comp, Comp, ...]

  with the appropriate Comp, as determined by the `index` or `name` of the
  `Selection`.
  """

  def should_transform(self, comp):
    return (isinstance(comp, building_blocks.Selection) and
            isinstance(comp.source, building_blocks.Tuple))

  def _get_index_from_name(self, selection_name, tuple_type_signature):
    named_type_signatures = anonymous_tuple.to_elements(tuple_type_signature)
    return [x[0] for x in named_type_signatures].index(selection_name)

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    if comp.name is not None:
      index = self._get_index_from_name(comp.name, comp.source.type_signature)
    else:
      index = comp.index
    return comp.source[index], True


def replace_selection_from_tuple_with_element(comp):
  """Replaces any selection from a tuple with the underlying tuple element."""
  return _apply_transforms(comp, ReplaceSelectionFromTuple())


def uniquify_compiled_computation_names(comp):
  """Replaces all the compiled computations names in `comp` with unique names.

  This transform traverses `comp` postorder and replaces the name of all the
  compiled computations found in `comp` with a unique name.

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  name_generator = building_block_factory.unique_name_generator(None, prefix='')

  def _should_transform(comp):
    return isinstance(comp, building_blocks.CompiledComputation)

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    transformed_comp = building_blocks.CompiledComputation(
        comp.proto, next(name_generator))
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
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  name_generator = building_block_factory.unique_name_generator(None)

  class _RenameNode(transformation_utils.BoundVariableTracker):
    """transformation_utils.SymbolTree node for renaming References in ASTs."""

    def __init__(self, name, value):
      super().__init__(name, value)
      py_typecheck.check_type(name, str)
      self.new_name = next(name_generator)

    def __str__(self):
      return 'Value: {}, name: {}, new_name: {}'.format(self.value, self.name,
                                                        self.new_name)

  def _transform(comp, context_tree):
    """Renames References in `comp` to unique names."""
    if isinstance(comp, building_blocks.Reference):
      payload = context_tree.get_payload_with_name(comp.name)
      if payload is None:
        return comp, False
      new_name = payload.new_name
      return building_blocks.Reference(new_name, comp.type_signature,
                                       comp.context), True
    elif isinstance(comp, building_blocks.Block):
      new_locals = []
      for name, val in comp.locals:
        context_tree.walk_down_one_variable_binding()
        new_name = context_tree.get_payload_with_name(name).new_name
        new_locals.append((new_name, val))
      return building_blocks.Block(new_locals, comp.result), True
    elif isinstance(comp, building_blocks.Lambda):
      if comp.parameter_type is None:
        return comp, False
      context_tree.walk_down_one_variable_binding()
      new_name = context_tree.get_payload_with_name(
          comp.parameter_name).new_name
      return building_blocks.Lambda(new_name, comp.parameter_type,
                                    comp.result), True
    return comp, False

  symbol_tree = transformation_utils.SymbolTree(_RenameNode)
  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, _transform, symbol_tree)


def group_block_locals_by_namespace(block):
  """Partitions `block.locals` into classes which share namespaces.

  That is, the computations in each class close over the same set of variables.
  The sets of variables each class closes over is defined by the sequence of
  variable bindings in the local_variables statement. That is, if the local
  variables declare 'a' then 'b', and the variable 'x' is the top level unbound
  ref, then this function will construct computation classes of the form:

  [[all computations having only x as unbound],
  [remaining computations having either a or x unbound],
  [remaining computations having a, b, or x unbound]]

  Args:
    block: Instance of `building_blocks.Block`, whose local variables we wish to
      partition in the manner described above.

  Returns:
    A list of lists, where each computation in the locals of `block` appears in
    exactly one list, each inner list contains two-tuples (whose second elements
    are the computations which share the same unbound variables and first
    element is the variable name bound to this computation), and each
    computation appears as early as possible. The order of the lists returned
    is defined by the order of the variable indings in `block`, as described
    above. In particular, the length of the outer list will always be the number
    of variables declared in the block locals statement, plus one, and the sum
    of the lengths of the inner lists will be identical to the number of local
    variables declared.
  """
  py_typecheck.check_type(block, building_blocks.Block)
  all_unbound_refs = transformation_utils.get_map_of_unbound_references(block)
  top_level_unbound_ref = all_unbound_refs[block]
  local_variables = block.locals

  arg_classes = [top_level_unbound_ref]

  for var, _ in local_variables:
    final_arg_class = arg_classes[-1].copy()
    final_arg_class.add(var)
    arg_classes.append(final_arg_class)

  comps_yet_to_partition = [
      (name, comp, all_unbound_refs[comp]) for (name, comp) in local_variables
  ]
  comp_classes = []
  for args in arg_classes:
    cls = []
    selected_indices = []
    for idx, (name, comp, refs) in enumerate(comps_yet_to_partition):
      if refs.issubset(args):
        cls.append((name, comp))
        selected_indices.append(idx)
    remaining_comps = []
    for idx, comp_tuple in enumerate(comps_yet_to_partition):
      if idx not in selected_indices:
        remaining_comps.append(comp_tuple)
    comps_yet_to_partition = remaining_comps
    comp_classes.append(cls)

  return comp_classes


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
  compiled into TensorFlow; that is, any `building_blocks.Reference` whose type
  is compatible with stamping into a TensorFlow graph.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` whose AST we
      will traverse, replacing appropriate instances of
      `building_blocks.Reference` with graphs representing the identity function
      of the appropriate type called on the same reference. `comp` must declare
      a parameter and result type which are both able to be stamped in to a
      TensorFlow graph.

  Returns:
    A possibly modified  version of `comp`, where any references now have a
    parent of type `building_blocks.Call` with function an instance
    of `building_blocks.CompiledComputation`.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)

  if isinstance(comp, building_blocks.CompiledComputation):
    return comp, False

  def _should_decorate(comp):
    return (isinstance(comp, building_blocks.Reference) and
            type_analysis.is_tensorflow_compatible_type(comp.type_signature))

  def _decorate(comp):
    identity_function = building_block_factory.create_compiled_identity(
        comp.type_signature)
    return building_blocks.Call(identity_function, comp)

  def _decorate_if_reference_without_graph(comp):
    """Decorates references under `comp` if necessary."""
    if (isinstance(comp, building_blocks.Tuple) and
        any(_should_decorate(x) for x in comp)):
      elems = []
      for x in anonymous_tuple.iter_elements(comp):
        if _should_decorate(x[1]):
          elems.append((x[0], _decorate(x[1])))
        else:
          elems.append((x[0], x[1]))
      return building_blocks.Tuple(elems), True
    elif (isinstance(comp, building_blocks.Call) and
          not isinstance(comp.function, building_blocks.CompiledComputation) and
          _should_decorate(comp.argument)):
      arg = _decorate(comp.argument)
      return building_blocks.Call(comp.function, arg), True
    elif (isinstance(comp, building_blocks.Selection) and
          _should_decorate(comp.source)):
      return building_blocks.Selection(
          _decorate(comp.source), name=comp.name, index=comp.index), True
    elif (isinstance(comp, building_blocks.Lambda) and
          _should_decorate(comp.result)):
      return building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                    _decorate(comp.result)), True
    elif isinstance(comp, building_blocks.Block) and (
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
      return building_blocks.Block(new_locals, new_result), True
    return comp, False

  return transformation_utils.transform_postorder(
      comp, _decorate_if_reference_without_graph)


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
  4. There are no instances of `building_blocks.Data` of federated
     type under `comp`; how these would be handled by a function such as this
     is not entirely clear.

  Under these conditions, `unwrap_placement` will produce a single call to
  federated_map, federated_apply or federated_value, depending on the placement
  and type signature of `comp`. Other than this single map or apply, no
  intrinsics will remain under `comp`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` satisfying the
      assumptions above.

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
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp.type_signature, computation_types.FederatedType)
  single_placement = comp.type_signature.placement

  tree_analysis.check_has_single_placement(comp, single_placement)

  name_generator = building_block_factory.unique_name_generator(comp)

  all_unbound_references = transformation_utils.get_map_of_unbound_references(
      comp)
  root_unbound_references = all_unbound_references[comp]

  if len(root_unbound_references) > 1:
    raise ValueError(
        '`unwrap_placement` can only handle computations with at most a single '
        'unbound reference; you have passed in the computation {} with {} '
        'unbound references.'.format(comp.compact_representation(),
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
      comp: Instance of `building_blocks.ComputationBuildingBlock` with at most
        a single unbound reference.
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
        super().__init__(name, value)
        self.unbound = False

      def __str__(self):
        return ''

      def update(self, x):
        del x  # Unused
        self.unbound = True

    symbol_tree = transformation_utils.SymbolTree(_UnboundVariableIdentifier)
    symbol_tree.ingest_variable_binding(unbound_variable_name, None)
    symbol_tree.update_payload_with_name(unbound_variable_name)

    def _should_transform(comp, symbol_tree):
      return (isinstance(comp, building_blocks.Reference) and
              comp.name == unbound_variable_name and
              symbol_tree.get_payload_with_name(comp.name).unbound)

    def _rename_unbound_variable(comp, symbol_tree):
      """Updates the nonlocal tracker, and renames the unbound variable."""
      if not _should_transform(comp, symbol_tree):
        return comp, False
      if unbound_reference_name_and_type_pair[0][1] is None:
        name = next(name_generator)
        unbound_reference_name_and_type_pair[0] = (name, comp.type_signature)
      else:
        name = unbound_reference_name_and_type_pair[0][0]
      return building_blocks.Reference(name, comp.type_signature), True

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
      comp: Instance of `building_blocks.ComputationBuildingBlock` from which we
        wish to remove placement.

    Returns:
      A transformed version of comp with its placements removed.

    Raises:
      NotImplementedError: In case a node of type
        `building_blocks.Data` is encountered in the AST, as
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
      new_type, _ = type_transformations.transform_type_postorder(
          comp.type_signature, _remove_placement_from_type)
      return building_blocks.Reference(comp.name, new_type)

    def _identity_function(arg_type):
      """Creates `lambda x: x` with argument type `arg_type`."""
      arg_name = next(name_generator)
      val = building_blocks.Reference(arg_name, arg_type)
      lam = building_blocks.Lambda(arg_name, arg_type, val)
      return lam

    def _call_first_with_second_function(fn_type, arg_type):
      """Creates `lambda x: x[0](x[1])` with the provided ."""
      arg_name = next(name_generator)
      tuple_ref = building_blocks.Reference(arg_name, [fn_type, arg_type])
      fn = building_blocks.Selection(tuple_ref, index=0)
      arg = building_blocks.Selection(tuple_ref, index=1)
      called_fn = building_blocks.Call(fn, arg)
      return building_blocks.Lambda(arg_name, tuple_ref.type_signature,
                                    called_fn)

    def _call_function(arg_type):
      """Creates `lambda x: x()` argument type `arg_type`."""
      arg_name = next(name_generator)
      arg_ref = building_blocks.Reference(arg_name, arg_type)
      called_arg = building_blocks.Call(arg_ref, None)
      return building_blocks.Lambda(arg_name, arg_type, called_arg)

    def _replace_intrinsics_with_functions(comp):
      """Helper to remove intrinsics from the AST."""
      tys = comp.type_signature

      # These functions have no runtime behavior and only exist to adjust
      # placement. They are replaced here with  `lambda x: x`.
      identities = [
          intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri,
          intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri,
          intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri,
          intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri,
      ]
      if comp.uri in identities:
        return _identity_function(tys.result.member)

      # These functions all `map` a value and are replaced with
      # `lambda args: args[0](args[1])
      maps = [
          intrinsic_defs.FEDERATED_MAP.uri,
          intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
          intrinsic_defs.FEDERATED_APPLY.uri,
      ]
      if comp.uri in maps:
        return _call_first_with_second_function(tys.parameter[0],
                                                tys.parameter[1].member)

      # `federated_eval`'s argument must simply be `call`ed and is replaced
      # with `lambda x: x()`
      evals = [
          intrinsic_defs.FEDERATED_EVAL_AT_SERVER.uri,
          intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS.uri,
      ]
      if comp.uri in evals:
        return _call_function(tys.parameter)

      raise ValueError('Disallowed intrinsic: {}'.format(comp))

    def _remove_lambda_placement(comp):
      """Removes placement from Lambda's parameter."""
      new_parameter_type, _ = type_transformations.transform_type_postorder(
          comp.parameter_type, _remove_placement_from_type)
      return building_blocks.Lambda(comp.parameter_name, new_parameter_type,
                                    comp.result)

    def _simplify_calls(comp):
      """Unwraps structures introduced by removing intrinsics."""
      zip_or_value_removed = (
          isinstance(comp.function.result, building_blocks.Reference) and
          comp.function.result.name == comp.function.parameter_name)
      if zip_or_value_removed:
        return comp.argument
      else:
        map_removed = (
            isinstance(comp.function.result, building_blocks.Call) and
            isinstance(comp.function.result.function, building_blocks.Selection)
            and comp.function.result.function.index == 0 and isinstance(
                comp.function.result.argument, building_blocks.Selection) and
            comp.function.result.argument.index == 1 and
            isinstance(comp.function.result.function.source,
                       building_blocks.Reference) and
            comp.function.result.function.source.name ==
            comp.function.parameter_name and
            isinstance(comp.function.result.function.source,
                       building_blocks.Reference) and
            comp.function.result.function.source.name ==
            comp.function.parameter_name and
            isinstance(comp.argument, building_blocks.Tuple))
        if map_removed:
          return building_blocks.Call(comp.argument[0], comp.argument[1])
      return comp

    def _transform(comp):
      """Dispatches to helpers above."""
      if isinstance(comp, building_blocks.Reference):
        return _remove_reference_placement(comp), True
      elif isinstance(comp, building_blocks.Intrinsic):
        return _replace_intrinsics_with_functions(comp), True
      elif isinstance(comp, building_blocks.Lambda):
        return _remove_lambda_placement(comp), True
      elif isinstance(comp, building_blocks.Call) and isinstance(
          comp.function, building_blocks.Lambda):
        return _simplify_calls(comp), True
      elif (isinstance(comp, building_blocks.Data) and
            isinstance(comp.type_signature, computation_types.FederatedType)):
        # TODO(b/135126947): Design and implement Data constructs.
        raise NotImplementedError
      return comp, False

    return transformation_utils.transform_postorder(comp, _transform)

  if unbound_reference_name is None:
    placement_removed, _ = _remove_placement(comp)
    return building_block_factory.create_federated_value(
        placement_removed, single_placement), True
  else:
    (unbound_variable_renamed,
     unbound_reference_info) = _rename_unbound_variable(comp,
                                                        unbound_reference_name)
    (new_reference_name, unbound_reference_type) = unbound_reference_info
    if not isinstance(unbound_reference_type, computation_types.FederatedType):
      raise TypeError(
          'The lone unbound reference is not of federated type; this is '
          'disallowed. The unbound type is {}'.format(unbound_reference_type))

    placement_removed, _ = _remove_placement(unbound_variable_renamed)
    ref_to_fed_arg = building_blocks.Reference(unbound_reference_name,
                                               unbound_reference_type)
    lambda_wrapping_placement_removal = building_blocks.Lambda(
        new_reference_name, unbound_reference_type.member, placement_removed)
    called_intrinsic = building_block_factory.create_federated_map_or_apply(
        lambda_wrapping_placement_removal, ref_to_fed_arg)
    return called_intrinsic, True
