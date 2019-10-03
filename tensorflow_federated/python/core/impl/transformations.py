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
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis


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


def remove_lambdas_and_blocks(comp):
  """Removes any called lambdas and blocks from `comp`.

  This function will rename all the variables in `comp` in a single walk of the
  AST, then replace called lambdas with blocks in another walk, since this
  transformation interacts with scope in delicate ways. It will chain inlining
  the blocks and collapsing the selection-from-tuple pattern together into a
  final pass.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` from which we
      want to remove called lambdas and blocks.

  Returns:
    A transformed version of `comp` which has no called lambdas or blocks, and
    no extraneous selections from tuples.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  comp, _ = uniquify_reference_names(comp)
  comp, _ = replace_called_lambda_with_block(comp)
  block_inliner = InlineBlock(comp)
  selection_replacer = ReplaceSelectionFromTuple()
  transforms = [block_inliner, selection_replacer]
  symbol_tree = transformation_utils.SymbolTree(
      transformation_utils.ReferenceCounter)

  def _transform_fn(comp, symbol_tree):
    """Transform function chaining inlining and collapsing selections.

    This function is inlined here as opposed to factored out and parameterized
    by the transforms to apply, due to the delicacy of chaining transformations
    which rely on state. These transformations should be safe if they appear
    first in the list of transforms, but due to the difficulty of reasoning
    about the invariants the transforms can rely on in this setting, there is
    no function exposed which hoists out the internal logic.

    Args:
      comp: Instance of `building_blocks.ComputationBuildingBlock` we wish to
        check for inlining and collapsing of selections.
      symbol_tree: Instance of `building_blocks.SymbolTree` defining the
        bindings available to `comp`.

    Returns:
      A transformed version of `comp`.
    """
    modified = False
    for transform in transforms:
      if transform.global_transform:
        comp, transform_modified = transform.transform(comp, symbol_tree)
      else:
        comp, transform_modified = transform.transform(comp)
      modified = modified or transform_modified
    return comp, modified

  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, _transform_fn, symbol_tree)


class ExtractComputation(transformation_utils.TransformSpec):
  """Extracts a computation if a variable it depends on is not bound.

  This transforms a computation which matches the `predicate` or is a Block, and
  replaces the computations with a LET
  construct if a variable it depends on is not bound by the current scope. Both
  the `parameter_name` of a `building_blocks.Lambda` and the name of
  any variable defined by a `building_blocks.Block` can affect the
  scope in which a reference in computation is bound.

  NOTE: This function extracts `computation_building_block.Block` because block
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
    super(ExtractComputation, self).__init__()
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    tree_analysis.check_has_unique_names(comp)
    self._name_generator = building_block_factory.unique_name_generator(comp)
    self._predicate = predicate
    self._unbound_references = get_map_of_unbound_references(comp)

  def _contains_unbound_reference(self, comp, names):
    """Returns `True` if `comp` contains unbound references to `names`.

    This function will update the non-local `_unbound_references` captured from
    the parent context if `comp` is not contained in that collection. This can
    happen when new computations are created and added to the AST.

    Args:
      comp: The computation building block to test.
      names: A Python string or a list, tuple, or set of Python strings.
    """
    if isinstance(names, six.string_types):
      names = (names,)
    if comp not in self._unbound_references:
      references = get_map_of_unbound_references(comp)
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
      name = six.next(self._name_generator)
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
      name = six.next(self._name_generator)
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
        name = six.next(self._name_generator)
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
      name = six.next(self._name_generator)
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
      name = six.next(self._name_generator)
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
        variable_name = six.next(self._name_generator)
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
  """Extracts computations to the scope which binds a variable it depends on.

  NOTE: If a computation does not contain a variable that is bound by a
  computation in `comp` it will be extracted to the root.

  Args:
    comp: The computation building block in which to perform the transformation.

  Returns:
    A new computation with the transformation applied or the original `comp`.
  """

  def _predicate(comp):
    return not isinstance(comp, building_blocks.Reference)

  return _apply_transforms(comp, ExtractComputation(comp, _predicate))


def extract_intrinsics(comp):
  """Extracts intrinsics to the scope which binds a variable it depends on.

  NOTE: If an intrinsic does not contain a variable that is bound by a
  computation in `comp` it will be extracted to the root.

  Args:
    comp: The computation building block in which to perform the transformation.

  Returns:
    A new computation with the transformation applied or the original `comp`.
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
    super(InlineBlock, self).__init__(global_transform=True)
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    tree_analysis.check_has_unique_names(comp)
    if variable_names is not None:
      py_typecheck.check_type(variable_names, (list, tuple, set))
      for name in variable_names:
        py_typecheck.check_type(name, six.string_types)
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
      try:
        value = symbol_tree.get_payload_with_name(comp.name).value
      except NameError:
        # This reference is unbound
        value = None
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
  is completely safe to add the locals lists together in this implementation,
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
    super(MergeChainedFederatedMapsOrApplys, self).__init__()
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
    functions_name = six.next(self._name_generator)
    functions_ref = building_blocks.Reference(functions_name,
                                              functions.type_signature)
    arg_name = six.next(self._name_generator)
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


def merge_chained_federated_maps_or_applys(comp):
  """Merges chained federated maps or federated apply into one structure."""
  return _apply_transforms(comp, MergeChainedFederatedMapsOrApplys(comp))


class MergeTupleIntrinsics(transformation_utils.TransformSpec):
  r"""Merges a tuple of called intrinsics into one called intrinsic.

  This transform matches the following pattern, and replaces the following
  computation containing a tuple of called intrinsics all represeting the same
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

  NOTE: This is just an example of what this transformation would look like when
  applied to a tuple of federated maps. The components `f1`, `f2`, `v1`, and
  `v2` and the number of those components are not important.

  This transformation is implemented to match the following intrinsics:

  * intrinsic_defs.FEDERATED_AGGREGATE.uri
  * intrinsic_defs.FEDERATED_APPLY.uri
  * intrinsic_defs.FEDERATED_BROADCAST.uri
  * intrinsic_defs.FEDERATED_MAP.uri
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
    super(MergeTupleIntrinsics, self).__init__()
    py_typecheck.check_type(uri, six.string_types)
    self._name_generator = building_block_factory.unique_name_generator(comp)
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
    fn_name = six.next(self._name_generator)
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
    arg_name = six.next(self._name_generator)
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
    parameter_type = computation_types.to_type(arg.type_signature)
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


def remove_duplicate_computations(comp):
  r"""Removes duplicated computations from `comp`.

  This transform traverses `comp` postorder and remove duplicated computation
  building blocks from `comp`. Additionally, Blocks variables whose value is a
  Reference and References pointing to References are removed.

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
          value, _computations_equal)
      if payloads_with_value:
        highest_payload = payloads_with_value[-1]
        lower_payloads = payloads_with_value[:-1]
        for payload in lower_payloads:
          symbol_tree.update_payload_with_name(payload.name)
        comp = building_blocks.Reference(highest_payload.name,
                                         highest_payload.value.type_signature)
        return comp, True
    return comp, False

  class TrackRemovedReferences(transformation_utils.BoundVariableTracker):
    """transformation_utils.SymbolTree node for removing References in ASTs."""

    def __init__(self, name, value):
      super(TrackRemovedReferences, self).__init__(name, value)
      self._removed = False

    @property
    def removed(self):
      return self._removed

    def update(self, value):
      self._removed = True

    def __str__(self):
      return 'Name: {}; value: {}; removed: {}'.format(self.name, self.value,
                                                       self.removed)

  symbol_tree = transformation_utils.SymbolTree(TrackRemovedReferences)
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

  def should_transform(self, comp):
    return (isinstance(comp, building_blocks.Call) and
            isinstance(comp.function, building_blocks.Lambda))

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    transformed_comp = building_blocks.Block(
        [(comp.function.parameter_name, comp.argument)], comp.function.result)
    return transformed_comp, True


def replace_called_lambda_with_block(comp):
  """Replaces all the called lambdas in `comp` with a block."""
  return _apply_transforms(comp, ReplaceCalledLambdaWithBlock())


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
  comiled computations found in `comp` with a unique name.

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
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  name_generator = building_block_factory.unique_name_generator(None)

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
    if isinstance(comp, building_blocks.Reference):
      try:
        new_name = context_tree.get_payload_with_name(comp.name).new_name
        return building_blocks.Reference(new_name, comp.type_signature,
                                         comp.context), True
      except NameError:
        return comp, False
    elif isinstance(comp, building_blocks.Block):
      new_locals = []
      for name, val in comp.locals:
        context_tree.walk_down_one_variable_binding()
        new_name = context_tree.get_payload_with_name(name).new_name
        new_locals.append((new_name, val))
      return building_blocks.Block(new_locals, comp.result), True
    elif isinstance(comp, building_blocks.Lambda):
      context_tree.walk_down_one_variable_binding()
      new_name = context_tree.get_payload_with_name(
          comp.parameter_name).new_name
      return building_blocks.Lambda(new_name, comp.parameter_type,
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
  `building_blocks.CompiledComputation` representing the same
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
     reference has as its parent a `building_blocks.Call`, whose
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
        compiled_computation_transforms.CalledGraphOnReplicatedArg(),
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
      comp: The `building_blocks.ComputationBuildingBlock` to check for
        possibility of reduction according to the parsing library.

    Returns:
      A tuple whose first element is a possibly transformed version of `comp`,
      and whose second is a Boolean indicating whether or not `comp` was
      transformed. This is in conforming to the conventions of
      `transformation_utils.transform_postorder`.
    """
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    for option in self._parse_library:
      if option.should_transform(comp):
        return option.transform(comp)
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
  `building_blocks.Lambda` whose parameter and result type
  can both be bound into TensorFlow. This pattern is enforced here as
  parameter validation on `comp`.

  Args:
    comp: Instance of `building_blocks.Lambda` whose AST we will traverse,
      replacing appropriate instances of `building_blocks.Reference` with graphs
      representing the identity function of the appropriate type called on the
      same reference. `comp` must declare a parameter and result type which are
      both able to be stamped in to a TensorFlow graph.

  Returns:
    A possibly modified  version of `comp`, where any references now have a
    parent of type `building_blocks.Call` with function an instance
    of `building_blocks.CompiledComputation`.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)

  if isinstance(comp, building_blocks.CompiledComputation):
    return comp, False

  if not (isinstance(comp, building_blocks.Lambda) and
          type_utils.is_tensorflow_compatible_type(comp.result.type_signature)
          and type_utils.is_tensorflow_compatible_type(comp.parameter_type)):
    raise ValueError(
        '`insert_called_tf_identity_at_leaves` should only be '
        'called on instances of '
        '`building_blocks.Lambda` whose parameter '
        'and result types can both be stamped into TensorFlow '
        'graphs. You have called in on a {} of type signature {}.'.format(
            comp.compact_representation(), comp.type_signature))

  def _should_decorate(comp):
    return (isinstance(comp, building_blocks.Reference) and
            type_utils.is_tensorflow_compatible_type(comp.type_signature))

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

  all_unbound_references = get_map_of_unbound_references(comp)
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
        super(_UnboundVariableIdentifier, self).__init__(name, value)
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
        name = six.next(name_generator)
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
      new_type, _ = type_utils.transform_type_postorder(
          comp.type_signature, _remove_placement_from_type)
      return building_blocks.Reference(comp.name, new_type)

    def _replace_intrinsics_with_functions(comp):
      """Helper to remove intrinsics from the AST."""
      if (comp.uri == intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri or
          comp.uri == intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri or
          comp.uri == intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri or
          comp.uri == intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri):
        arg_name = six.next(name_generator)
        arg_type = comp.type_signature.result.member
        val = building_blocks.Reference(arg_name, arg_type)
        lam = building_blocks.Lambda(arg_name, arg_type, val)
        return lam
      elif comp.uri not in (intrinsic_defs.FEDERATED_MAP.uri,
                            intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
                            intrinsic_defs.FEDERATED_APPLY.uri):
        raise ValueError('Disallowed intrinsic: {}'.format(comp))
      arg_name = six.next(name_generator)
      tuple_ref = building_blocks.Reference(arg_name, [
          comp.type_signature.parameter[0],
          comp.type_signature.parameter[1].member,
      ])
      fn = building_blocks.Selection(tuple_ref, index=0)
      arg = building_blocks.Selection(tuple_ref, index=1)
      called_fn = building_blocks.Call(fn, arg)
      return building_blocks.Lambda(arg_name, tuple_ref.type_signature,
                                    called_fn)

    def _remove_lambda_placement(comp):
      """Removes placement from Lambda's parameter."""
      new_parameter_type, _ = type_utils.transform_type_postorder(
          comp.parameter_type, _remove_placement_from_type)
      return building_blocks.Lambda(comp.parameter_name, new_parameter_type,
                                    comp.result)

    def _transform(comp):
      """Dispatches to helpers above."""
      if isinstance(comp, building_blocks.Reference):
        return _remove_reference_placement(comp), True
      elif isinstance(comp, building_blocks.Intrinsic):
        return _replace_intrinsics_with_functions(comp), True
      elif isinstance(comp, building_blocks.Lambda):
        return _remove_lambda_placement(comp), True
      elif (isinstance(comp, building_blocks.Data) and
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
    return building_block_factory.create_federated_value(
        placement_removed, single_placement), True

  ref_to_fed_arg = building_blocks.Reference(unbound_reference_name,
                                             unbound_reference_type)

  lambda_wrapping_placement_removal = building_blocks.Lambda(
      new_reference_name, unbound_reference_type.member, placement_removed)

  called_intrinsic = building_block_factory.create_federated_map_or_apply(
      lambda_wrapping_placement_removal, ref_to_fed_arg)

  return called_intrinsic, True


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
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  references = {}

  def _update(comp):
    """Updates the Python dict of references."""
    if isinstance(comp, building_blocks.Reference):
      references[comp] = set((comp.name,))
    elif isinstance(comp, building_blocks.Block):
      references[comp] = set()
      names = []
      for name, variable in comp.locals:
        elements = references[variable]
        references[comp].update([e for e in elements if e not in names])
        names.append(name)
      elements = references[comp.result]
      references[comp].update([e for e in elements if e not in names])
    elif isinstance(comp, building_blocks.Call):
      elements = references[comp.function]
      if comp.argument is not None:
        elements.update(references[comp.argument])
      references[comp] = elements
    elif isinstance(comp, building_blocks.Lambda):
      elements = references[comp.result]
      references[comp] = set([e for e in elements if e != comp.parameter_name])
    elif isinstance(comp, building_blocks.Selection):
      references[comp] = references[comp.source]
    elif isinstance(comp, building_blocks.Tuple):
      elements = [references[e] for e in comp]
      references[comp] = set(itertools.chain.from_iterable(elements))
    else:
      references[comp] = set()
    return comp, False

  transformation_utils.transform_postorder(comp, _update)
  return references


def _computations_equal(comp_1, comp_2):
  """Returns `True` if the computations are equal.

  If you pass objects other than instances of
  `building_blocks.ComputationBuildingBlock` this function will
  return `False`. Structurally equaivalent computations with different variable
  names are not considered to be equal.

  NOTE: This function could be quite expensive if you do not
  `extract_computations` first. Extracting all comptations reduces the equality
  of two computations in most cases to an identity check. One notable exception
  to this is `CompiledComputation` for which equality is delegated to the proto
  object.

  Args:
    comp_1: A `building_blocks.ComputationBuildingBlock` to test.
    comp_2: A `building_blocks.ComputationBuildingBlock` to test.

  Raises:
    TypeError: If `comp_1` or `comp_2` is not an instance of
      `building_blocks.ComputationBuildingBlock`.
    NotImplementedError: If `comp_1` and `comp_2` are an unexpected subclass of
      `building_blocks.ComputationBuildingBlock`.
  """
  py_typecheck.check_type(comp_1, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(comp_2, building_blocks.ComputationBuildingBlock)
  if comp_1 is comp_2:
    return True
  # The unidiomatic-typecheck is intentional, for the purposes of equality this
  # function requires that the types are identical and that a subclass will not
  # be equal to it's baseclass.
  if type(comp_1) != type(comp_2):  # pylint: disable=unidiomatic-typecheck
    return False
  if comp_1.type_signature != comp_2.type_signature:
    return False
  if isinstance(comp_1, building_blocks.Block):
    if not _computations_equal(comp_1.result, comp_2.result):
      return False
    if len(comp_1.locals) != len(comp_2.locals):
      return False
    for (name_1, value_1), (name_2, value_2) in zip(comp_1.locals,
                                                    comp_2.locals):
      if name_1 != name_2 or not _computations_equal(value_1, value_2):
        return False
    return True
  elif isinstance(comp_1, building_blocks.Call):
    return (_computations_equal(comp_1.function, comp_2.function) and
            (comp_1.argument is None and comp_2.argument is None or
             _computations_equal(comp_1.argument, comp_2.argument)))
  elif isinstance(comp_1, building_blocks.CompiledComputation):
    return comp_1.proto == comp_2.proto
  elif isinstance(comp_1, building_blocks.Data):
    return comp_1.uri == comp_2.uri
  elif isinstance(comp_1, building_blocks.Intrinsic):
    return comp_1.uri == comp_2.uri
  elif isinstance(comp_1, building_blocks.Lambda):
    return (comp_1.parameter_name == comp_2.parameter_name and
            comp_1.parameter_type == comp_2.parameter_type and
            _computations_equal(comp_1.result, comp_2.result))
  elif isinstance(comp_1, building_blocks.Placement):
    return comp_1.uri == comp_2.uri
  elif isinstance(comp_1, building_blocks.Reference):
    return comp_1.name == comp_2.name
  elif isinstance(comp_1, building_blocks.Selection):
    return (_computations_equal(comp_1.source, comp_2.source) and
            comp_1.name == comp_2.name and comp_1.index == comp_2.index)
  elif isinstance(comp_1, building_blocks.Tuple):
    # The element names are checked as part of the `type_signature`.
    if len(comp_1) != len(comp_2):
      return False
    for element_1, element_2 in zip(comp_1, comp_2):
      if not _computations_equal(element_1, element_2):
        return False
    return True
  raise NotImplementedError('Unexpected type found: {}.'.format(type(comp_1)))
