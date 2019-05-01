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
      return comp, False
    transformed_comp = computation_building_blocks.CompiledComputation(
        comp.proto, str(six.next(name_generator)))
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
      return comp, False
    transformed_comp = computation_building_blocks.Block(
        [(comp.function.parameter_name, comp.argument)], comp.function.result)
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
      return comp, False
    transformed_comp = comp.argument[1]
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


def replace_chained_federated_maps_with_federated_map(comp):
  r"""Replaces all the chained federated maps in `comp` with one federated map.

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

  federated_map(<x, federated_map(<y, z>)>)

  with the following computation containing one federated map intrinsic:


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

  federated_map(<(let fn=<y, x> in (arg -> fn[1](fn[0](arg)))), z>)

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
        comps: a Python list of computations.

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


def replace_tuple_intrinsics_with_intrinsic(comp):
  r"""Replaces all the tuples of intrinsics in `comp` with one intrinsic.

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

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Block,               Tuple, ...]
                 /     \               |
         fn=Tuple       Lambda(arg)    [Comp(v1), Comp(v2), ...]
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
      return comp, False

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

    def _create_block_to_calls(call_names, comps):
      r"""Constructs a transformed block computation from `comps`.

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
        call_names: a Python list of names.
        comps: a Python list of computations.

      Returns:
        A `computation_building_blocks.Block`.
      """
      functions = computation_building_blocks.Tuple(zip(call_names, comps))
      fn = computation_building_blocks.Reference('fn', functions.type_signature)
      arg_type = [element.type_signature.parameter for element in comps]
      arg = computation_building_blocks.Reference('arg', arg_type)
      elements = []
      for index, name in enumerate(call_names):
        sel_fn = computation_building_blocks.Selection(fn, index=index)
        sel_arg = computation_building_blocks.Selection(arg, index=index)
        call = computation_building_blocks.Call(sel_fn, sel_arg)
        elements.append((name, call))
      calls = computation_building_blocks.Tuple(elements)
      lam = computation_building_blocks.Lambda(arg.name, arg.type_signature,
                                               calls)
      return computation_building_blocks.Block([('fn', functions)], lam)

    def _create_transformed_args_from_comps(call_names, elements):
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
        call_names: a Python list of names.
        elements: A 2 dimentional Python list of computations.

      Returns:
        A Python list of computations.
      """
      args = []
      for comps in elements:
        if isinstance(comps[0].type_signature, computation_types.FunctionType):
          arg = _create_block_to_calls(call_names, comps)
        else:
          arg = computation_building_blocks.Tuple(zip(call_names, comps))
        args.append(arg)
      return args

    elements = anonymous_tuple.to_elements(comp)
    call_names = [name for name, _ in elements]
    comps = _get_comps(comp)
    args = _create_transformed_args_from_comps(call_names, comps)
    arg = computation_building_blocks.Tuple(args)
    parameter_type = computation_types.to_type(arg.type_signature)
    result_type = [(name, call.type_signature) for name, call in elements]
    intrinsic_type = computation_types.FunctionType(parameter_type, result_type)
    intrinsic = computation_building_blocks.Intrinsic(comp[0].function.uri,
                                                      intrinsic_type)
    transformed_comp = computation_building_blocks.Call(intrinsic, arg)
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


def merge_chained_blocks(comp):
  r"""Merges Block constructs defined in sequence in the AST of `comp`.

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
    comp: The `computation_building_blocks.ComputationBuildingBlock` whose
      blocks should be merged if possible.

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
    transformed_comp = computation_building_blocks.Block(comp.locals +
                                                         comp.result.locals,
                                                         comp.result.result)
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


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
