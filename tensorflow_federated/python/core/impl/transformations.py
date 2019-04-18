# Lint as: python2, python3
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
    """Internal transform function."""
    if not _should_transform(comp):
      return comp
    # We need 'wrapped_body' to accept exactly one argument.
    wrapped_body = lambda x: body(x)  # pylint: disable=unnecessary-lambda
    return federated_computation_utils.zero_or_one_arg_fn_to_building_block(
        wrapped_body,
        'arg',
        comp.type_signature.parameter,
        context_stack,
        suggested_name=uri)

  return transformation_utils.transform_postorder(comp, _transform)


def replace_called_lambda_with_block(comp):
  r"""Replaces all the called lambdas in `comp` with a block.

  This transform traverses `comp` postorder, matches the following pattern `*`,
  and replaces the following computation containing a called lambda:

            *Call
            /    \
     *Lambda      Comp(y)
            \
             Comp(x)

  (arg -> x)(y)

  with the following computation containing a block:

                    Block
                   /     \
   arg=Computation(y)       Computation(x)

  let arg=y in x

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
                 /     \
          *Lambda       Comp(x)
                 \
                 *Ref(arg)

  Intrinsic(<(arg -> arg), x>)

  with its argument:

  Comp(x)

  x

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _is_identity_function(comp):
    """Returns `True` if `comp` is an identity function."""
    return (isinstance(comp, computation_building_blocks.Lambda) and
            isinstance(comp.result, computation_building_blocks.Reference) and
            comp.parameter_name == comp.result.name)

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
                 /     \
          Comp(x)      *Call
                       /    \
             *Intrinsic     *Tuple
                            /     \
                     Comp(y)       Comp(z)

  federated_map(<x, federated_map(<y, z>)>)

  with the following computation containing one federated map intrinsic:

            Call
           /    \
  Intrinsic      Tuple
                /     \
          Lambda       Comp(z)
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

  def _is_federated_map(comp):
    """Returns `True` if `comp` is a federated map."""
    return (isinstance(comp, computation_building_blocks.Call) and
            isinstance(comp.function, computation_building_blocks.Intrinsic) and
            comp.function.uri == intrinsic_defs.FEDERATED_MAP.uri)

  def _should_transform(comp):
    """Returns `True` if `comp` is a chained federated map."""
    if _is_federated_map(comp):
      outer_arg = comp.argument[1]
      if _is_federated_map(outer_arg):
        return True
    return False

  def _transform(comp):
    """Internal transform function."""
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
