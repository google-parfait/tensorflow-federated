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
"""A library of all transformations to be performed by the compiler pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_building_utils
from tensorflow_federated.python.core.impl import context_stack_base


def transform_postorder(comp, func):
  """Traverses `comp` recursively postorder and replaces its constituents.

  For each element of `comp` viewed as an expression tree, the transformation
  `func` is applied first to building blocks it is parameterized by, then the
  element itself. The transformation `func` should act as an identity function
  on the kinds of elements (computation building blocks) it does not care to
  transform. This corresponds to a post-order traversal of the expression tree,
  i.e., parameters are alwaysd transformed left-to-right (in the order in which
  they are listed in building block constructors), then the parent is visited
  and transformed with the already-visited, and possibly transformed arguments
  in place.

  NOTE: In particular, in `Call(f,x)`, both `f` and `x` are arguments to `Call`.
  Therefore, `f` is transformed into `f'`, next `x` into `x'` and finally,
  `Call(f',x')` is transformed at the end.

  Args:
    comp: The computation to traverse and transform bottom-up.
    func: The transformation to apply locally to each building block in `comp`.
      It is a Python function that accepts a building block at input, and should
      return either the same, or transformed building block at output. Both the
      intput and output of `func` are instances of `ComputationBuildingBlock`.

  Returns:
    The result of applying `func` to parts of `comp` in a bottom-up fashion.

  Raises:
    TypeError: If the arguments are of the wrong computation_types.
    NotImplementedError: If the argument is a kind of computation building block
      that is currently not recognized.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  if isinstance(
      comp,
      (computation_building_blocks.CompiledComputation,
       computation_building_blocks.Data, computation_building_blocks.Intrinsic,
       computation_building_blocks.Placement,
       computation_building_blocks.Reference)):
    return func(comp)
  elif isinstance(comp, computation_building_blocks.Selection):
    return func(
        computation_building_blocks.Selection(
            transform_postorder(comp.source, func), comp.name, comp.index))
  elif isinstance(comp, computation_building_blocks.Tuple):
    return func(
        computation_building_blocks.Tuple([(k, transform_postorder(
            v, func)) for k, v in anonymous_tuple.to_elements(comp)]))
  elif isinstance(comp, computation_building_blocks.Call):
    transformed_func = transform_postorder(comp.function, func)
    if comp.argument is not None:
      transformed_arg = transform_postorder(comp.argument, func)
    else:
      transformed_arg = None
    return func(
        computation_building_blocks.Call(transformed_func, transformed_arg))
  elif isinstance(comp, computation_building_blocks.Lambda):
    transformed_result = transform_postorder(comp.result, func)
    return func(
        computation_building_blocks.Lambda(
            comp.parameter_name, comp.parameter_type, transformed_result))
  elif isinstance(comp, computation_building_blocks.Block):
    return func(
        computation_building_blocks.Block(
            [(k, transform_postorder(v, func)) for k, v in comp.locals],
            transform_postorder(comp.result, func)))
  else:
    raise NotImplementedError(
        'Unrecognized computation building block: {}'.format(str(comp)))


def name_compiled_computations(comp):
  """Labels all compiled computations with names that are unique within `comp`.

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A modified variant of `comp` with all compiled computations given unique
    names.
  """

  def _name_generator():
    n = 0
    while True:
      n = n + 1
      yield str(n)

  def _transformation_func(x, name_sequence):
    if not isinstance(x, computation_building_blocks.CompiledComputation):
      return x
    else:
      return computation_building_blocks.CompiledComputation(
          x.proto, six.next(name_sequence))

  name_sequence = _name_generator()
  return transform_postorder(
      comp, lambda x: _transformation_func(x, name_sequence))


def replace_intrinsic(comp, uri, body, context_stack):
  """Replaces all occurrences of an intrinsic.

  Args:
    comp: The computation building block in which to perform the replacements.
    uri: The URI of the intrinsic to replace.
    body: A polymorphic callable that represents the body of the implementation
      of the intrinsic, i.e., one that given the parameter of the intrinsic
      constructs the intended result. This will typically be a Python function
      decorated with `@federated_computation` to make it into a polymorphic
      callable.
    context_stack: The context stack to use.

  Returns:
    A modified variant of `comp` with all occurrences of the intrinsic with
    the URI equal to `uri` replaced with the logic constructed by `replacement`.

  Raises:
    TypeError: if types do not match somewhere in the course of replacement.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(uri, six.string_types)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  if not callable(body):
    raise TypeError('The body of the intrinsic must be a callable.')

  def _transformation_func(comp, uri, body):
    """Internal function to replace occurrences of an intrinsic."""
    if not isinstance(comp, computation_building_blocks.Intrinsic):
      return comp
    elif comp.uri != uri:
      return comp
    else:
      py_typecheck.check_type(comp.type_signature,
                              computation_types.FunctionType)
      # We need 'wrapped_body' to accept exactly one argument.
      wrapped_body = lambda x: body(x)  # pylint: disable=unnecessary-lambda
      return computation_building_utils.zero_or_one_arg_func_to_lambda(
          wrapped_body, 'arg', comp.type_signature.parameter, context_stack)

  return transform_postorder(comp, lambda x: _transformation_func(x, uri, body))
