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

import six

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl import intrinsic_bodies
from tensorflow_federated.python.core.impl import transformation_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks


def replace_intrinsics_with_callable(comp, uri, body, context_stack):
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
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(uri, six.string_types)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  if not callable(body):
    raise TypeError('The body of the intrinsic must be a callable.')

  def _should_transform(comp):
    return (isinstance(comp, building_blocks.Intrinsic) and comp.uri == uri and
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


def replace_all_intrinsics_with_bodies(comp, context_stack):
  """Iterates over all intrinsic bodies, inlining the intrinsics in `comp`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` in which we
      wish to replace all intrinsics with their bodies.
    context_stack: Instance of `context_stack_base.ContextStack`, the context
      stack to use for the bodies of the intrinsics.

  Returns:
    Instance of `building_blocks.ComputationBuildingBlock` with all
    the intrinsics from `intrinsic_bodies.py` inlined with their bodies, along
    with a Boolean indicating whether there was any inlining in fact done.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  bodies = intrinsic_bodies.get_intrinsic_bodies(context_stack)
  transformed = False
  for uri, body in six.iteritems(bodies):
    comp, uri_found = replace_intrinsics_with_callable(comp, uri, body,
                                                       context_stack)
    transformed = transformed or uri_found
  return comp, transformed
