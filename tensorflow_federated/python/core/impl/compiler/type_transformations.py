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
# limitations under the License.
"""A library of transformations that can be applied to a computation type."""

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types


def transform_type_postorder(type_signature, transform_fn):
  """Walks type tree of `type_signature` postorder, calling `transform_fn`.

  Args:
    type_signature: Instance of `computation_types.Type` to transform
      recursively.
    transform_fn: Transformation function to apply to each node in the type tree
      of `type_signature`. Must be instance of Python function type.

  Returns:
    A possibly transformed version of `type_signature`, with each node in its
    tree the result of applying `transform_fn` to the corresponding node in
    `type_signature`.

  Raises:
    TypeError: If the types don't match the specification above.
  """
  # TODO(b/134525440): Investigate unifying the recursive methods in type_utils,
  # rather than proliferating them.
  # TODO(b/134595038): Revisit the change here to add a mutated flag.
  py_typecheck.check_type(type_signature, computation_types.Type)
  py_typecheck.check_callable(transform_fn)
  if isinstance(type_signature, computation_types.FederatedType):
    transformed_member, member_mutated = transform_type_postorder(
        type_signature.member, transform_fn)
    if member_mutated:
      type_signature = computation_types.FederatedType(transformed_member,
                                                       type_signature.placement,
                                                       type_signature.all_equal)
    fed_type_signature, type_signature_mutated = transform_fn(type_signature)
    return fed_type_signature, type_signature_mutated or member_mutated
  elif isinstance(type_signature, computation_types.SequenceType):
    transformed_element, element_mutated = transform_type_postorder(
        type_signature.element, transform_fn)
    if element_mutated:
      type_signature = computation_types.SequenceType(transformed_element)
    seq_type_signature, type_signature_mutated = transform_fn(type_signature)
    return seq_type_signature, type_signature_mutated or element_mutated
  elif isinstance(type_signature, computation_types.FunctionType):
    transformed_param, param_mutated = transform_type_postorder(
        type_signature.parameter, transform_fn)
    transformed_result, result_mutated = transform_type_postorder(
        type_signature.result, transform_fn)
    if param_mutated or result_mutated:
      type_signature = computation_types.FunctionType(transformed_param,
                                                      transformed_result)
    fn_type_signature, fn_mutated = transform_fn(type_signature)
    return fn_type_signature, fn_mutated or param_mutated or result_mutated
  elif isinstance(type_signature, computation_types.NamedTupleType):
    elems = []
    elems_mutated = False
    for element in anonymous_tuple.iter_elements(type_signature):
      transformed_element, element_mutated = transform_type_postorder(
          element[1], transform_fn)
      elems_mutated = elems_mutated or element_mutated
      elems.append((element[0], transformed_element))
    if elems_mutated:
      if isinstance(type_signature,
                    computation_types.NamedTupleTypeWithPyContainerType):
        type_signature = computation_types.NamedTupleTypeWithPyContainerType(
            elems,
            computation_types.NamedTupleTypeWithPyContainerType
            .get_container_type(type_signature))
      else:
        type_signature = computation_types.NamedTupleType(elems)
    tuple_type_signature, tuple_mutated = transform_fn(type_signature)
    return tuple_type_signature, elems_mutated or tuple_mutated
  elif isinstance(type_signature,
                  (computation_types.AbstractType, computation_types.TensorType,
                   computation_types.PlacementType)):
    return transform_fn(type_signature)
