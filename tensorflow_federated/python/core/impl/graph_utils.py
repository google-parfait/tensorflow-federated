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
"""Utilities for interacting with and manipulating TensorFlow graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import itertools
import logging

import attr
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl import dtype_utils
from tensorflow_federated.python.core.impl import type_utils


def stamp_parameter_in_graph(parameter_name, parameter_type, graph):
  """Stamps a parameter of a given type in the given tf.Graph instance.

  Tensors are stamped as placeholders, sequences are stamped as data sets
  constructed from string tensor handles, and named tuples are stamped by
  independently stamping their elements.

  Args:
    parameter_name: The suggested (string) name of the parameter to use in
      determining the names of the graph components to construct. The names that
      will actually appear in the graph are not guaranteed to be based on this
      suggested name, and may vary, e.g., due to existing naming conflicts, but
      a best-effort attempt will be made to make them similar for ease of
      debugging.
    parameter_type: The type of the parameter to stamp. Must be either an
      instance of computation_types.Type (or convertible to it), or None.
    graph: The instance of tf.Graph to stamp in.

  Returns:
    A tuple (val, binding), where 'val' is a Python object (such as a dataset,
    a placeholder, or a dictionary that represents a named tuple) that
    represents the stamped parameter for use in the body of a Python function
    that consumes this parameter, and the 'binding' is an instance of
    TensorFlow.Binding that indicates how parts of the type signature relate
    to the tensors and ops stamped into the graph.

  Raises:
    TypeError: If the arguments are of the wrong computation_types.
    ValueError: If the parameter type cannot be stamped in a TensorFlow graph.
  """
  py_typecheck.check_type(parameter_name, six.string_types)
  py_typecheck.check_type(graph, tf.Graph)
  if parameter_type is None:
    return (None, None)
  parameter_type = computation_types.to_type(parameter_type)
  if isinstance(parameter_type, computation_types.TensorType):
    with graph.as_default():
      placeholder = tf.placeholder(
          dtype=parameter_type.dtype,
          shape=parameter_type.shape,
          name=parameter_name)
      binding = pb.TensorFlow.Binding(
          tensor=pb.TensorFlow.TensorBinding(tensor_name=placeholder.name))
      return (placeholder, binding)
  elif isinstance(parameter_type, computation_types.NamedTupleType):
    element_name_value_pairs = []
    element_bindings = []
    for e in anonymous_tuple.to_elements(parameter_type):
      e_val, e_binding = stamp_parameter_in_graph(
          '{}_{}'.format(parameter_name, e[0]), e[1], graph)
      element_name_value_pairs.append((e[0], e_val))
      element_bindings.append(e_binding)
    return (anonymous_tuple.AnonymousTuple(element_name_value_pairs),
            pb.TensorFlow.Binding(
                tuple=pb.TensorFlow.NamedTupleBinding(
                    element=element_bindings)))
  elif isinstance(parameter_type, computation_types.SequenceType):
    with graph.as_default():
      variant_tensor = tf.placeholder(tf.variant, shape=[])
      ds = make_dataset_from_variant_tensor(variant_tensor,
                                            parameter_type.element)
    return (ds,
            pb.TensorFlow.Binding(
                sequence=pb.TensorFlow.SequenceBinding(
                    variant_tensor_name=variant_tensor.name)))
  else:
    raise ValueError(
        'Parameter type component {} cannot be stamped into a TensorFlow '
        'graph.'.format(repr(parameter_type)))


class OneShotDataset(typed_object.TypedObject):
  """A factory of `tf.data.Dataset`-like objects based on a no-argument lambda.

  This factory supports the same methods as the data sets constructed by the
  lambda. Upon invocation, it constructs a new data set by invoking the lambda,
  then forwards the call to that data set. A new data set is created per call.
  """

  # TODO(b/129956296): Eventually delete this deprecated class.

  def __init__(self, fn, element_type):
    """Constructs this factory from `fn`.

    Args:
      fn: A no-argument callable that creates instances of `tf.data.Dataset`.
      element_type: The type of elements.
    """
    # TODO(b/131426323) Possibly reuse TensorFlow's @deprecation.deprecated()
    # here if possible.
    logging.warning('OneShotDataset is deprecated.')
    py_typecheck.check_type(element_type, computation_types.Type)
    self._type_signature = computation_types.SequenceType(element_type)
    self._fn = fn

  @property
  def type_signature(self):
    """Returns the TFF type of this object (an instance of `tff.Type`)."""
    return self._type_signature

  def __getattr__(self, name):
    return getattr(self._fn(), name)


# TODO(b/129956296): Eventually delete this deprecated declaration.
DATASET_REPRESENTATION_TYPES = (tf.data.Dataset, tf.compat.v1.data.Dataset,
                                tf.compat.v2.data.Dataset, OneShotDataset)


def make_dataset_from_string_handle(handle, type_spec):
  """Constructs a `tf.data.Dataset` from a string handle tensor and type spec.

  Args:
    handle: The tensor that represents the string handle.
    type_spec: The type spec of elements of the data set, either an instance of
      `types.Type` or something convertible to it.

  Returns:
    A corresponding instance of `tf.data.Dataset`.
  """
  # TODO(b/129956296): Eventually delete this deprecated code path.

  type_spec = computation_types.to_type(type_spec)
  tf_dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes(type_spec)

  def make(handle=handle, tf_dtypes=tf_dtypes, shapes=shapes):
    """An embedded no-argument function that constructs the data set on-demand.

    This is invoked by `OneShotDataset` on each access to the data set argument
    passed to the body of the TF computation to ensure that the iterators and
    tje map are constructed in the appropriate context (e.g., in a defun).

    Args:
      handle: Captured from the local (above).
      tf_dtypes: Captured from the local (above).
      shapes: Captured from the local (above).

    Returns:
      An instance of `tf.data.Dataset`.
    """
    with handle.graph.as_default():
      it = tf.data.Iterator.from_string_handle(handle, tf_dtypes, shapes)
      # In order to convert an iterator into something that looks like a data
      # set, we create a dummy data set that consists of an infinite sequence
      # of zeroes, and filter it through a map function that invokes
      # 'it.get_next()' for each of those zeroes.
      # TODO(b/113112108): Possibly replace this with something more canonical
      # if and when we can find adequate support for abstractly defined data
      # sets (at the moment of this writing it does not appear to exist yet).
      return tf.data.Dataset.range(1).repeat().map(lambda _: it.get_next())

  # NOTE: To revert to the old behavior, simply invoke `make()` here directly.
  return OneShotDataset(make, type_spec)


def make_dataset_from_variant_tensor(variant_tensor, type_spec):
  """Constructs a `tf.data.Dataset` from a variant tensor and type spec.

  Args:
    variant_tensor: The variant tensor that represents the dataset.
    type_spec: The type spec of elements of the data set, either an instance of
      `types.Type` or something convertible to it.

  Returns:
    A corresponding instance of `tf.data.Dataset`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  if not tf.is_tensor(variant_tensor):
    raise TypeError(
        'Expected `variant_tensor` to be a tensor, found {}.'.format(
            py_typecheck.type_string(type(variant_tensor))))
  if variant_tensor.dtype != tf.variant:
    raise TypeError(
        'Expected `variant_tensor` to be of a variant type, found {}.'.format(
            str(variant_tensor.dtype)))
  return tf.data.experimental.from_variant(
      variant_tensor,
      structure=(type_utils.type_to_tf_structure(
          computation_types.to_type(type_spec))))


def capture_result_from_graph(result, graph):
  """Captures a result stamped into a tf.Graph as a type signature and binding.

  Args:
    result: The result to capture, a Python object that is composed of tensors,
      possibly nested within Python structures such as dictionaries, lists,
      tuples, or named tuples.
    graph: The instance of tf.Graph to use.

  Returns:
    A tuple (type_spec, binding), where 'type_spec' is an instance of
    computation_types.Type that describes the type of the result, and 'binding'
    is an instance of TensorFlow.Binding that indicates how parts of the result
    type relate to the tensors and ops that appear in the result.

  Raises:
    TypeError: If the argument or any of its parts are of an uexpected type.
  """

  def _get_bindings_for_elements(name_value_pairs, graph, type_fn):
    """Build `(type_spec, binding)` tuple for name value pairs."""
    element_name_type_binding_triples = [
        ((k,) + capture_result_from_graph(v, graph))
        for k, v in name_value_pairs
    ]
    type_spec = type_fn([((e[0], e[1]) if e[0] else e[1])
                         for e in element_name_type_binding_triples])
    binding = pb.TensorFlow.Binding(
        tuple=pb.TensorFlow.NamedTupleBinding(
            element=[e[2] for e in element_name_type_binding_triples]))
    return type_spec, binding

  # TODO(b/113112885): The emerging extensions for serializing SavedModels may
  # end up introducing similar concepts of bindings, etc., we should look here
  # into the possibility of reusing some of that code when it's available.
  if isinstance(result, dtype_utils.TENSOR_REPRESENTATION_TYPES):
    with graph.as_default():
      result = tf.constant(result)
  if tf.is_tensor(result):
    if hasattr(result, 'read_value'):
      # We have a tf.Variable-like result, get a proper tensor to fetch.
      with graph.as_default():
        result = result.read_value()
    return (computation_types.TensorType(result.dtype.base_dtype, result.shape),
            pb.TensorFlow.Binding(
                tensor=pb.TensorFlow.TensorBinding(tensor_name=result.name)))
  elif py_typecheck.is_named_tuple(result):
    # Special handling needed for collections.namedtuples since they do not have
    # anything in the way of a shared base class. Note we don't want to rely on
    # the fact that collections.namedtuples inherit from 'tuple' because we'd be
    # failing to retain the information about naming of tuple members.
    # pylint: disable=protected-access
    name_value_pairs = six.iteritems(result._asdict())
    # pylint: enable=protected-access
    return _get_bindings_for_elements(
        name_value_pairs, graph,
        functools.partial(
            computation_types.NamedTupleTypeWithPyContainerType,
            container_type=type(result)))
  elif py_typecheck.is_attrs(result):
    name_value_pairs = attr.asdict(
        result, dict_factory=collections.OrderedDict, recurse=False)
    return _get_bindings_for_elements(
        six.iteritems(name_value_pairs), graph,
        functools.partial(
            computation_types.NamedTupleTypeWithPyContainerType,
            container_type=type(result)))
  elif isinstance(result, anonymous_tuple.AnonymousTuple):
    return _get_bindings_for_elements(
        anonymous_tuple.to_elements(result), graph,
        computation_types.NamedTupleType)
  elif isinstance(result, dict):
    if isinstance(result, collections.OrderedDict):
      name_value_pairs = six.iteritems(result)
    else:
      name_value_pairs = sorted(six.iteritems(result))
    return _get_bindings_for_elements(
        name_value_pairs, graph,
        functools.partial(
            computation_types.NamedTupleTypeWithPyContainerType,
            container_type=type(result)))
  elif isinstance(result, (list, tuple)):
    element_type_binding_pairs = [
        capture_result_from_graph(e, graph) for e in result
    ]
    return (computation_types.NamedTupleTypeWithPyContainerType(
        [e[0] for e in element_type_binding_pairs], type(result)),
            pb.TensorFlow.Binding(
                tuple=pb.TensorFlow.NamedTupleBinding(
                    element=[e[1] for e in element_type_binding_pairs])))
  elif isinstance(result,
                  (tf.compat.v1.data.Dataset, tf.compat.v2.data.Dataset)):
    variant_tensor = tf.data.experimental.to_variant(result)
    # TODO(b/130032140): Switch to TF2.0 way of doing it while cleaning up the
    # legacy structures all over the code base and replacing them with the new
    # tf.data.experimenta.Structure variants.
    element_type = type_utils.tf_dtypes_and_shapes_to_type(
        tf.compat.v1.data.get_output_types(result),
        tf.compat.v1.data.get_output_shapes(result))
    return (computation_types.SequenceType(element_type),
            pb.TensorFlow.Binding(
                sequence=pb.TensorFlow.SequenceBinding(
                    variant_tensor_name=variant_tensor.name)))
  elif isinstance(result, OneShotDataset):
    # TODO(b/129956296): Eventually delete this deprecated code path.
    element_type = type_utils.tf_dtypes_and_shapes_to_type(
        tf.compat.v1.data.get_output_types(result),
        tf.compat.v1.data.get_output_shapes(result))
    handle_name = result.make_one_shot_iterator().string_handle().name
    return (computation_types.SequenceType(element_type),
            pb.TensorFlow.Binding(
                sequence=pb.TensorFlow.SequenceBinding(
                    iterator_string_handle_name=handle_name)))
  else:
    raise TypeError('Cannot capture a result of an unsupported type {}.'.format(
        py_typecheck.type_string(type(result))))


def compute_map_from_bindings(source, target):
  """Computes a dictionary for renaming tensors from a matching bindings pair.

  Args:
    source: An instance of `pb.TensorFlow.Binding` that contains names of
      tensors that will form the keys in the dictionary.
    target: An instance of `pb.TensorFlow.Binding` that contains names of
      tensors that will form the values in the dictionary. The structure of this
      binding must be identical as that of the `source`.

  Returns:
    A dictionary mapping names of tensors in `source` to names of the
    tensors in the corresponding parts of `target`.

  Raises:
    TypeError: If the arguments are of the wrong computation_types.
    ValueError: If the bindings have mismatching structures.
  """
  py_typecheck.check_type(source, pb.TensorFlow.Binding)
  py_typecheck.check_type(target, pb.TensorFlow.Binding)
  source_oneof = source.WhichOneof('binding')
  target_oneof = target.WhichOneof('binding')
  if source_oneof != target_oneof:
    raise ValueError(
        'Source and target binding variants mismatch: {} vs. {}'.format(
            source_oneof, target_oneof))
  if source_oneof == 'tensor':
    return collections.OrderedDict([(str(source.tensor.tensor_name),
                                     str(target.tensor.tensor_name))])
  elif source_oneof == 'sequence':
    sequence_oneof = source.sequence.WhichOneof('binding')
    if target.sequence.WhichOneof('binding') != sequence_oneof:
      raise ValueError(
          'Source and target sequence bindings mismatch: {} vs. {}'.format(
              sequence_oneof, target.sequence.WhichOneof('binding')))
    if sequence_oneof == 'iterator_string_handle_name':
      # TODO(b/129956296): Eventually delete this deprecated code path.
      return collections.OrderedDict([
          (str(source.sequence.iterator_string_handle_name),
           str(target.sequence.iterator_string_handle_name))
      ])
    elif sequence_oneof == 'variant_tensor_name':
      return collections.OrderedDict([
          (str(source.sequence.variant_tensor_name),
           str(target.sequence.variant_tensor_name)),
      ])
    else:
      raise ValueError('Unsupported sequence binding {}'.format(sequence_oneof))
  elif source_oneof == 'tuple':
    if len(source.tuple.element) != len(target.tuple.element):
      raise ValueError(
          'Source and target binding tuple lengths mismatch: {} vs. {}.'.format(
              len(source.tuple.element), len(target.tuple.element)))
    else:
      result = collections.OrderedDict()
      for source_element, target_element in zip(source.tuple.element,
                                                target.tuple.element):
        result.update(compute_map_from_bindings(source_element, target_element))
      return result
  else:
    raise ValueError('Unsupported type of binding \'{}\'.'.format(source_oneof))


def extract_tensor_names_from_binding(binding):
  """Returns a list of tensor names extracted from a given binding.

  Args:
    binding: An instance of `pb.TensorFlow.Binding`.

  Returns:
    All tensor names that appear in `binding`.
  """
  py_typecheck.check_type(binding, pb.TensorFlow.Binding)
  binding_oneof = binding.WhichOneof('binding')
  if binding_oneof == 'tensor':
    return [str(binding.tensor.tensor_name)]
  elif binding_oneof == 'sequence':
    sequence_oneof = binding.sequence.WhichOneof('binding')
    if sequence_oneof == 'iterator_string_handle_name':
      # TODO(b/129956296): Eventually delete this deprecated code path.
      return [str(binding.sequence.iterator_string_handle_name)]
    elif sequence_oneof == 'variant_tensor_name':
      return [str(binding.sequence.variant_tensor_name)]
    else:
      raise ValueError('Unsupported sequence binding {}'.format(sequence_oneof))
  elif binding_oneof == 'tuple':
    return list(
        itertools.chain.from_iterable([
            extract_tensor_names_from_binding(e) for e in binding.tuple.element
        ]))
  else:
    raise ValueError(
        'Unsupported type of binding \'{}\'.'.format(binding_oneof))


def assemble_result_from_graph(type_spec, binding, output_map):
  """Assembles a result stamped into a `tf.Graph` given type signature/binding.

  This method does roughly the opposite of `capture_result_from_graph`, in that
  whereas `capture_result_from_graph` starts with a single structured object
  made up of tensors and computes its type and bindings, this method starts
  with the type/bindings and constructs a structured object made up of tensors.

  Args:
    type_spec: The type signature of the result to assemble, an instance of
      `types.Type` or something convertible to it.
    binding: The binding that relates the type signature to names of tensors in
      the graph, an instance of `pb.TensorFlow.Binding`.
    output_map: The mapping from tensor names that appear in the binding to
      actual stamped tensors (possibly renamed during import).

  Returns:
    The assembled result, a Python object that is composed of tensors, possibly
    nested within Python structures such as anonymous tuples.

  Raises:
    TypeError: If the argument or any of its parts are of an uexpected type.
    ValueError: If the arguments are invalid or inconsistent witch other, e.g.,
      the type and binding don't match, or the tensor is not found in the map.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  py_typecheck.check_type(binding, pb.TensorFlow.Binding)
  py_typecheck.check_type(output_map, dict)
  for k, v in six.iteritems(output_map):
    py_typecheck.check_type(k, six.string_types)
    if not tf.is_tensor(v):
      raise TypeError(
          'Element with key {} in the output map is {}, not a tensor.'.format(
              k, py_typecheck.type_string(type(v))))

  binding_oneof = binding.WhichOneof('binding')
  if isinstance(type_spec, computation_types.TensorType):
    if binding_oneof != 'tensor':
      raise ValueError(
          'Expected a tensor binding, found {}.'.format(binding_oneof))
    elif binding.tensor.tensor_name not in output_map:
      raise ValueError('Tensor named {} not found in the output map.'.format(
          binding.tensor.tensor_name))
    else:
      return output_map[binding.tensor.tensor_name]
  elif isinstance(type_spec, computation_types.NamedTupleType):
    if binding_oneof != 'tuple':
      raise ValueError(
          'Expected a tuple binding, found {}.'.format(binding_oneof))
    else:
      type_elements = anonymous_tuple.to_elements(type_spec)
      if len(binding.tuple.element) != len(type_elements):
        raise ValueError(
            'Mismatching tuple sizes in type ({}) and binding ({}).'.format(
                len(type_elements), len(binding.tuple.element)))
      result_elements = []
      for (element_name,
           element_type), element_binding in zip(type_elements,
                                                 binding.tuple.element):
        element_object = assemble_result_from_graph(element_type,
                                                    element_binding, output_map)
        result_elements.append((element_name, element_object))
      if not isinstance(type_spec,
                        computation_types.NamedTupleTypeWithPyContainerType):
        return anonymous_tuple.AnonymousTuple(result_elements)
      container_type = computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
          type_spec)
      if (py_typecheck.is_named_tuple(container_type) or
          py_typecheck.is_attrs(container_type)):
        return container_type(**dict(result_elements))
      return container_type(result_elements)
  elif isinstance(type_spec, computation_types.SequenceType):
    if binding_oneof != 'sequence':
      raise ValueError(
          'Expected a sequence binding, found {}.'.format(binding_oneof))
    else:
      sequence_oneof = binding.sequence.WhichOneof('binding')
      if sequence_oneof == 'iterator_string_handle_name':
        # TODO(b/129956296): Eventually delete this deprecated code path.
        handle = output_map[binding.sequence.iterator_string_handle_name]
        return make_dataset_from_string_handle(handle, type_spec.element)
      elif sequence_oneof == 'variant_tensor_name':
        variant_tensor = output_map[binding.sequence.variant_tensor_name]
        return make_dataset_from_variant_tensor(variant_tensor,
                                                type_spec.element)
      else:
        raise ValueError(
            'Unsupported sequence binding \'{}\'.'.format(sequence_oneof))
  else:
    raise ValueError('Unsupported type \'{}\'.'.format(str(type_spec)))


def nested_structures_equal(x, y):
  """Determines if nested structures `x` and `y` are equal.

  Args:
    x: A nested structure.
    y: Another nested structure.

  Returns:
    `True` iff `x` and `y` are equal, `False` otherwise.
  """
  try:
    tf.nest.assert_same_structure(x, y)
  except ValueError:
    return False
  return tf.nest.flatten(x) == tf.nest.flatten(y)


def make_empty_list_structure_for_element_type_spec(type_spec):
  """Creates a nested structure of empty Python lists for `type_spec`.

  This function prepares a nested structure made of `collections.OrderedDict`s
  and Python `tuple`s at the intermediate (non-leaf) levels, and that has empty
  Python `list`s at the leaf level, with the shape of the structure matching
  that of `type_spec`. This structure is used to accumulate elemnts of a data
  set for ingestion by `tf.data.Dataset.from_tensor_slices`.

  Args:
    type_spec: An instance of `tff.Type` or something convertible to it that
      consists of only tensor and named tuple types, and in which rach of the
      named tuples either have all or none of their elements named.

  Returns:
    The nested structure, as described above.

  Raises:
    TypeError: If the `type_spec` is not of a form described above.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    return []
  elif isinstance(type_spec, computation_types.NamedTupleType):
    elements = anonymous_tuple.to_elements(type_spec)
    if all(k is not None for k, _ in elements):
      return collections.OrderedDict([
          (k, make_empty_list_structure_for_element_type_spec(v))
          for k, v in elements
      ])
    elif all(k is None for k, _ in elements):
      return tuple([
          make_empty_list_structure_for_element_type_spec(v)
          for _, v in elements
      ])
    else:
      raise TypeError(
          'Expected a named tuple type with either all elements named '
          'or all unnamed, got {}.'.format(str(type_spec)))
  else:
    raise TypeError('Expected a tensor or named tuple type, found {}.'.format(
        str(type_spec)))


def _make_dummy_element_for_type_spec(type_spec):
  """Creates ndarray of zeros corresponding to `type_spec`.

  Returns a list containing this ndarray, whose type is *compatible* with, not
  necessarily equal to, `type_spec`. This is due to the fact that some
  dimensions of `type_spec` may be indeterminate, representing compatibility
  of `type_spec` with any number (e.g. leaving a batch dimension indeterminate
  to signify compatibility with batches of any size). However a concrete
  structure (like the ndarray) must have specified sizes for its dimensions.
  So we construct a dummy element where any `None` dimensions of the shape
  of `type_spec` are replaced with the value 1. This function therefore
  returns a dummy element of minimal nonzero size which matches `type_spec`.

  Args:
    type_spec: Instance of `computation_types.Type`, or something convertible to
      one by `computation_types.to_type`.

  Returns:
    Returns possibly nested `numpy ndarray`s containing all zeros: a single
    `ndarray` if `type_spec` is a `computation_types.TensorType` and a list
    of such arrays if  `type_spec` is `computation_types.NamedTupleType`.
    This data structure is of the minimal nonzero size necessary in order to be
    compatible with `type_spec`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    dummy_shape = [x if x is not None else 1 for x in type_spec.shape]
    return np.zeros(dummy_shape, type_spec.dtype.as_numpy_dtype)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    elements = anonymous_tuple.to_elements(type_spec)
    elem_list = []
    for _, elem_type in elements:
      elem_list.append(_make_dummy_element_for_type_spec(elem_type))
    return elem_list
  else:
    raise TypeError('Expected a tensor or named tuple type, found {}.'.format(
        str(type_spec)))


def append_to_list_structure_for_element_type_spec(structure, value, type_spec):
  """Adds an element `value` to a nested `structure` of lists for `type_spec`.

  This function appends tensor-level constituents of an element `value` to the
  lists created by `make_empty_list_structure_for_element_type_spec`. The
  nested structure of `value` must match that created by the above function,
  and consistent with `type_spec`.

  Args:
    structure: Output of `make_empty_list_structure_for_element_type_spec`.
    value: A value (Python object) that a hierarchical structure of dictionary,
      list, and other containers holding tensor-like items that matches the
      hierarchy of `type_spec`.
    type_spec: An instance of `tff.Type` or something convertible to it, as in
      `make_empty_list_structure_for_element_type_spec`.

  Raises:
    TypeError: If the `type_spec` is not of a form described above, or the value
      is not of a type compatible with `type_spec`.
  """
  if value is None:
    return
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  # TODO(b/113116813): This could be made more efficient, but for now we won't
  # need to worry about it as this is an odd corner case.
  if isinstance(value, anonymous_tuple.AnonymousTuple):
    elements = anonymous_tuple.to_elements(value)
    if all(k is not None for k, _ in elements):
      value = collections.OrderedDict(elements)
    elif all(k is None for k, _ in elements):
      value = tuple([v for _, v in elements])
    else:
      raise TypeError(
          'Expected an anonymous tuple to either have all elements named '
          'or all unnamed, got {}.'.format(str(value)))
  if isinstance(type_spec, computation_types.TensorType):
    py_typecheck.check_type(structure, list)
    structure.append(value)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    elements = anonymous_tuple.to_elements(type_spec)
    if isinstance(structure, collections.OrderedDict):
      if py_typecheck.is_named_tuple(value):
        value = value._asdict()
      if isinstance(value, dict):
        if set(value.keys()) != set(k for k, _ in elements):
          raise TypeError('Value {} does not match type {}.'.format(
              str(value), str(type_spec)))
        for elem_name, elem_type in elements:
          append_to_list_structure_for_element_type_spec(
              structure[elem_name], value[elem_name], elem_type)
      elif isinstance(value, (list, tuple)):
        if len(value) != len(elements):
          raise TypeError('Value {} does not match type {}.'.format(
              str(value), str(type_spec)))
        for idx, (elem_name, elem_type) in enumerate(elements):
          append_to_list_structure_for_element_type_spec(
              structure[elem_name], value[idx], elem_type)
      else:
        raise TypeError('Unexpected type of value {} for TFF type {}.'.format(
            py_typecheck.type_string(type(value)), str(type_spec)))
    elif isinstance(structure, tuple):
      py_typecheck.check_type(value, (list, tuple))
      if len(value) != len(elements):
        raise TypeError('Value {} does not match type {}.'.format(
            str(value), str(type_spec)))
      for idx, (_, elem_type) in enumerate(elements):
        append_to_list_structure_for_element_type_spec(structure[idx],
                                                       value[idx], elem_type)
    else:
      raise TypeError(
          'Invalid nested structure, unexpected container type {}.'.format(
              py_typecheck.type_string(type(structure))))
  else:
    raise TypeError('Expected a tensor or named tuple type, found {}.'.format(
        str(type_spec)))


def to_tensor_slices_from_list_structure_for_element_type_spec(
    structure, type_spec):
  """Converts `structure` for use with `tf.data.Dataset.from_tensor_slices`.

  This function wraps lists in the leaves of `structure` with `np.array()` for
  consumption by `tf.data.Dataset.from_tensor_slices`.

  Args:
    structure: Output of `make_empty_list_structure_for_element_type_spec`.
    type_spec: An instance of `tff.Type` or something convertible to it, as in
      `make_empty_list_structure_for_element_type_spec`.

  Returns:
    The transformed version of `structure`, in a form that can be consumed and
    correctly parsed by `tf.data.Dataset.from_tensor_slices`.

  Raises:
    TypeError: If the `type_spec` is not of a form described above, or the
      structure is not of a type compatible with `type_spec`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    py_typecheck.check_type(structure, list)
    # TODO(b/113116813): Perhaps this can be done more selectively, based on the
    # type of elements on the list as well as the `type_spec`. Also, passing the
    # explicit `dtype` here will trigger implicit conversion, e.g., from `int32`
    # to `bool`, which may not be desirable.
    return np.array(structure, dtype=type_spec.dtype.as_numpy_dtype)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    elements = anonymous_tuple.to_elements(type_spec)
    if isinstance(structure, collections.OrderedDict):
      to_return = []
      for elem_name, elem_type in elements:
        elem_val = to_tensor_slices_from_list_structure_for_element_type_spec(
            structure[elem_name], elem_type)
        to_return.append((elem_name, elem_val))
      return collections.OrderedDict(to_return)
    elif isinstance(structure, tuple):
      to_return = []
      for idx, (_, elem_type) in enumerate(elements):
        elem_val = to_tensor_slices_from_list_structure_for_element_type_spec(
            structure[idx], elem_type)
        to_return.append(elem_val)
      return tuple(to_return)
    else:
      raise TypeError(
          'Invalid nested structure, unexpected container type {}.'.format(
              py_typecheck.type_string(type(structure))))
  else:
    raise TypeError('Expected a tensor or named tuple type, found {}.'.format(
        str(type_spec)))


def make_data_set_from_elements(graph, elements, element_type):
  """Creates a `tf.data.Dataset` in `graph` from explicitly listed `elements`.

  NOTE: The underlying implementation attempts to use the
  `tf.data.Dataset.from_tensor_slices() method to build the data set quickly,
  but this doesn't always work. The typical scenario where it breaks is one
  with data set being composed of unequal batches. Typically, only the last
  batch is odd, so on the first attempt, we try to construct two data sets,
  one from all elements but the last one, and one from the last element, then
  concatenate the two. In the unlikely case that this fails (e.g., because
  all data set elements are batches of unequal sizes), we revert to the slow,
  but reliable method of constructing data sets from singleton elements, and
  then concatenating them all.

  Args:
    graph: The graph in which to construct the `tf.data.Dataset`.
    elements: A list of elements.
    element_type: The type of elements.

  Returns:
    The constructed `tf.data.Dataset` instance.

  Raises:
    TypeError: If element types do not match `element_type`.
    ValueError: If the elements are of incompatible types and shapes.
  """
  py_typecheck.check_type(graph, tf.Graph)
  py_typecheck.check_type(elements, list)
  element_type = computation_types.to_type(element_type)
  py_typecheck.check_type(element_type, computation_types.Type)

  def _make(element_subset):
    structure = make_empty_list_structure_for_element_type_spec(element_type)
    for el in element_subset:
      append_to_list_structure_for_element_type_spec(structure, el,
                                                     element_type)
    tensor_slices = to_tensor_slices_from_list_structure_for_element_type_spec(
        structure, element_type)
    return tf.data.Dataset.from_tensor_slices(tensor_slices)

  with graph.as_default():
    if not elements:
      # Just return an empty data set with the appropriate types.
      dummy_element = _make_dummy_element_for_type_spec(element_type)
      ds = _make([dummy_element]).take(0)
    elif len(elements) == 1:
      ds = _make(elements)
    else:
      try:
        # It is common for the last element to be a batch of a size different
        # from all the preceding batches. With this in mind, we proactively
        # single out the last element (optimizing for the common case).
        ds = _make(elements[0:-1]).concatenate(_make(elements[-1:]))
      except ValueError:
        # In case elements beyond just the last one are of unequal shapes, we
        # may have failed (the most likely cause), so fall back onto the slow
        # process of constructing and joining data sets from singletons. Not
        # optimizing this for now, as it's very unlikely in scenarios
        # we're targeting.
        ds = None
        for i in range(len(elements)):
          singleton_ds = _make(elements[i:i + 1])
          ds = singleton_ds if ds is None else ds.concatenate(singleton_ds)
    ds_element_type = type_utils.tf_dtypes_and_shapes_to_type(
        tf.compat.v1.data.get_output_types(ds),
        tf.compat.v1.data.get_output_shapes(ds))
    if not type_utils.is_assignable_from(element_type, ds_element_type):
      raise TypeError(
          'Failure during data set construction, expected elements of type {}, '
          'but the constructed data set has elements of type {}.'.format(
              str(element_type), str(ds_element_type)))
  return ds


def fetch_value_in_session(sess, value):
  """Fetches `value` in `session`.

  Args:
    sess: The session in which to perform the fetch (as a single run).
    value: A Python object of a form analogous to that constructed by the
      function `assemble_result_from_graph`, made of tensors and anononymous
      tuples, or a `tf.data.Dataset`.

  Returns:
    A Python object with structure similar to `value`, but with tensors
    replaced with their values, and data sets replaced with lists of their
    elements, all fetched with a single call `session.run()`.

  Raises:
    ValueError: If `value` is not a `tf.data.Dataset` or not a structure of
      tensors and anonoymous tuples.
  """
  py_typecheck.check_type(sess, tf.Session)
  # TODO(b/113123634): Investigate handling `list`s and `tuple`s of
  # `tf.data.Dataset`s and what the API would look like to support this.
  if isinstance(value, DATASET_REPRESENTATION_TYPES):
    with sess.graph.as_default():
      iterator = tf.compat.v1.data.make_one_shot_iterator(value)
      next_element = iterator.get_next()
    elements = []
    while True:
      try:
        elements.append(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        break
    return elements
  else:
    flattened_value = anonymous_tuple.flatten(value)
    for v in flattened_value:
      if not tf.is_tensor(v):
        raise ValueError('Unsupported value type {}.'.format(str(v)))
    flattened_results = sess.run(flattened_value)

    def _to_unicode(v):
      if six.PY3 and isinstance(v, bytes):
        return v.decode('utf-8')
      return v

    if tf.is_tensor(value) and value.dtype == tf.string:
      flattened_results = [_to_unicode(result) for result in flattened_results]
    return anonymous_tuple.pack_sequence_as(value, flattened_results)
