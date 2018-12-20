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

# Dependency imports

import six
from six.moves import zip
import tensorflow as tf

from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
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
    TypeError: if the arguments are of the wrong computation_types.
    ValueError: if the parameter type cannot be stamped in a TensorFlow graph.
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
    for e in parameter_type.elements:
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
      handle = tf.placeholder(tf.string, shape=[])
    ds = make_dataset_from_string_handle(handle, parameter_type.element)
    return (ds,
            pb.TensorFlow.Binding(
                sequence=pb.TensorFlow.SequenceBinding(
                    iterator_string_handle_name=handle.name)))
  else:
    raise ValueError(
        'Parameter type component {} cannot be stamped into a TensorFlow '
        'graph.'.format(repr(parameter_type)))


def make_dataset_from_string_handle(handle, type_spec):
  """Constructs a `tf.data.Dataset` from a string handle tensor and type spec.

  Args:
    handle: The tensor that represents the string handle.
    type_spec: The type spec of elements of the data set, either an instance of
      `types.Type` or something convertible to it.

  Returns:
    A corresponding instance of `tf.data.Dataset`.
  """
  type_spec = computation_types.to_type(type_spec)
  dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes(type_spec)
  with handle.graph.as_default():
    it = tf.data.Iterator.from_string_handle(handle, dtypes, shapes)
    # In order to convert an iterator into something that looks like a data
    # set, we create a dummy data set that consists of an infinite sequence
    # of zeroes, and filter it through a map function that invokes
    # 'it.get_next()' for each of those zeroes.
    # TODO(b/113112108): Possibly replace this with something more canonical
    # if and when we can find adequate support for abstractly defined data
    # sets (at the moment of this writing it does not appear to exist yet).
    return tf.data.Dataset.range(1).repeat().map(lambda _: it.get_next())


def capture_result_from_graph(result):
  """Captures a result stamped into a tf.Graph as a type signature and binding.

  Args:
    result: The result to capture, a Python object that is composed of tensors,
      possibly nested within Python structures such as dictionaries, lists,
      tuples, or named tuples.

  Returns:
    A tuple (type_spec, binding), where 'type_spec' is an instance of
    computation_types.Type that describes the type of the result, and 'binding'
    is an instance of TensorFlow.Binding that indicates how parts of the result
    type relate to the tensors and ops that appear in the result.

  Raises:
    TypeError: if the argument or any of its parts are of an uexpected type.
  """
  # TODO(b/113112885): The emerging extensions for serializing SavedModels may
  # end up introducing similar concepts of bindings, etc., we should look here
  # into the possibility of reusing some of that code when it's available.
  if tensor_util.is_tensor(result):
    return (computation_types.TensorType(result.dtype.base_dtype, result.shape),
            pb.TensorFlow.Binding(
                tensor=pb.TensorFlow.TensorBinding(tensor_name=result.name)))
  elif '_asdict' in type(result).__dict__:
    # Special handling needed for collections.namedtuples since they do not have
    # anything in the way of a shared base class. Note we don't want to rely on
    # the fact that collections.namedtuples inherit from 'tuple' because we'd be
    # failing to retain the information about naming of tuple members.
    # pylint: disable=protected-access
    return capture_result_from_graph(result._asdict())
    # pylint: enable=protected-access
  elif isinstance(result, (dict, anonymous_tuple.AnonymousTuple)):
    # This also handles 'OrderedDict', as it inherits from 'dict'.
    if isinstance(result, anonymous_tuple.AnonymousTuple):
      name_value_pairs = anonymous_tuple.to_elements(result)
    else:
      name_value_pairs = six.iteritems(result)
    element_name_type_binding_triples = [
        ((k,) + capture_result_from_graph(v)) for k, v in name_value_pairs
    ]
    return (computation_types.NamedTupleType(
        [((e[0], e[1]) if e[0] else e[1])
         for e in element_name_type_binding_triples]),
            pb.TensorFlow.Binding(
                tuple=pb.TensorFlow.NamedTupleBinding(
                    element=[e[2] for e in element_name_type_binding_triples])))
  elif isinstance(result, (list, tuple)):
    element_type_binding_pairs = [capture_result_from_graph(e) for e in result]
    return (computation_types.NamedTupleType(
        [e[0] for e in element_type_binding_pairs]),
            pb.TensorFlow.Binding(
                tuple=pb.TensorFlow.NamedTupleBinding(
                    element=[e[1] for e in element_type_binding_pairs])))
  elif isinstance(result, tf.data.Dataset):
    element_type = type_utils.tf_dtypes_and_shapes_to_type(
        result.output_types, result.output_shapes)
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
    return collections.OrderedDict(
        [(str(source.sequence.iterator_string_handle_name),
          str(target.sequence.iterator_string_handle_name))])
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
    return [str(binding.sequence.iterator_string_handle_name)]
  elif binding_oneof == 'tuple':
    return [
        name for e in binding.tuple.element
        for name in extract_tensor_names_from_binding(e)
    ]
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
    if not tensor_util.is_tensor(v):
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
      type_elements = type_spec.elements
      if len(binding.tuple.element) != len(type_elements):
        raise ValueError(
            'Mismatching tuple sizes in type ({}) and binding ({}).'.format(
                len(type_elements), len(binding.tuple.element)))
      result_elements = []
      for (element_name, element_type), element_binding in zip(
          type_elements, binding.tuple.element):
        element_object = assemble_result_from_graph(element_type,
                                                    element_binding, output_map)
        result_elements.append((element_name, element_object))
      return anonymous_tuple.AnonymousTuple(result_elements)
  elif isinstance(type_spec, computation_types.SequenceType):
    if binding_oneof != 'sequence':
      raise ValueError(
          'Expected a sequence binding, found {}.'.format(binding_oneof))
    else:
      handle = output_map[binding.sequence.iterator_string_handle_name]
      return make_dataset_from_string_handle(handle, type_spec.element)
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
    nest.assert_same_structure(x, y)
  except ValueError:
    return False
  return nest.flatten(x) == nest.flatten(y)


def make_data_set_from_elements(graph, elements, element_type):
  """Creates a `tf.data.Dataset` in `graph` from explicitly listed `elements`.

  Args:
    graph: The graph in which to construct the `tf.data.Dataset`.
    elements: A list of elements.
    element_type: The type of elements.

  Returns:
    The constructed `tf.data.Dataset` instance.

  Raises:
    TypeError: If element types do not match `element_type`.
  """
  # TODO(b/121032194): It would be great not to have to stitch the input by
  # concatenation, as done below. Upgrade this to something better after we
  # identify a better solution.

  py_typecheck.check_type(graph, tf.Graph)
  py_typecheck.check_type(elements, list)
  element_type = computation_types.to_type(element_type)
  py_typecheck.check_type(element_type, computation_types.Type)
  output_types, output_shapes = (
      type_utils.type_to_tf_dtypes_and_shapes(element_type))
  with graph.as_default():
    if not elements:
      # Just return an empty data set.
      return tf.data.Dataset.from_generator((lambda: ()),
                                            output_types=output_types,
                                            output_shapes=output_shapes)
    elif len(elements) > 1:
      return make_data_set_from_elements(
          graph, elements[:-1], element_type).concatenate(
              make_data_set_from_elements(graph, elements[-1:], element_type))
    else:
      ds = tf.data.Dataset.from_tensors(elements[0])
      if not nested_structures_equal(ds.output_types, output_types):
        raise TypeError('Expected dtypes {}, found {}.'.format(
            str(output_types), str(ds.output_types)))
      if not nested_structures_equal(ds.output_shapes, output_shapes):
        raise TypeError('Expected shapes {}, found {}.'.format(
            str(output_shapes), str(ds.output_shapes)))
      return ds


def fetch_value_in_session(value, session):
  """Fetches `value` in `session`.

  Args:
    value: A Python object of a form analogous to that constructed by the
      function `assemble_result_from_graph`, made of tensors, data sets, and
      anononymous tuples.
    session: The session in which to perform the fetch (as a single run).

  Returns:
    A Python object with structure similar to `value`, but with tensors
    replaced with their values, and data sets replaced with lists of their
    elements, all fetched with a single call `session.run()`.
  """
  py_typecheck.check_type(session, tf.Session)
  flattened_value = anonymous_tuple.flatten(value)

  # TODO(b/113123634): Add support for tf.data.Datasets.
  for v in flattened_value:
    if not tensor_util.is_tensor(v):
      raise ValueError('Unsupported value type {}.'.format(str(v)))

  flattened_results = session.run(flattened_value)
  return anonymous_tuple.pack_sequence_as(value, flattened_results)
