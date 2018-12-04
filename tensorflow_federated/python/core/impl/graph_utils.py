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

# Dependency imports
import six
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb

from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import anonymous_tuple
from tensorflow_federated.python.core.impl import type_utils


def stamp_parameter_in_graph(parameter_name, parameter_type, graph=None):
  """Stamps a parameter of a given type in the given tf.Graph instance.

  Tensors are stamped as placeholders, sequences are stamped as data sets
  constructed from string tensor handles, and named tuples are stamped by
  independently stamping their elements.

  Args:
    parameter_name: The suggested (string) name of the parameter to use in
      determining the names of the graph components to construct. The names
      that will actually appear in the graph are not guaranteed to be based
      on this suggested name, and may vary, e.g., due to existing naming
      conflicts, but a best-effort attempt will be made to make them similar
      for ease of debugging.
    parameter_type: The type of the parameter to stamp. Must be either an
      instance of types.Type (or convertible to it), or None.
    graph: The optional instance of tf.Graph to stamp in. If not specified,
      stamping is done in the default graph.

  Returns:
    A tuple (val, binding), where 'val' is a Python object (such as a dataset,
    a placeholder, or a dictionary that represents a named tuple) that
    represents the stamped parameter for use in the body of a Python function
    that consumes this parameter, and the 'binding' is an instance of
    TensorFlow.Binding that indicates how parts of the type signature relate
    to the tensors and ops stamped into the graph.

  Raises:
    TypeError: if the arguments are of the wrong types.
    ValueError: if the parameter type cannot be stamped in a TensorFlow graph.
  """
  py_typecheck.check_type(parameter_name, six.string_types)
  if graph is None:
    graph = tf.get_default_graph()
  else:
    py_typecheck.check_type(graph, tf.Graph)
  if parameter_type is None:
    return (None, None)
  parameter_type = types.to_type(parameter_type)
  if isinstance(parameter_type, types.TensorType):
    with graph.as_default():
      placeholder = tf.placeholder(
          dtype=parameter_type.dtype,
          shape=parameter_type.shape,
          name=parameter_name)
      binding = pb.TensorFlow.Binding(
          tensor=pb.TensorFlow.TensorBinding(tensor_name=placeholder.name))
      return (placeholder, binding)
  elif isinstance(parameter_type, types.NamedTupleType):
    element_name_value_pairs = []
    element_bindings = []
    for e in parameter_type.elements:
      e_val, e_binding = stamp_parameter_in_graph(
          '{}_{}'.format(parameter_name, e[0]), e[1], graph)
      element_name_value_pairs.append((e[0], e_val))
      element_bindings.append(e_binding)
    return (anonymous_tuple.AnonymousTuple(element_name_value_pairs),
            pb.TensorFlow.Binding(tuple=pb.TensorFlow.NamedTupleBinding(
                element=element_bindings)))
  elif isinstance(parameter_type, types.SequenceType):
    dtypes, shapes = (
        type_utils.type_to_tf_dtypes_and_shapes(parameter_type.element))
    with graph.as_default():
      handle = tf.placeholder(tf.string, shape=[])
      it = tf.data.Iterator.from_string_handle(handle, dtypes, shapes)
      # In order to convert an iterator into something that looks like a data
      # set, we create a dummy data set that consists of an infinite sequence
      # of zeroes, and filter it through a map function that invokes
      # 'it.get_next()' for each of those zeroes.
      # TODO(b/113112108): Possibly replace this with something more canonical
      # if and when we can find adequate support for abstractly defined data
      # sets (at the moment of this writing it does not appear to exist yet).
      ds = tf.data.Dataset.from_tensors(0).repeat().map(lambda _: it.get_next())
      return (ds, pb.TensorFlow.Binding(sequence=pb.TensorFlow.SequenceBinding(
          iterator_string_handle_name=handle.name)))
  else:
    raise ValueError(
        'Parameter type component {} cannot be stamped into a TensorFlow '
        'graph.'.format(repr(parameter_type)))


def capture_result_from_graph(result):
  """Captures a result stamped into a tf.Graph as a type signature and binding.

  Args:
    result: The result to capture, a Python object that is composed of tensors,
      possibly nested within Python structures such as dictionaries, lists,
      tuples, or named tuples.

  Returns:
    A tuple (type_spec, binding), where 'type_spec' is an instance of types.Type
    that describes the type of the result, and 'binding'is an instance of
    TensorFlow.Binding that indicates how parts of the result type relate to the
    tensors and ops that appear in the result.

  Raises:
    TypeError: if the argument or any of its parts are of an uexpected type.
  """
  # TODO(b/113112885): The emerging extensions for serializing SavedModels may
  # end up introducing similar concepts of bindings, etc., we should look here
  # into the possibility of reusing some of that code when it's available.
  if tf.contrib.framework.is_tensor(result):
    return (types.TensorType(result.dtype, result.shape),
            pb.TensorFlow.Binding(tensor=pb.TensorFlow.TensorBinding(
                tensor_name=result.name)))
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
    name_value_pairs = (
        anonymous_tuple.to_elements(result) if isinstance(
            result, anonymous_tuple.AnonymousTuple) else six.iteritems(result))
    element_name_type_binding_triples = [
        ((k,) + capture_result_from_graph(v)) for k, v in name_value_pairs]
    return (types.NamedTupleType([((e[0], e[1]) if e[0] else e[1])
                                  for e in element_name_type_binding_triples]),
            pb.TensorFlow.Binding(tuple=pb.TensorFlow.NamedTupleBinding(
                element=[e[2] for e in element_name_type_binding_triples])))
  elif isinstance(result, (list, tuple)):
    element_type_binding_pairs = [capture_result_from_graph(e) for e in result]
    return (types.NamedTupleType([e[0] for e in element_type_binding_pairs]),
            pb.TensorFlow.Binding(tuple=pb.TensorFlow.NamedTupleBinding(
                element=[e[1] for e in element_type_binding_pairs])))
  else:
    raise TypeError('Cannot capture a result of an unsupported type {}.'.format(
        py_typecheck.type_string(type(result))))
