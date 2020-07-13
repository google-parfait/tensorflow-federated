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

import collections
import functools
import itertools
import typing

import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.utils import function_utils

TENSOR_REPRESENTATION_TYPES = (
    # Python native types
    str,
    int,
    float,
    bool,
    bytes,

    # Numpy data types
    np.generic,
    np.ndarray,
)


class UniqueNameFn(object):
  """Callable which generates names that are unique across calls."""

  # TODO(b/133329320): Perhaps use a more general solution here.

  def __init__(self):
    self._used_names = set()

  def __call__(self, s):
    s = s if s is not None else ''
    py_typecheck.check_type(s, str)
    next_name_int = 1
    proposal = s
    while proposal in self._used_names:
      proposal = '{}_{}'.format(s, next_name_int)
      next_name_int += 1
    self._used_names.add(proposal)
    return proposal


def get_tf_typespec_and_binding(parameter_type, arg_names, unpack=None):
  """Computes a `TensorSpec` input_signature and bindings for parameter_type.

  This is the TF2 analog to `stamp_parameter_in_graph`.

  Args:
    parameter_type: The TFF type of the input to a tensorflow function. Must be
      either an instance of computation_types.Type (or convertible to it), or
      None in the case of a no-arg function.
    arg_names: String names for any positional arguments to the tensorflow
      function.
    unpack: Whether or not to unpack parameter_type into args and kwargs. See
      e.g. `function_utils.pack_args_into_anonymous_tuple`.

  Returns:
    A tuple (args_typespec, kwargs_typespec, binding), where args_typespec is a
    list and kwargs_typespec is a dict, both containing `tf.TensorSpec`
    objects. These structures are intended to be passed to the
    `get_concrete_function` method of a `tf.function`.
    Note the "binding" is "preliminary" in that it includes the names embedded
    in the TensorSpecs produced; these must be converted to the names of actual
    tensors based on the SignatureDef of the SavedModel before the binding is
    finalized.
  """
  if parameter_type is None:
    return ([], {}, None)
  if unpack:
    arg_types, kwarg_types = function_utils.unpack_args_from_tuple(
        parameter_type)
    pack_in_tuple = True
  else:
    pack_in_tuple = False
    arg_types, kwarg_types = [parameter_type], {}

  py_typecheck.check_type(arg_names, collections.Iterable)
  if len(arg_names) < len(arg_types):
    raise ValueError(
        'If provided, arg_names must be a list of at least {} strings to '
        'match the number of positional arguments. Found: {}'.format(
            len(arg_types), arg_names))

  get_unique_name = UniqueNameFn()

  def _get_one_typespec_and_binding(parameter_name, parameter_type):
    """Returns a (tf.TensorSpec, binding) pair."""
    parameter_type = computation_types.to_type(parameter_type)
    if parameter_type.is_tensor():
      name = get_unique_name(parameter_name)
      tf_spec = tf.TensorSpec(
          shape=parameter_type.shape, dtype=parameter_type.dtype, name=name)
      binding = pb.TensorFlow.Binding(
          tensor=pb.TensorFlow.TensorBinding(tensor_name=name))
      return (tf_spec, binding)
    elif parameter_type.is_tuple():
      element_typespec_pairs = []
      element_bindings = []
      have_names = False
      have_nones = False
      for e_name, e_type in anonymous_tuple.iter_elements(parameter_type):
        if e_name is None:
          have_nones = True
        else:
          have_names = True
        name = '_'.join([n for n in [parameter_name, e_name] if n])
        e_typespec, e_binding = _get_one_typespec_and_binding(
            name if name else None, e_type)
        element_typespec_pairs.append((e_name, e_typespec))
        element_bindings.append(e_binding)
      # For a given argument or kwarg, we shouldn't have both:
      if (have_names and have_nones):
        raise ValueError(
            'A mix of named and unnamed entries are not supported inside a '
            'nested structure representing a single argument in a call to a '
            'TensorFlow or Python function.\n{}'.format(parameter_type))
      tf_typespec = anonymous_tuple.AnonymousTuple(element_typespec_pairs)
      return (tf_typespec,
              pb.TensorFlow.Binding(
                  tuple=pb.TensorFlow.NamedTupleBinding(
                      element=element_bindings)))
    elif parameter_type.is_sequence():
      raise NotImplementedError('Sequence iputs not yet supported for TF 2.0.')
    else:
      raise ValueError(
          'Parameter type component {!r} cannot be converted to a TensorSpec'
          .format(parameter_type))

  def get_arg_name(i):
    name = arg_names[i]
    if not isinstance(name, str):
      raise ValueError('arg_names must be strings, but got: {}'.format(name))
    return name

  # Main logic --- process arg_types and kwarg_types:
  arg_typespecs = []
  kwarg_typespecs = {}
  bindings = []
  for i, arg_type in enumerate(arg_types):
    name = get_arg_name(i)
    typespec, binding = _get_one_typespec_and_binding(name, arg_type)
    typespec = type_conversions.type_to_py_container(typespec, arg_type)
    arg_typespecs.append(typespec)
    bindings.append(binding)
  for name, kwarg_type in kwarg_types.items():
    typespec, binding = _get_one_typespec_and_binding(name, kwarg_type)
    typespec = type_conversions.type_to_py_container(typespec, kwarg_type)
    kwarg_typespecs[name] = typespec
    bindings.append(binding)

  assert bindings, 'Given parameter_type {}, but produced no bindings.'.format(
      parameter_type)
  if pack_in_tuple:
    final_binding = pb.TensorFlow.Binding(
        tuple=pb.TensorFlow.NamedTupleBinding(element=bindings))
  else:
    final_binding = bindings[0]

  return (arg_typespecs, kwarg_typespecs, final_binding)


def get_tf2_result_dict_and_binding(result, name_prefix=None):
  """Returns the result_dict, type, and bindings for this result.

  This is the TF2 equivalent of `capture_result_from_graph`. This function:
    - flattens result into a dict, so that it can be returned from
      a tf.function that is serialized with `tf.saved_model.save` (which
      does not support returning nested structures).
    - Calculates a `tff.Type` for the original result, and bindings
      from the result_dict to that type.

  Args:
    result: The result returned by a Python TensorFlow function (possibly a
      tf.function), which will be flattened as a dict to allow it to be returned
      from a tf.function which is to be serialized. Thus, `result` should be a
      Python object that is composed of tensors, possibly nested within Python
      structures such as dictionaries, lists, tuples, or named tuples.
    name_prefix: Prefix for names to be added to result_dict.

  Returns:
    A tuple (result_dict, result_type, binding), where
      result_dict: A `dict` containing the same Tensors as result.
      result_type: The TFF type of `result`.
      binding: A `pb.Binding` for mapping result_dict to the result_type. This
        binding is "preliminary" in the sense that the `tensor_name`s of
        the leaf `TensorBinding`s are populated with the corresponding keys
        from the `result_dict`. These bindings are finalized later (see
        `finalize_binding` in `tensorflow_serialization.py`).

  Raises:
    TypeError: if the argument or any of its parts are of an uexpected type.
  """
  result_dict = collections.OrderedDict()
  get_unique_name = UniqueNameFn()

  def get_name(s, name_prefix=''):
    return get_unique_name(
        s if not name_prefix else '{}_{}'.format(name_prefix, s))

  def _handle_elements(name_value_pairs, name_prefix, container_type=None):
    """Returns (result_type, partial_binding) for the name_value_pairs."""
    ntt_pairs = []
    partial_bindings = []
    for name, value in name_value_pairs:
      result_type, output_binding = _get_result_type_and_binding_impl(
          value, get_name(name, name_prefix))
      partial_bindings.append(output_binding)
      ntt_pairs.append((name, result_type))

    if container_type:
      tff_type = computation_types.NamedTupleTypeWithPyContainerType(
          ntt_pairs, container_type=container_type)
    else:
      tff_type = computation_types.NamedTupleType(ntt_pairs)

    return (tff_type,
            pb.TensorFlow.Binding(
                tuple=pb.TensorFlow.NamedTupleBinding(
                    element=partial_bindings)))

  # Recusrive helper function.
  def _get_result_type_and_binding_impl(result, name_prefix):
    """Recurse, update `result_dict`, return (result_type, partial_binding)."""
    if tf.is_tensor(result):
      output_name = get_name(result.name)
      assert output_name not in result_dict
      result_dict[output_name] = result
      return (computation_types.TensorType(result.dtype.base_dtype,
                                           result.shape),
              pb.TensorFlow.Binding(
                  tensor=pb.TensorFlow.TensorBinding(tensor_name=output_name)))
    elif py_typecheck.is_named_tuple(result):
      name_value_pairs = result._asdict().items()  # pylint: disable=protected-access
      return _handle_elements(
          name_value_pairs, name_prefix, container_type=type(result))
    elif isinstance(result, anonymous_tuple.AnonymousTuple):
      return _handle_elements(anonymous_tuple.to_elements(result), name_prefix)
    elif isinstance(result, collections.Mapping):
      if isinstance(result, collections.OrderedDict):
        name_value_pairs = result.items()
      else:
        name_value_pairs = sorted(result.items())
      return _handle_elements(
          name_value_pairs, name_prefix, container_type=type(result))
    elif isinstance(result, (list, tuple)):
      name_value_pairs = [(None, v) for i, v in enumerate(result)]
      return _handle_elements(
          name_value_pairs, name_prefix, container_type=type(result))
    elif isinstance(result, tf.data.Dataset):
      # TODO(b/122081673): Support Datasets here.
      raise NotImplementedError(
          'Datasets not currently supported for TF 2 serialization.')
    else:
      raise TypeError(
          'Cannot capture a result of an unsupported type {}.'.format(
              py_typecheck.type_string(type(result))))

  result_type, binding = _get_result_type_and_binding_impl(result, name_prefix)
  return result_dict, result_type, binding


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
    a placeholder, or a `anonymous_tuple.AnonymousTuple` that represents a named
    tuple) that represents the stamped parameter for use in the body of a Python
    function that consumes this parameter, and the 'binding' is an instance of
    TensorFlow.Binding that indicates how parts of the type signature relate
    to the tensors and ops stamped into the graph.

  Raises:
    TypeError: If the arguments are of the wrong computation_types.
    ValueError: If the parameter type cannot be stamped in a TensorFlow graph.
  """
  py_typecheck.check_type(parameter_name, str)
  py_typecheck.check_type(graph, tf.Graph)
  if parameter_type is None:
    return (None, None)
  parameter_type = computation_types.to_type(parameter_type)
  if parameter_type.is_tensor():
    with graph.as_default():
      placeholder = tf.compat.v1.placeholder(
          dtype=parameter_type.dtype,
          shape=parameter_type.shape,
          name=parameter_name)
      binding = pb.TensorFlow.Binding(
          tensor=pb.TensorFlow.TensorBinding(tensor_name=placeholder.name))
      return (placeholder, binding)
  elif parameter_type.is_tuple():
    # The parameter_type could be a NamedTupleTypeWithPyContainer, however, we
    # ignore that for now. Instead, the proper containers will be inserted at
    # call time by function_utils.wrap_as_zero_or_one_arg_callable.
    if not parameter_type:
      # Stamps dummy element to "populate" graph, as TensorFlow does not support
      # empty graphs.
      dummy_tensor = tf.no_op()
    element_name_value_pairs = []
    element_bindings = []
    for e in anonymous_tuple.iter_elements(parameter_type):
      e_val, e_binding = stamp_parameter_in_graph(
          '{}_{}'.format(parameter_name, e[0]), e[1], graph)
      element_name_value_pairs.append((e[0], e_val))
      element_bindings.append(e_binding)
    return (anonymous_tuple.AnonymousTuple(element_name_value_pairs),
            pb.TensorFlow.Binding(
                tuple=pb.TensorFlow.NamedTupleBinding(
                    element=element_bindings)))
  elif parameter_type.is_sequence():
    with graph.as_default():
      variant_tensor = tf.compat.v1.placeholder(tf.variant, shape=[])
      ds = make_dataset_from_variant_tensor(variant_tensor,
                                            parameter_type.element)
    return (ds,
            pb.TensorFlow.Binding(
                sequence=pb.TensorFlow.SequenceBinding(
                    variant_tensor_name=variant_tensor.name)))
  else:
    raise ValueError(
        'Parameter type component {!r} cannot be stamped into a TensorFlow '
        'graph.'.format(parameter_type))


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
            variant_tensor.dtype))
  return tf.data.experimental.from_variant(
      variant_tensor,
      structure=(type_conversions.type_to_tf_structure(
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
  if isinstance(result, TENSOR_REPRESENTATION_TYPES):
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
    name_value_pairs = result._asdict().items()
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
        name_value_pairs.items(), graph,
        functools.partial(
            computation_types.NamedTupleTypeWithPyContainerType,
            container_type=type(result)))
  elif isinstance(result, anonymous_tuple.AnonymousTuple):
    return _get_bindings_for_elements(
        anonymous_tuple.to_elements(result), graph,
        computation_types.NamedTupleType)
  elif isinstance(result, collections.Mapping):
    if isinstance(result, collections.OrderedDict):
      name_value_pairs = result.items()
    else:
      name_value_pairs = sorted(result.items())
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
  elif isinstance(result, type_conversions.TF_DATASET_REPRESENTATION_TYPES):
    variant_tensor = tf.data.experimental.to_variant(result)
    element_structure = result.element_spec
    try:
      element_type = computation_types.to_type(element_structure)
    except TypeError as e:
      raise TypeError(
          'TFF does not support Datasets that yield elements of structure {!s}'
          .format(element_structure)) from e
    return (computation_types.SequenceType(element_type),
            pb.TensorFlow.Binding(
                sequence=pb.TensorFlow.SequenceBinding(
                    variant_tensor_name=variant_tensor.name)))
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
    if sequence_oneof == 'variant_tensor_name':
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
    if sequence_oneof == 'variant_tensor_name':
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
  for k, v in output_map.items():
    py_typecheck.check_type(k, str)
    if not tf.is_tensor(v):
      raise TypeError(
          'Element with key {} in the output map is {}, not a tensor.'.format(
              k, py_typecheck.type_string(type(v))))

  binding_oneof = binding.WhichOneof('binding')
  if type_spec.is_tensor():
    if binding_oneof != 'tensor':
      raise ValueError(
          'Expected a tensor binding, found {}.'.format(binding_oneof))
    elif binding.tensor.tensor_name not in output_map:
      raise ValueError('Tensor named {} not found in the output map.'.format(
          binding.tensor.tensor_name))
    else:
      return output_map[binding.tensor.tensor_name]
  elif type_spec.is_tuple():
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
      if not type_spec.is_tuple_with_py_container():
        return anonymous_tuple.AnonymousTuple(result_elements)
      container_type = computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
          type_spec)
      if (py_typecheck.is_named_tuple(container_type) or
          py_typecheck.is_attrs(container_type)):
        return container_type(**dict(result_elements))
      return container_type(result_elements)
  elif type_spec.is_sequence():
    if binding_oneof != 'sequence':
      raise ValueError(
          'Expected a sequence binding, found {}.'.format(binding_oneof))
    else:
      sequence_oneof = binding.sequence.WhichOneof('binding')
      if sequence_oneof == 'variant_tensor_name':
        variant_tensor = output_map[binding.sequence.variant_tensor_name]
        return make_dataset_from_variant_tensor(variant_tensor,
                                                type_spec.element)
      else:
        raise ValueError(
            'Unsupported sequence binding \'{}\'.'.format(sequence_oneof))
  else:
    raise ValueError('Unsupported type \'{}\'.'.format(type_spec))


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
  if type_spec.is_tensor():
    return []
  elif type_spec.is_tuple():
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
          'Expected a named tuple type with either all elements named or all '
          'unnamed, got {}.'.format(type_spec))
  else:
    raise TypeError(
        'Expected a tensor or named tuple type, found {}.'.format(type_spec))


def make_dummy_element_for_type_spec(type_spec, none_dim_replacement=0):
  """Creates ndarray of zeros corresponding to `type_spec`.

  Returns a list containing this ndarray, whose type is *compatible* with, not
  necessarily equal to, `type_spec`. This is due to the fact that some
  dimensions of `type_spec` may be indeterminate, representing compatibility
  of `type_spec` with any number (e.g. leaving a batch dimension indeterminate
  to signify compatibility with batches of any size). However a concrete
  structure (like the ndarray) must have specified sizes for its dimensions.
  So we construct a dummy element where any `None` dimensions of the shape
  of `type_spec` are replaced with the value `none_dim_replacement`.The
  default value of 0 therefore returns a dummy element of minimal size which
  matches `type_spec`.

  Args:
    type_spec: Instance of `computation_types.Type`, or something convertible to
      one by `computation_types.to_type`.
    none_dim_replacement: `int` with which to replace any unspecified tensor
      dimensions.

  Returns:
    Returns possibly nested `numpy ndarray`s containing all zeros: a single
    `ndarray` if `type_spec` is a `computation_types.TensorType` and a list
    of such arrays if  `type_spec` is `computation_types.NamedTupleType`.
    This data structure is of the minimal size necessary in order to be
    compatible with `type_spec`.
  """
  type_spec = computation_types.to_type(type_spec)
  if not type_analysis.contains_only(type_spec,
                                     lambda t: t.is_tuple() or t.is_tensor()):
    raise ValueError('Cannot construct array for TFF type containing anything '
                     'other than `computation_types.TensorType` or '
                     '`computation_types.NamedTupleType`; you have passed the '
                     'type {}'.format(type_spec))
  py_typecheck.check_type(none_dim_replacement, int)
  if none_dim_replacement < 0:
    raise ValueError('Please pass nonnegative integer argument as '
                     '`none_dim_replacement`.')

  def _handle_none_dimension(x):
    if x is None or (isinstance(x, tf.compat.v1.Dimension) and x.value is None):
      return none_dim_replacement
    return x

  if type_spec.is_tensor():
    dummy_shape = [_handle_none_dimension(x) for x in type_spec.shape]
    if type_spec.dtype == tf.string:
      return np.empty(dummy_shape, dtype=str)
    return np.zeros(dummy_shape, type_spec.dtype.as_numpy_dtype)
  elif type_spec.is_tuple():
    elements = anonymous_tuple.to_elements(type_spec)
    elem_list = []
    for _, elem_type in elements:
      elem_list.append(make_dummy_element_for_type_spec(elem_type))
    return elem_list


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
          'Expected an anonymous tuple to either have all elements named or '
          'all unnamed, got {}.'.format(value))
  if type_spec.is_tensor():
    py_typecheck.check_type(structure, list)
    # Convert the structure members to tensors to ensure that they are properly
    # typed and grouped before being passed to
    # tf.data.Dataset.from_tensor_slices.
    structure.append(tf.convert_to_tensor(value, type_spec.dtype))  # pytype: disable=attribute-error
  elif type_spec.is_tuple():
    elements = anonymous_tuple.to_elements(type_spec)
    if isinstance(structure, collections.OrderedDict):
      if py_typecheck.is_named_tuple(value):
        value = value._asdict()  # pytype: disable=attribute-error
      if isinstance(value, dict):
        if set(value.keys()) != set(k for k, _ in elements):
          raise TypeError('Value {} does not match type {}.'.format(
              value, type_spec))
        for elem_name, elem_type in elements:
          append_to_list_structure_for_element_type_spec(
              structure[elem_name], value[elem_name], elem_type)
      elif isinstance(value, (list, tuple)):
        if len(value) != len(elements):
          raise TypeError('Value {} does not match type {}.'.format(
              value, type_spec))
        for idx, (elem_name, elem_type) in enumerate(elements):
          append_to_list_structure_for_element_type_spec(
              structure[elem_name], value[idx], elem_type)
      else:
        raise TypeError('Unexpected type of value {} for TFF type {}.'.format(
            py_typecheck.type_string(type(value)), type_spec))
    elif isinstance(structure, tuple):
      py_typecheck.check_type(value, (list, tuple))
      if len(value) != len(elements):
        raise TypeError('Value {} does not match type {}.'.format(
            value, type_spec))
      for idx, (_, elem_type) in enumerate(elements):
        append_to_list_structure_for_element_type_spec(structure[idx],
                                                       value[idx], elem_type)
    else:
      raise TypeError(
          'Invalid nested structure, unexpected container type {}.'.format(
              py_typecheck.type_string(type(structure))))
  else:
    raise TypeError(
        'Expected a tensor or named tuple type, found {}.'.format(type_spec))


def replace_empty_leaf_lists_with_numpy_arrays(structure, type_spec):
  """Replaces empty leaf lists in `structure` with numpy arrays.

  This function is primarily used to ensure that an appropriate TF dtype is
  inferrable for a structure, even if no elements are actually present.

  Args:
    structure: Output of `make_empty_list_structure_for_element_type_spec`.
    type_spec: An instance of `tff.Type` or something convertible to it, as in
      `make_empty_list_structure_for_element_type_spec`.

  Returns:
    The transformed version of `structure`.

  Raises:
    TypeError: If the `type_spec` is not of a form described above, or if
      `structure` is not of a type compatible with `type_spec`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if type_spec.is_tensor():
    py_typecheck.check_type(structure, list)
    if len(structure) > 0:  # pylint: disable=g-explicit-length-test
      return structure
    else:
      return np.array([], dtype=type_spec.dtype.as_numpy_dtype)
  elif type_spec.is_tuple():
    elements = anonymous_tuple.to_elements(type_spec)
    if isinstance(structure, collections.OrderedDict):
      to_return = []
      for elem_name, elem_type in elements:
        elem_val = replace_empty_leaf_lists_with_numpy_arrays(
            structure[elem_name], elem_type)
        to_return.append((elem_name, elem_val))
      return collections.OrderedDict(to_return)
    elif isinstance(structure, tuple):
      to_return = []
      for idx, (_, elem_type) in enumerate(elements):
        elem_val = replace_empty_leaf_lists_with_numpy_arrays(
            structure[idx], elem_type)
        to_return.append(elem_val)
      return tuple(to_return)
    else:
      raise TypeError(
          'Invalid nested structure, unexpected container type {}.'.format(
              py_typecheck.type_string(type(structure))))
  else:
    raise TypeError(
        'Expected a tensor or named tuple type, found {}.'.format(type_spec))


def make_data_set_from_elements(graph, elements, element_type):
  """Creates a `tf.data.Dataset` in `graph` from explicitly listed `elements`.

  Note: The underlying implementation attempts to use the
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
    graph: The graph in which to construct the `tf.data.Dataset`, or `None` if
      the construction is to happen in the eager context.
    elements: A list of elements.
    element_type: The type of elements.

  Returns:
    The constructed `tf.data.Dataset` instance.

  Raises:
    TypeError: If element types do not match `element_type`.
    ValueError: If the elements are of incompatible types and shapes, or if
      no graph was specified outside of the eager context.
  """
  # Note: We allow the graph to be `None` to allow this function to be used in
  # the eager context.
  if graph is not None:
    py_typecheck.check_type(graph, tf.Graph)
  elif not tf.executing_eagerly():
    raise ValueError('Only in eager context may the graph be `None`.')
  py_typecheck.check_type(elements, list)
  element_type = computation_types.to_type(element_type)
  py_typecheck.check_type(element_type, computation_types.Type)

  def _make(element_subset):
    structure = make_empty_list_structure_for_element_type_spec(element_type)
    for el in element_subset:
      append_to_list_structure_for_element_type_spec(structure, el,
                                                     element_type)
    tensor_slices = replace_empty_leaf_lists_with_numpy_arrays(
        structure, element_type)
    return tf.data.Dataset.from_tensor_slices(tensor_slices)

  def _work():  # pylint: disable=missing-docstring
    if not elements:
      # Just return an empty data set with the appropriate types.
      dummy_element = make_dummy_element_for_type_spec(element_type)
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
        #
        # Note: this will not remain `None` because `element`s is not empty.
        ds = None
        ds = typing.cast(tf.data.Dataset, ds)
        for i in range(len(elements)):
          singleton_ds = _make(elements[i:i + 1])
          ds = singleton_ds if ds is None else ds.concatenate(singleton_ds)
    ds_element_type = computation_types.to_type(ds.element_spec)
    if not element_type.is_assignable_from(ds_element_type):
      raise TypeError(
          'Failure during data set construction, expected elements of type {}, '
          'but the constructed data set has elements of type {}.'.format(
              element_type, ds_element_type))
    return ds

  if graph is not None:
    with graph.as_default():
      return _work()
  else:
    return _work()


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
  py_typecheck.check_type(sess, tf.compat.v1.Session)
  # TODO(b/113123634): Investigate handling `list`s and `tuple`s of
  # `tf.data.Dataset`s and what the API would look like to support this.
  if isinstance(value, type_conversions.TF_DATASET_REPRESENTATION_TYPES):
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
    dataset_results = {}
    flat_tensors = []
    for idx, v in enumerate(flattened_value):
      if isinstance(v, type_conversions.TF_DATASET_REPRESENTATION_TYPES):
        dataset_tensors = fetch_value_in_session(sess, v)
        if not dataset_tensors:
          # An empty list has been returned; we must pack the shape information
          # back in or the result won't typecheck.
          element_structure = v.element_spec
          dummy_elem = make_dummy_element_for_type_spec(element_structure)
          dataset_tensors = [dummy_elem]
        dataset_results[idx] = dataset_tensors
      elif tf.is_tensor(v):
        flat_tensors.append(v)
      else:
        raise ValueError('Unsupported value type {}.'.format(v))
    # Note that `flat_tensors` could be an empty tuple, but it could also be a
    # list of empty tuples.
    if flat_tensors or any(x for x in flat_tensors):
      flat_computed_tensors = sess.run(flat_tensors)
    else:
      flat_computed_tensors = flat_tensors
    flattened_results = _interleave_dataset_results_and_tensors(
        dataset_results, flat_computed_tensors)

    def _to_unicode(v):
      if isinstance(v, bytes):
        return v.decode('utf-8')
      return v

    if tf.is_tensor(value) and value.dtype == tf.string:
      flattened_results = [_to_unicode(result) for result in flattened_results]
    return anonymous_tuple.pack_sequence_as(value, flattened_results)


def _interleave_dataset_results_and_tensors(dataset_results, flat_run_tensors):
  flattened_results = []
  for idx in range(len(dataset_results) + len(flat_run_tensors)):
    if dataset_results.get(idx):
      flattened_results.append(dataset_results[idx])
    else:
      flattened_results.append(flat_run_tensors.pop(0))
  return flattened_results


def to_node_name(name):
  """Returns the name of a node in `graph_def` that `name` refers to.

  Args:
    name: A string.

  Returns:
    A stripped version of `name` without control dependency prefix or output
    suffix.

  Raises:
    ValueError: If `name` is not a valid name of a node or node input.
  """
  py_typecheck.check_type(name, str)
  if not name:
    raise ValueError('The argument cannot be empty.')
  if name[0] == '^':
    name = name[1:]
  colon = name.rfind(':')
  if colon >= 0:
    return name[:colon]
  else:
    return name


def get_deps_for_graph_node(graph_def, node_name):
  """Returns the set of node names that a node named `node_name` depends on.

  Args:
    graph_def: The input graph, an instance of `tf.compat.v1.GraphDef`.
    node_name: The node name, a string.

  Returns:
    An instance of `set()` containing string names of the nodes `node_name`
    depends on in `graph_def`.
  """
  py_typecheck.check_type(graph_def, tf.compat.v1.GraphDef)
  py_typecheck.check_type(node_name, str)
  input_map = {}
  for node in graph_def.node:
    input_map[node.name] = set(to_node_name(x) for x in node.input)
  dependencies = set()
  initial_singleton = set([node_name])
  nodes_to_process = initial_singleton
  while nodes_to_process:
    dependencies.update(nodes_to_process)
    nodes_to_process = set.union(
        *[input_map[name]
          for name in nodes_to_process]).difference(dependencies)
  return dependencies.difference(initial_singleton)


def add_control_deps_for_init_op(graph_def, init_op):
  """Adds control deps on `init_op` to `graph_def`.

  Args:
    graph_def: The input graph, an instance of `tf.compat.v1.GraphDef`.
    init_op: The init op name, a string.

  Returns:
    The updated graph, an instance of `tf.compat.v1.GraphDef`.
  """
  py_typecheck.check_type(graph_def, tf.compat.v1.GraphDef)
  py_typecheck.check_type(init_op, str)
  init_op_str = to_node_name(init_op)
  init_op_control_dep = '^{}'.format(init_op_str)
  deps = get_deps_for_graph_node(graph_def,
                                 init_op_str).union(set([init_op_str]))
  new_graph_def = tf.compat.v1.GraphDef()
  new_graph_def.CopyFrom(graph_def)
  for new_node in new_graph_def.node:
    if new_node.name not in deps:
      node_inputs = set(new_node.input)
      if init_op_control_dep not in node_inputs:
        new_node.input.extend([init_op_control_dep])
  return new_graph_def


def coerce_dataset_elements_to_tff_type_spec(dataset, element_type):
  """Map the elements of a dataset to a specified type.

  This is used to coerce a `tf.data.Dataset` that may have lost the ordering
  of dictionary keys back into a `collections.OrderedDict` (required by TFF).

  Args:
    dataset: a `tf.data.Dataset` instance.
    element_type: a `tff.Type` specifying the type of the elements of `dataset`.
      Must be a `tff.TensorType` or `tff.NamedTupleType`.

  Returns:
    A `tf.data.Dataset` whose output types are compatible with
    `element_type`.

  Raises:
    ValueError: if the elements of `dataset` cannot be coerced into
      `element_type`.
  """
  py_typecheck.check_type(dataset,
                          type_conversions.TF_DATASET_REPRESENTATION_TYPES)
  py_typecheck.check_type(element_type, computation_types.Type)

  if element_type.is_tensor():
    return dataset

  # This is a similar to `reference_executor.to_representation_for_type`,
  # look for opportunities to consolidate?
  def _to_representative_value(type_spec, elements):
    """Convert to a container to a type understood by TF and TFF."""
    if type_spec.is_tensor():
      return elements
    elif type_spec.is_tuple():
      field_types = anonymous_tuple.to_elements(type_spec)
      is_all_named = all([name is not None for name, _ in field_types])
      if is_all_named:
        if py_typecheck.is_named_tuple(elements):
          values = collections.OrderedDict(
              (name, _to_representative_value(field_type, e))
              for (name, field_type), e in zip(field_types, elements))
          return type(elements)(**values)
        else:
          values = [(name, _to_representative_value(field_type, elements[name]))
                    for name, field_type in field_types]
          return collections.OrderedDict(values)
      else:
        return tuple(
            _to_representative_value(t, e) for t, e in zip(type_spec, elements))
    else:
      raise ValueError(
          'Coercing a dataset with elements of expected type {!s}, '
          'produced a value with incompatible type `{!s}. Value: '
          '{!s}'.format(type_spec, type(elements), elements))

  # tf.data.Dataset of tuples will unwrap the tuple in the `map()` call, so we
  # must pass a function taking *args. However, if the call was originally only
  # a single tuple, it is now "double wrapped" and must be unwrapped before
  # traversing.
  def _unwrap_args(*args):
    if len(args) == 1:
      return _to_representative_value(element_type, args[0])
    else:
      return _to_representative_value(element_type, args)

  return dataset.map(_unwrap_args)
