# Copyright 2021, The TensorFlow Federated Authors.
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
"""Runtime components for use by the XLA executor."""

from jax.lib.xla_bridge import xla_client
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.backends.xla import xla_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import typed_object


def normalize_tensor_representation(value, type_spec):
  """Normalizes the representation of a tensor `value` of a given `type_spec`.

  This converts vairous sorts of constants into a numpy representation.

  Args:
    value: The value whose representation to normalize.
    type_spec: An instance of `computation_types.TensorType` for this value.

  Returns:
    A normalized representation of `value` in numpy.

  Raises:
    TypeError: if the arguments are of the wrong types.
  """
  py_typecheck.check_type(type_spec, computation_types.TensorType)
  type_spec.shape.assert_is_fully_defined()
  type_analysis.check_type(value, type_spec)
  if type_spec.shape.rank == 0:
    return np.dtype(type_spec.dtype.as_numpy_dtype).type(value)
  if type_spec.shape.rank > 0:
    return np.array(value, dtype=type_spec.dtype.as_numpy_dtype)
  raise TypeError('Unsupported tensor shape {}.'.format(type_spec.shape))


def _binding_to_tensor_indexes(binding):
  """Returns the list of tensor indexes in `binding` in the flatten order.

  Args:
    binding: An XLA binding.

  Returns:
    A list of tensor indexes in it.

  Raises:
    ValueError: if the binding is of an unrecognized kind.
  """
  py_typecheck.check_type(binding, pb.Xla.Binding)
  kind = binding.WhichOneof('binding')
  if kind is None:
    return []
  if kind == 'tensor':
    return [binding.tensor.index]
  if kind == 'struct':
    tensor_indexes = []
    for element in binding.struct.element:
      tensor_indexes += _binding_to_tensor_indexes(element)
    return tensor_indexes
  raise ValueError('Unknown kind of binding {}.'.format(kind))


class ComputationCallable(typed_object.TypedObject):
  """An executor callable that encapsulates the logic of an XLA computation."""

  # TODO(b/175888145): It may be more efficient for internal representation of
  # any structured values to be a flat list to avoid the unnecessary overheads
  # of packing and unpacking during calls, etc. To discuss and institute this
  # in a follow-up, possibly applying the same ideas across all the leaf-level
  # executors.

  def __init__(self, comp_pb: pb.Computation,
               type_spec: computation_types.FunctionType,
               backend: xla_client.Client):
    """Creates this callable for a given computation, type, and backend.

    Args:
      comp_pb: An instance of `pb.Computation`.
      type_spec: An instance of `computation_types.FunctionType`.
      backend: An instance of `xla_client.Client`.

    Raises:
      ValueError: if the arguments are invalid.
    """
    py_typecheck.check_type(comp_pb, pb.Computation)
    py_typecheck.check_type(type_spec, computation_types.FunctionType)
    py_typecheck.check_type(backend, xla_client.Client)
    which_computation = comp_pb.WhichOneof('computation')
    if which_computation != 'xla':
      raise ValueError(
          'Unsupported computation type: {}'.format(which_computation))
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    compile_options = xla_client.CompileOptions()
    compile_options.parameter_is_tupled_arguments = True
    self._executable = backend.compile(xla_comp, compile_options)
    self._inverted_parameter_tensor_indexes = list(
        np.argsort(_binding_to_tensor_indexes(comp_pb.xla.parameter)))
    self._result_tensor_indexes = _binding_to_tensor_indexes(comp_pb.xla.result)
    self._type_signature = type_spec
    self._backend = backend

  @property
  def type_signature(self):
    return self._type_signature

  def __call__(self, *args, **kwargs):
    """Invokes this callable with the given set of arguments.

    Args:
      *args: Positional arguments.
      **kwargs: Keyword arguments.

    Returns:
      The result of the call.

    Raises:
      ValueError: if the arguments or the result are incorrect.
    """
    if kwargs:
      raise ValueError('Not expecting keyword arguments.')
    if len(args) > 1:
      raise ValueError('Not expecting more than one positional argument.')
    param_type = self.type_signature.parameter
    if param_type is None:
      if len(args) > 0:  # pylint: disable=g-explicit-length-test
        raise ValueError('Not expecting any arguments.')
      else:
        flat_py_args = []
    else:
      if len(args) == 0:  # pylint: disable=g-explicit-length-test
        raise ValueError('Positional argument missing.')
      positional_arg = args[0]
      if isinstance(param_type, computation_types.TensorType):
        flat_py_args = [positional_arg]
      else:
        py_typecheck.check_type(param_type, computation_types.StructType)
        py_typecheck.check_type(positional_arg, structure.Struct)
        flat_py_args = structure.flatten(positional_arg)

    reordered_flat_py_args = [
        flat_py_args[idx] for idx in self._inverted_parameter_tensor_indexes
    ]

    unordered_result = xla_client.execute_with_python_values(
        self._executable, reordered_flat_py_args, self._backend)
    py_typecheck.check_type(unordered_result, list)
    result = [unordered_result[idx] for idx in self._result_tensor_indexes]
    result_type = self.type_signature.result
    if isinstance(result_type, computation_types.TensorType):
      if len(result) != 1:
        raise ValueError('Expected one result, found {}.'.format(len(result)))
      return normalize_tensor_representation(result[0], result_type)
    else:
      py_typecheck.check_type(result_type, computation_types.StructType)
      return structure.pack_sequence_as(result_type, result)
