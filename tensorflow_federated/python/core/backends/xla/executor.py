# Copyright 2020, The TensorFlow Federated Authors.
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
"""An executor that delegates to the XLA compiler."""

from jax.lib.xla_bridge import xla_client

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.backends.xla import runtime
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import typed_object


def to_representation_for_type(value, type_spec, backend=None):
  """Verifies or converts the `value` to executor payload matching `type_spec`.

  The following kinds of `value` are supported:

  * Computations, either `pb.Computation` or `computation_impl.ComputationImpl`.

  * Numpy arrays and scalars, or Python scalars that are converted to Numpy.

  * Nested structures of the above.

  Args:
    value: The raw representation of a value to compare against `type_spec` and
      potentially to be converted.
    type_spec: An instance of `tff.Type`. Can be `None` for values that derive
      from `typed_object.TypedObject`.
    backend: The backend to use; an instance of `xla_client.Client`. Only used
      for functional types. Can be `None` if unused.

  Returns:
    Either `value` itself, or a modified version of it.

  Raises:
    TypeError: If the `value` is not compatible with `type_spec`.
    ValueError: If the arguments are incorrect.
  """
  if backend is not None:
    py_typecheck.check_type(backend, xla_client.Client)
  if type_spec is not None:
    type_spec = computation_types.to_type(type_spec)
  type_spec = executor_utils.reconcile_value_with_type_spec(value, type_spec)
  if isinstance(value, computation_base.Computation):
    return to_representation_for_type(
        computation_impl.ComputationImpl.get_proto(value), type_spec, backend)
  if isinstance(value, pb.Computation):
    comp_type = type_serialization.deserialize_type(value.type)
    if type_spec is not None:
      comp_type.check_equivalent_to(type_spec)
    return runtime.ComputationCallable(value, comp_type, backend)
  if isinstance(type_spec, computation_types.StructType):
    return structure.map_structure(
        lambda v, t: to_representation_for_type(v, t, backend),
        structure.from_container(value, recursive=True), type_spec)
  if isinstance(type_spec, computation_types.TensorType):
    return runtime.normalize_tensor_representation(value, type_spec)
  raise TypeError('Unexpected type {}.'.format(type_spec))


class XlaValue(executor_value_base.ExecutorValue):
  """A representation of a value managed by the XLA executor."""

  def __init__(self, value, type_spec, backend):
    """Creates an instance of a value in this executor.

    Args:
      value: Same as in `to_representation_for_type()`.
      type_spec: Same as in `to_representation_for_type()`.
      backend: Same as in `to_representation_for_type()`.
    """
    if type_spec is None:
      py_typecheck.check_type(value, typed_object.TypedObject)
      type_spec = value.type_signature
    else:
      type_spec = computation_types.to_type(type_spec)
    self._type_signature = type_spec
    self._value = to_representation_for_type(value, type_spec, backend)

  @property
  def internal_representation(self):
    """Returns internal representation of the value embedded in the executor."""
    return self._value

  @property
  def type_signature(self):
    return self._type_signature

  @tracing.trace
  async def compute(self):
    py_typecheck.check_type(
        self._type_signature,
        (computation_types.StructType, computation_types.TensorType))
    return self._value


class XlaExecutor(executor_base.Executor):
  """The XLA executor delegates execution to the XLA compiler.

  NOTE: This executor is in the process of gewtting developed. Most capabilities
  are not implemented, and those that are, only work in experimental mode. All
  aspects of this executor are subject to change.

  Capabilities currently supported in experimental-only mode:

  * Creating and invoking XLA computations with inputs and outputs composed of
    tensors or (possibly recursively) nested structures of tensors.

  Notable capabilities currently missing:

  * Support for sequence types.

  This executor is designed as a drop-in replacement for the eager TF executor.
  It uses the XLA compiler instead an eager TensorFlow runtime. At this point,
  the use of this executor is limited to incubation and testing.
  """

  # TODO(b/175888145): Reach full functional parity with the eager TF executor.

  # TODO(b/175888145): Add support for explicitly specifying devices.

  # TODO(b/175888145): There's a dependency here on the caller knowing what if
  # any are the valid device names, and since executor construction in some
  # configurations (e.g., in simulations) can be lazy (e.g., because it can be
  # a function of cardinality), errors can manifest at runtime. Consider ways
  # to bundle device enumeration logic with executors that accept device names
  # here and elsewhere in the executor stack.

  def __init__(self, device=None):
    """Creates a new instance of an XLA executor.

    Args:
      device: An optional device name (currently unsupported; must be `None`).
    """
    if device is not None:
      raise ValueError(
          'Explicitly specifying a device is currently not supported.')
    self._backend = xla_client.get_local_backend(None)

  @tracing.trace(span=True)
  async def create_value(self, value, type_spec=None):
    """Embeds `value` of type `type_spec` within this executor.

    Args:
      value: An object that represents the value to embed within the executor.
      type_spec: The `tff.Type` of the value represented by this object, or
        something convertible to it. Can optionally be `None` if `value` is an
        instance of `typed_object.TypedObject`.

    Returns:
      An instance of `XlaValue`.

    Raises:
      TypeError: If the arguments are of the wrong types.
      ValueError: If the type was not specified and cannot be determined from
        the value.
    """
    return XlaValue(value, type_spec, self._backend)

  @tracing.trace
  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, XlaValue)
    if arg is not None:
      py_typecheck.check_type(arg, XlaValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    py_typecheck.check_type(comp.internal_representation,
                            runtime.ComputationCallable)
    if comp.type_signature.parameter is not None:
      result = comp.internal_representation(arg.internal_representation)
    else:
      result = comp.internal_representation()
    return XlaValue(result, comp.type_signature.result, self._backend)

  @tracing.trace
  async def create_struct(self, elements):
    val_elements = []
    type_elements = []
    for k, v in structure.iter_elements(structure.from_container(elements)):
      py_typecheck.check_type(v, XlaValue)
      val_elements.append((k, v.internal_representation))
      type_elements.append((k, v.type_signature))
    struct_val = structure.Struct(val_elements)
    struct_type = computation_types.StructType([
        (k, v) if k is not None else v for k, v in type_elements
    ])
    return XlaValue(struct_val, struct_type, self._backend)

  @tracing.trace
  async def create_selection(self, source, index=None, name=None):
    py_typecheck.check_type(source, XlaValue)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    py_typecheck.check_type(source.internal_representation, structure.Struct)
    if index is not None:
      py_typecheck.check_type(index, int)
      if name is not None:
        raise ValueError(
            'Cannot simultaneously specify name {} and index {}.'.format(
                name, index))
      else:
        return XlaValue(source.internal_representation[index],
                        source.type_signature[index], self._backend)
    elif name is not None:
      py_typecheck.check_type(name, str)
      return XlaValue(
          getattr(source.internal_representation, str(name)),
          getattr(source.type_signature, str(name)), self._backend)
    else:
      raise ValueError('Must specify either name or index.')

  def close(self):
    pass
