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
"""Module for `tff.learning.Model` serialization."""

import collections
import functools

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.learning import model as model_lib


class _LoadedSavedModel(model_lib.Model):
  """Private class for instantiating `tff.learning.Model` from a SavedModel."""

  def __init__(self, loaded_module):
    self._loaded_module = loaded_module
    self._trainable_variables = loaded_module.tff_trainable_variables
    self._non_trainable_variables = loaded_module.tff_non_trainable_variables
    self._local_variables = loaded_module.tff_local_variables

    self._forward_pass_training = _unflatten_fn(
        loaded_module.flat_forward_pass_training,
        loaded_module.forward_pass_training_type_spec, model_lib.BatchOutput)
    self._forward_pass_inference = _unflatten_fn(
        loaded_module.flat_forward_pass_inference,
        loaded_module.forward_pass_inference_type_spec, model_lib.BatchOutput)

    self._predict_on_batch_training = _unflatten_fn(
        loaded_module.predict_on_batch_training,
        loaded_module.predict_on_batch_training_type_spec, tuple)
    self._predict_on_batch_inference = _unflatten_fn(
        loaded_module.predict_on_batch_inference,
        loaded_module.predict_on_batch_inference_type_spec, tuple)

    self._report_local_outputs = loaded_module.report_local_outputs
    self._input_spec = _deserialize_type_spec(
        loaded_module.serialized_input_spec)
    self._federated_output_computation = computation_serialization.deserialize_computation(
        computation_pb2.Computation.FromString(
            loaded_module.serialized_federated_output_computation.read_value(
            ).numpy()))

  @property
  def input_spec(self):
    return self._input_spec

  @tf.function
  def forward_pass(self, batch_input, training=True):
    if training:
      return self._forward_pass_training(batch_input)
    else:
      return self._forward_pass_inference(batch_input)

  @tf.function
  def predict_on_batch(self, x, training=True):
    if training:
      return self._predict_on_batch_training(x)
    else:
      return self._predict_on_batch_inference(x)

  @property
  def trainable_variables(self):
    return self._trainable_variables

  @property
  def non_trainable_variables(self):
    return self._non_trainable_variables

  @property
  def local_variables(self):
    return self._local_variables

  @tf.function
  def report_local_outputs(self):
    return self._report_local_outputs()

  @property
  def federated_output_computation(self):
    return self._federated_output_computation


def _save_tensorflow_module(tf_module: tf.Module, path: str) -> None:
  """Serialize a `tf.Module` to a path using the SavedModel format."""
  tf.saved_model.save(
      tf_module,
      export_dir=path,
      signatures=tf_module.predict_on_batch_inference)


def _make_concrete_flat_output_fn(fn, *args, **kwargs):
  """Create a concrete function that has flattened output.

  TensorFlow SavedModel format requires flat structures of outputs, and
  cannot serialize custom Python classes (e.g. the BatchOutput attrs
  classes). Here we wrap the method in a `tf.function` that flattens its
  output. Then we repack the flattened output when loading the SavedModel.

  Args:
    fn: Function to wrap in `tf.function` decorator and concretize with
      arguments in `*args` and `**kwargs` for adding to a `tf.Module.
    *args: Positional arguments to `tf.function.get_concrete_function`.
    **kwargs: Keyword arguments to `tf.function.get_concrete_function`.

  Returns:
    A 2-tuple of concrete `tf.function` instance and a `tff.Type` protocol
    buffer message documenting the the result structure returned by the concrete
    function.
  """
  # Save the un-flattened type spec for deserialization later. Wrap in a lambda
  # because `tf.function` doesn't know how to deal with functools.Partial types
  # that we may have created earlier.
  structured_fn = lambda *args, **kwargs: fn(*args, **kwargs)  # pylint: disable=unnecessary-lambda
  concrete_fn = tf.function(structured_fn).get_concrete_function(
      *args, **kwargs)
  tensor_types = tf.nest.map_structure(computation_types.TensorType,
                                       concrete_fn.output_dtypes,
                                       concrete_fn.output_shapes)
  result_type_spec = type_serialization.serialize_type(
      computation_types.to_type(tensor_types))

  def flattened_output(*args, **kwargs):
    return tf.nest.flatten(fn(*args, **kwargs))

  flat_concrete_fn = tf.function(flattened_output).get_concrete_function(
      *args, **kwargs)
  return flat_concrete_fn, result_type_spec


def _deserialize_type_spec(serialize_type_variable, python_container=None):
  """Deserialize a `tff.Type` protocol buffer into a python class instance."""
  type_spec = type_serialization.deserialize_type(
      computation_pb2.Type.FromString(
          serialize_type_variable.read_value().numpy()))
  if type_spec.is_struct() and python_container is not None:
    type_spec = computation_types.StructWithPythonType(
        structure.iter_elements(type_spec), python_container)
  return type_conversions.type_to_tf_structure(type_spec)


def _unflatten_fn(fn, serialized_type_variable, python_container=None):
  """Unflattens a previously flattened concrete function.

  Args:
    fn: A tf.function loaded from a TensorFlow SavedModel.
    serialized_type_variable: A `tf.Variable` holding the serialized `tff.Type`
      protocol buffer message specifying the structured output format of `fn`.
    python_container: A Python class that which the resulting
      `tff.struct.Struct` will be converted to.

  Returns:
    A tf.function callable that returns output in the nested structure.
  """
  nested_tensor_specs = _deserialize_type_spec(serialized_type_variable,
                                               python_container)

  def structured_output_fn(*args, **kwargs):
    result = fn(*args, **kwargs)
    structured_result = tf.nest.pack_sequence_as(nested_tensor_specs, result)
    return structured_result

  return tf.function(structured_output_fn)


def save(model: model_lib.Model, path: str) -> None:
  """Serializes `model` as a TensorFlow SavedModel to `path`.

  The resulting SavedModel will contain the default serving signature, which
  can be used with the TFLite converter to create a TFLite flatbuffer for
  inference.

  NOTE: The model returned by `tff.learning.models.load` will _not_ be the same
  Python type as the saved model. If the model serialized using this method is
  a subclass of `tff.learning.Model`, that subclass is _not_ returned. All
  method behavior is retained, but the Python type does not cross serialization
  boundaries.

  Args:
    model: The `tff.learning.Model` to save.
    path: The `str` directory path to serialize the model to.
  """
  py_typecheck.check_type(model, model_lib.Model)
  py_typecheck.check_type(path, str)
  if not path:
    raise ValueError('`path` must be a non-empty string, cannot serialize '
                     'models without an output path.')
  if isinstance(model, _LoadedSavedModel):
    # If we're saving a previously loaded model, we can simply use the module
    # already internal to the Model.
    _save_tensorflow_module(model._loaded_module, path)  # pylint: disable=protected-access
    return

  m = tf.Module()
  # We prefixed with `tff_` because `trainable_variables` is an attribute
  # reserved by `tf.Module`.
  m.tff_trainable_variables = model.trainable_variables
  m.tff_non_trainable_variables = model.non_trainable_variables
  m.tff_local_variables = model.local_variables
  # Serialize forward_pass. We must get two concrete versions of the
  # function, as the `training` argument is a Python value that changes the
  # graph computation. We serialize the output type so that we can repack the
  # flattened values after loaded the saved model.
  forward_pass_training = _make_concrete_flat_output_fn(
      functools.partial(model.forward_pass, training=True), model.input_spec)
  m.flat_forward_pass_training = forward_pass_training[0]
  m.forward_pass_training_type_spec = tf.Variable(
      forward_pass_training[1].SerializeToString(deterministic=True),
      trainable=False)

  forward_pass_inference = _make_concrete_flat_output_fn(
      functools.partial(model.forward_pass, training=False), model.input_spec)
  m.flat_forward_pass_inference = forward_pass_inference[0]
  m.forward_pass_inference_type_spec = tf.Variable(
      forward_pass_inference[1].SerializeToString(deterministic=True),
      trainable=False)
  # Serialize predict_on_batch. We must get two concrete versions of the
  # function, as the `training` argument is a Python value that changes the
  # graph computation.
  if isinstance(model.input_spec, collections.abc.Mapping):
    x_type = model.input_spec['x']
  else:
    x_type = model.input_spec[0]
  predict_on_batch_training = _make_concrete_flat_output_fn(
      functools.partial(model.predict_on_batch, training=True), x_type)
  m.predict_on_batch_training = predict_on_batch_training[0]
  m.predict_on_batch_training_type_spec = tf.Variable(
      predict_on_batch_training[1].SerializeToString(deterministic=True),
      trainable=False)
  predict_on_batch_inference = _make_concrete_flat_output_fn(
      functools.partial(model.predict_on_batch, training=False), x_type)
  m.predict_on_batch_inference = predict_on_batch_inference[0]
  m.predict_on_batch_inference_type_spec = tf.Variable(
      predict_on_batch_inference[1].SerializeToString(deterministic=True),
      trainable=False)
  # Serialize the report_local_outputs tf.function.
  m.report_local_outputs = model.report_local_outputs
  # Serialize the TFF values as string variables that contain the serialized
  # protos from the computation or the type.
  m.serialized_input_spec = tf.Variable(
      type_serialization.serialize_type(
          computation_types.to_type(
              model.input_spec)).SerializeToString(deterministic=True),
      trainable=False)
  m.serialized_federated_output_computation = tf.Variable(
      computation_serialization.serialize_computation(
          model.federated_output_computation).SerializeToString(
              deterministic=True),
      trainable=False)
  _save_tensorflow_module(m, path)


def load(path: str) -> model_lib.Model:
  """Deserializes a TensorFlow SavedModel at `path` to a `tff.learning.Model`.

  Args:
    path: The `str` path pointing to a SavedModel.

  Returns:
    A `tff.learning.Model`.
  """
  py_typecheck.check_type(path, str)
  if not path:
    raise ValueError('`path` must be a non-empty string, cannot deserialize '
                     'models without an output path.')
  return _LoadedSavedModel(tf.saved_model.load(path))
