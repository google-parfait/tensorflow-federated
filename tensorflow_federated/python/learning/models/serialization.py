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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Module for model serialization."""

import collections
import functools

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.computation import computation_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning.models import functional


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

    def raise_not_implemented_error():
      raise NotImplementedError(
          'Do not implement. `report_local_outputs` and '
          '`federated_output_computation` are deprecated and will be removed '
          'in 2022Q1. You should use `report_local_unfinalized_metrics` and '
          '`metric_finalizers` instead. The cross-client metrics aggregation '
          'should be specified as the `metrics_aggregator` argument when you '
          'build a training process or evaluation computation using this model.'
      )

    self._report_local_outputs = raise_not_implemented_error

    self._input_spec = _deserialize_type_spec(
        loaded_module.serialized_input_spec)

    self._federated_output_computation = raise_not_implemented_error

    self._report_local_unfinalized_metrics = loaded_module.report_local_unfinalized_metrics
    self._serialized_metric_finalizers = loaded_module.serialized_metric_finalizers

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

  @tf.function
  def report_local_unfinalized_metrics(self):
    return self._report_local_unfinalized_metrics()

  def metric_finalizers(self):

    def deserialize_metric_finalizer(finalizer):
      return computation_serialization.deserialize_computation(
          computation_pb2.Computation.FromString(
              finalizer.read_value().numpy()))

    return collections.OrderedDict(
        (metric_name, deserialize_metric_finalizer(finalizer)) for metric_name,
        finalizer in self._serialized_metric_finalizers.items())

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


def save(model: model_lib.Model, path: str, input_type=None) -> None:
  """Serializes `model` as a TensorFlow SavedModel to `path`.

  The resulting SavedModel will contain the default serving signature, which
  can be used with the TFLite converter to create a TFLite flatbuffer for
  inference.

  NOTE: The model returned by `tff.learning.models.load` will _not_ be the same
  Python type as the saved model. If the model serialized using this method is
  a subclass of `tff.learning.Model`, that subclass is _not_ returned. All
  method behavior is retained, but the Python type does not cross serialization
  boundaries. The return type of `metric_finalizers` will be an OrderedDict of
  str to `tff.tf_computation` (annotated TFF computations) which could be
  different from that of the model before serialization.

  Args:
    model: The `tff.learning.Model` to save.
    path: The `str` directory path to serialize the model to.
    input_type: An optional structure of `tf.TensorSpec`s representing the
      expected input of `model.predict_on_batch`, to override reading from
      `model.input_spec`. Typically this will be similar to `model.input_spec`,
      with any example labels removed. If None, default to
      `model.input_spec['x']` if the input_spec is a mapping, otherwise default
      to `model.input_spec[0]`.
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
  # Get model prediction input type. If `None`, default to assuming the 'x' key
  # or first element of the model input spec is the input.
  if input_type is None:
    if isinstance(model.input_spec, collections.abc.Mapping):
      input_type = model.input_spec['x']
    else:
      input_type = model.input_spec[0]
  # Serialize predict_on_batch. We must get two concrete versions of the
  # function, as the `training` argument is a Python value that changes the
  # graph computation.
  predict_on_batch_training = _make_concrete_flat_output_fn(
      functools.partial(model.predict_on_batch, training=True), input_type)
  m.predict_on_batch_training = predict_on_batch_training[0]
  m.predict_on_batch_training_type_spec = tf.Variable(
      predict_on_batch_training[1].SerializeToString(deterministic=True),
      trainable=False)
  predict_on_batch_inference = _make_concrete_flat_output_fn(
      functools.partial(model.predict_on_batch, training=False), input_type)
  m.predict_on_batch_inference = predict_on_batch_inference[0]
  m.predict_on_batch_inference_type_spec = tf.Variable(
      predict_on_batch_inference[1].SerializeToString(deterministic=True),
      trainable=False)

  # Serialize the report_local_unfinalized_metrics tf.function.
  m.report_local_unfinalized_metrics = (
      model.report_local_unfinalized_metrics.get_concrete_function())

  # Serialize the metric_finalizers as `tf.Variable`s.
  m.serialized_metric_finalizers = collections.OrderedDict()

  def serialize_metric_finalizer(finalizer, metric_type):
    finalizer_computation = computations.tf_computation(finalizer, metric_type)
    return tf.Variable(
        computation_serialization.serialize_computation(
            finalizer_computation).SerializeToString(deterministic=True),
        trainable=False)

  for metric_name, finalizer in model.metric_finalizers().items():
    metric_type = type_conversions.type_from_tensors(
        model.report_local_unfinalized_metrics()[metric_name])
    m.serialized_metric_finalizers[metric_name] = serialize_metric_finalizer(
        finalizer, metric_type)

  # Serialize the TFF values as string variables that contain the serialized
  # protos from the computation or the type.
  m.serialized_input_spec = tf.Variable(
      type_serialization.serialize_type(
          computation_types.to_type(
              model.input_spec)).SerializeToString(deterministic=True),
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


def save_functional_model(functional_model: functional.FunctionalModel,
                          path: str):
  """Serializes a `FunctionalModel` as a `tf.SavedModel` to `path`.

  Args:
    functional_model: A `tff.learning.models.FunctionalModel`.
    path: A `str` directory path to serialize the model to.
  """
  m = tf.Module()
  # Serialize the initial_weights values as a tf.function that creates a
  # structure of tensors with the initial weights. This way we can add it to the
  # tf.SavedModel and call it to create initial weights after deserialization.
  create_initial_weights = lambda: functional_model.initial_weights
  with tf.Graph().as_default():
    concrete_structured_fn = tf.function(
        create_initial_weights).get_concrete_function()
  model_weights_tensor_specs = tf.nest.map_structure(
      tf.TensorSpec.from_tensor, concrete_structured_fn.structured_outputs)
  initial_weights_result_type_spec = type_serialization.serialize_type(
      computation_types.to_type(model_weights_tensor_specs))
  m.create_initial_weights_type_spec = tf.Variable(
      initial_weights_result_type_spec.SerializeToString(deterministic=True))

  def flat_initial_weights():
    return tf.nest.flatten(create_initial_weights())

  with tf.Graph().as_default():
    m.create_initial_weights = tf.function(
        flat_initial_weights).get_concrete_function()

  # Serialize forward pass concretely, once for training and once for
  # non-training.
  # TODO(b/198150431): try making `training` a `tf.Tensor` parameter to remove
  # the need to for serializing two different function graphs.
  def make_concrete_flat_forward_pass(training: bool):
    """Create a concrete forward_pass function that has flattened output.

    Args:
      training: A boolean indicating whether this is a call in a training loop,
        or evaluation loop.

    Returns:
      A 2-tuple of concrete `tf.function` instance and a `tff.Type` protocol
      buffer message documenting the the result structure returned by the
      concrete function.
    """
    # Save the un-flattened type spec for deserialization later.
    # Note: `training` is a Python boolean, which gets "curried", in a sense,
    # during function conretization. The resulting concrete function only has
    # parameters for `model_weights` and `batch_input`, which are
    # `tf.TensorSpec` structures here.
    with tf.Graph().as_default():
      concrete_structured_fn = functional_model.forward_pass.get_concrete_function(
          model_weights_tensor_specs,
          functional_model.input_spec,
          # Note: training does not appear in the resulting concrete function.
          training=training)
    output_tensor_spec_structure = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, concrete_structured_fn.structured_outputs)
    result_type_spec = type_serialization.serialize_type(
        computation_types.to_type(output_tensor_spec_structure))

    @tf.function
    def flat_forward_pass(model_weights, batch_input, training):
      return tf.nest.flatten(
          functional_model.forward_pass(model_weights, batch_input, training))

    with tf.Graph().as_default():
      flat_concrete_fn = flat_forward_pass.get_concrete_function(
          model_weights_tensor_specs,
          functional_model.input_spec,
          # Note: training does not appear in the resulting concrete function.
          training=training)
    return flat_concrete_fn, result_type_spec

  fw_pass_training, fw_pass_training_type_spec = make_concrete_flat_forward_pass(
      training=True)
  m.flat_forward_pass_training = fw_pass_training
  m.forward_pass_training_type_spec = tf.Variable(
      fw_pass_training_type_spec.SerializeToString(deterministic=True),
      trainable=False)

  fw_pass_inference, fw_pass_inference_type_spec = make_concrete_flat_forward_pass(
      training=False)
  m.flat_forward_pass_inference = fw_pass_inference
  m.forward_pass_inference_type_spec = tf.Variable(
      fw_pass_inference_type_spec.SerializeToString(deterministic=True),
      trainable=False)

  # Serialize predict_on_batch, once for training, once for non-training.
  x_type = functional_model.input_spec[0]

  # TODO(b/198150431): try making `training` a `tf.Tensor` parameter to remove
  # the need to for serializing two different function graphs.
  def make_concrete_flat_predict_on_batch(training: bool):
    """Create a concrete predict_on_batch function that has flattened output.

    Args:
      training: A boolean indicating whether this is a call in a training loop,
        or evaluation loop.

    Returns:
      A 2-tuple of concrete `tf.function` instance and a `tff.Type` protocol
      buffer message documenting the the result structure returned by the
      concrete
      function.
    """
    # Save the un-flattened type spec for deserialization later.
    # Note: `training` is a Python boolean, which gets "curried", in a sense,
    # during function conretization. The resulting concrete function only has
    # parameters for `model_weights` and `batch_input`, which are
    # `tf.TensorSpec` structures here.
    concrete_structured_fn = tf.function(
        functional_model.predict_on_batch).get_concrete_function(
            model_weights_tensor_specs,
            x_type,
            # Note: training does not appear in the resulting concrete function.
            training=training)
    output_tensor_spec_structure = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, concrete_structured_fn.structured_outputs)
    result_type_spec = type_serialization.serialize_type(
        computation_types.to_type(output_tensor_spec_structure))

    @tf.function
    def flat_predict_on_batch(model_weights, x, training):
      return tf.nest.flatten(
          functional_model.predict_on_batch(model_weights, x, training))

    flat_concrete_fn = tf.function(flat_predict_on_batch).get_concrete_function(
        model_weights_tensor_specs,
        x_type,
        # Note: training does not appear in the resulting concrete function.
        training=training)
    return flat_concrete_fn, result_type_spec

  with tf.Graph().as_default():
    predict_training, predict_training_type_spec = make_concrete_flat_predict_on_batch(
        training=True)
  m.predict_on_batch_training = predict_training
  m.predict_on_batch_training_type_spec = tf.Variable(
      predict_training_type_spec.SerializeToString(deterministic=True),
      trainable=False)

  with tf.Graph().as_default():
    predict_inference, predict_inference_type_spec = make_concrete_flat_predict_on_batch(
        training=False)
  m.predict_on_batch_inference = predict_inference
  m.predict_on_batch_inference_type_spec = tf.Variable(
      predict_inference_type_spec.SerializeToString(deterministic=True),
      trainable=False)

  # Serialize TFF values as string variables that contain the serialized
  # protos from the computation or the type.
  m.serialized_input_spec = tf.Variable(
      type_serialization.serialize_type(
          computation_types.to_type(
              functional_model.input_spec)).SerializeToString(
                  deterministic=True),
      trainable=False)

  # Save everything
  _save_tensorflow_module(m, path)


class _LoadedFunctionalModel(functional.FunctionalModel):
  """Creates a `FunctionalModel` from a loaded SavedModel."""

  def __init__(self, loaded_module):
    self._loaded_module = loaded_module
    self._input_spec = tf.nest.map_structure(
        lambda t: tf.TensorSpec(dtype=t.dtype, shape=t.shape),
        _deserialize_type_spec(loaded_module.serialized_input_spec))

    weights_nested_tensor_specs = _deserialize_type_spec(
        loaded_module.create_initial_weights_type_spec, tuple)
    self._initial_weights = tf.nest.pack_sequence_as(
        weights_nested_tensor_specs,
        # Convert EagerTensors to numpy arrays, necessary to avoid trying
        # to capture EagerTensors in different graphs when doing:
        # build_fedarated_averaging_process(
        #   ModelFromFunctional(_LoadedFunctionalModel)
        [w.numpy() for w in loaded_module.create_initial_weights()])

    def unflatten_forward_pass_fn(flat_forward_pass,
                                  serialized_result_type_variable):
      result_tensor_specs = _deserialize_type_spec(
          serialized_result_type_variable, model_lib.BatchOutput)

      def forward_pass(model_weights, batch_input):
        result = flat_forward_pass(model_weights, batch_input)
        return tf.nest.pack_sequence_as(result_tensor_specs, result)

      return forward_pass

    self._forward_pass_training = unflatten_forward_pass_fn(
        loaded_module.flat_forward_pass_training,
        loaded_module.forward_pass_training_type_spec)
    self._forward_pass_inference = unflatten_forward_pass_fn(
        loaded_module.flat_forward_pass_inference,
        loaded_module.forward_pass_inference_type_spec)

    def unflatten_predict_on_batch_fn(flat_predict_on_batch,
                                      serialized_result_type_variable):
      result_tensor_specs = _deserialize_type_spec(
          serialized_result_type_variable, tuple)

      def predict_on_batch(model_weights, x):
        result = flat_predict_on_batch(model_weights=model_weights, x=x)
        if tf.is_tensor(result):
          return result
        return tf.nest.pack_sequence_as(result_tensor_specs, result)

      return predict_on_batch

    self._predict_on_batch_training = unflatten_predict_on_batch_fn(
        loaded_module.predict_on_batch_training,
        loaded_module.predict_on_batch_training_type_spec)
    self._predict_on_batch_inference = unflatten_predict_on_batch_fn(
        loaded_module.predict_on_batch_inference,
        loaded_module.predict_on_batch_inference_type_spec)

  @property
  def initial_weights(self):
    return self._initial_weights

  def forward_pass(self,
                   model_weights,
                   batch_input,
                   training=True) -> model_lib.BatchOutput:
    """Runs the forward pass and returns results."""
    if training:
      return self._forward_pass_training(
          model_weights=model_weights, batch_input=batch_input)
    else:
      return self._forward_pass_inference(
          model_weights=model_weights, batch_input=batch_input)

  def predict_on_batch(self, model_weights, x, training=True):
    """Returns tensor(s) interpretable by the loss function."""
    if training:
      return self._predict_on_batch_training(model_weights=model_weights, x=x)
    else:
      return self._predict_on_batch_inference(model_weights=model_weights, x=x)

  @property
  def input_spec(self):
    return self._input_spec


def load_functional_model(path: str) -> functional.FunctionalModel:
  """Deserializes a TensorFlow SavedModel at `path` to a functional model.

  Args:
    path: The `str` path pointing to a SavedModel.

  Returns:
    A `tff.learning.models.FunctionalModel`.
  """
  py_typecheck.check_type(path, str)
  if not path:
    raise ValueError('`path` must be a non-empty string, cannot deserialize '
                     'models without an output path.')
  return _LoadedFunctionalModel(tf.saved_model.load(path))
