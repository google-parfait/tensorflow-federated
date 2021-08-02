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
"""Tests for model serialization."""

import collections
import os

from absl.testing import parameterized
import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning.models import serialization

# Convenience aliases.
TensorType = computation_types.TensorType
StructType = computation_types.StructType
StructWithPythonType = computation_types.StructWithPythonType


class FlattenTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_arg_empty_result', lambda: (), (), {}, StructType([])),
      ('no_arg_tensor_result', lambda: 1.0, (), {}, TensorType(tf.float32)),
      ('no_arg_list_result', lambda: [1.0, 2], (), {},
       StructWithPythonType([TensorType(tf.float32),
                             TensorType(tf.int32)], list)),
      ('no_arg_tuple_result', lambda: (2, 1.0), (), {},
       StructWithPythonType(
           [TensorType(tf.int32), TensorType(tf.float32)], tuple)),
      ('no_arg_odict_result', lambda: collections.OrderedDict(a='abc', b=0),
       (), {},
       StructWithPythonType([('a', TensorType(tf.string)),
                             ('b', TensorType(tf.int32))],
                            collections.OrderedDict)),
      ('no_arg_nested_result',
       lambda: collections.OrderedDict(a='abc', b=(0, 2)), (), {},
       StructWithPythonType([
           ('a', TensorType(tf.string)),
           ('b',
            StructWithPythonType([TensorType(tf.int32),
                                  TensorType(tf.int32)], tuple))
       ], collections.OrderedDict)),
      ('identity_tensor', lambda x: x,
       (tf.TensorSpec([], tf.int32),), {}, TensorType(tf.int32)),
      ('identity_tuple', lambda *args: args,
       (tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.float32)), {},
       StructType(
           [TensorType(tf.int32), TensorType(tf.float32)], tuple)),
      ('identity_list', lambda *args: args, [
          tf.TensorSpec([], tf.int32),
          tf.TensorSpec([], tf.float32)
      ], {}, StructType(
          [TensorType(tf.int32), TensorType(tf.float32)], list)),
      ('identity_odict',
       lambda *args, **kwargs: collections.OrderedDict(**kwargs), (), {
           'a': tf.TensorSpec([], tf.int32),
           'b': tf.TensorSpec([], tf.float32)
       },
       StructType([('a', TensorType(tf.int32)),
                   ('b', TensorType(tf.float32))], collections.OrderedDict)),
  )
  def test_flatten_tf_function(self, fn, args, kwargs, expected_type_spec):
    concrete_fn, type_spec = serialization._make_concrete_flat_output_fn(
        tf.function(fn), *args, **kwargs)
    # The TensorFlow ConcreteFunction type is not exported, so we compare by the
    # name here.
    self.assertEqual(type(concrete_fn).__name__, 'ConcreteFunction')
    self.assertProtoEquals(
        type_spec, type_serialization.serialize_type(expected_type_spec))


class UnflattenTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty_result', (), StructType([]), None, tuple),
      ('tensor_result', tf.constant(1.0), TensorType(
          tf.float32), None, tf.Tensor),
      ('list_result', [1.0, 2],
       StructType([TensorType(tf.float32),
                   TensorType(tf.int32)]), list, list),
      ('tuple_result',
       (2, 1.0), StructType([TensorType(tf.int32),
                             TensorType(tf.float32)]), tuple, tuple),
      ('tuple_result_no_hint',
       (2, 1.0), StructType([TensorType(tf.int32),
                             TensorType(tf.float32)]), None, tuple),
      ('odict_result', collections.OrderedDict(a='abc', b=0),
       StructType([('a', TensorType(tf.string)), ('b', TensorType(tf.int32))
                  ]), collections.OrderedDict, collections.OrderedDict),
      ('odict_result_no_hint', collections.OrderedDict(a='abc', b=0),
       StructType([('a', TensorType(tf.string)), ('b', TensorType(tf.int32))
                  ]), None, collections.OrderedDict),
      ('nested_result', collections.OrderedDict(a='abc', b=(0, 2)),
       StructType([
           ('a', TensorType(tf.string)),
           ('b', StructType([TensorType(tf.int32),
                             TensorType(tf.int32)]))
       ]), collections.OrderedDict, collections.OrderedDict),
      ('nested_result_no_hint', collections.OrderedDict(a='abc', b=(0, 2)),
       StructType([
           ('a', TensorType(tf.string)),
           ('b', StructType([TensorType(tf.int32),
                             TensorType(tf.int32)]))
       ]), None, collections.OrderedDict),
  )
  def test_unflatten_tf_function(self, result, result_type_spec,
                                 python_container_hint,
                                 expected_python_container):
    type_spec_var = tf.Variable(
        type_serialization.serialize_type(result_type_spec).SerializeToString(
            deterministic=True))

    @tf.function
    def fn():
      return tf.nest.flatten(result)

    packed_fn = serialization._unflatten_fn(fn, type_spec_var,
                                            python_container_hint)
    actual_output = packed_fn()
    self.assert_types_equivalent(
        type_conversions.type_from_tensors(actual_output), result_type_spec)
    self.assertIsInstance(actual_output, expected_python_container)


def _test_model_fn(keras_model_fn, loss_fn, test_input_spec):
  """Builds a `model_fn` for testing."""

  def model_fn():
    return keras_utils.from_keras_model(
        keras_model_fn(), input_spec=test_input_spec, loss=loss_fn())

  return model_fn


class _TestModel(model_lib.Model):
  """Test model that returns different signatures when `training` value changes."""

  def __init__(self):
    input_tensor = tf.keras.layers.Input(shape=(3,))
    logits = tf.keras.layers.Dense(5,)(input_tensor)
    predictions = tf.keras.layers.Softmax()(logits)
    self._model = tf.keras.Model(
        inputs=[input_tensor], outputs=[logits, predictions])

  @tf.function
  def predict_on_batch(self, x, training=True):
    if training:
      # Logits returned for training.
      return self._model(x)[0]
    else:
      # Predicted classes returned for inference.
      return tf.argmax(self._model(x)[1], axis=1)

  @tf.function
  def forward_pass(self, batch_input, training=True):
    if training:
      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      logits = self.predict_on_batch(batch_input['x'], training=True)
      loss = loss_fn(y_true=batch_input['y'], y_pred=logits)
      num_examples = tf.shape(logits)[0]
      return model_lib.BatchOutput(
          loss=loss, predictions=(), num_examples=num_examples)
    else:
      predictions = self.predict_on_batch(batch_input['x'], training=False)
      return model_lib.BatchOutput(
          loss=(), predictions=predictions, num_examples=())

  @property
  def trainable_variables(self):
    return self._model.trainable_variables

  @property
  def non_trainable_variables(self):
    return self._model.non_trainable_variables

  @property
  def local_variables(self):
    return []

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        y=tf.TensorSpec(shape=(None, 1), dtype=tf.int32))

  @tf.function
  def report_local_outputs(self):
    return collections.OrderedDict()

  @property
  def federated_output_computation(self):

    @computations.federated_computation(computation_types.at_clients(()))
    def noop(empty_tuple):
      del empty_tuple  # Unused.
      return intrinsics.federated_value((), placements.SERVER)

    return noop


_TEST_MODEL_FNS = [
    ('linear_regression', model_examples.LinearRegression),
    ('inference_training_diff', _TestModel),
    ('keras_linear_regression_tuple_input',
     _test_model_fn(
         model_examples.build_linear_regression_keras_sequential_model,
         tf.keras.losses.MeanSquaredError,
         (tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32)))),
    ('keras_with_embedding',
     _test_model_fn(
         model_examples.build_embedding_keras_model,
         tf.keras.losses.SparseCategoricalCrossentropy,
         collections.OrderedDict(
             x=tf.TensorSpec(shape=[None], dtype=tf.float32),
             y=tf.TensorSpec(shape=[None], dtype=tf.float32)))),
    ('keras_multiple_input',
     _test_model_fn(
         model_examples.build_multiple_inputs_keras_model,
         tf.keras.losses.MeanSquaredError,
         collections.OrderedDict(
             x=collections.OrderedDict(
                 a=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                 b=tf.TensorSpec(shape=[1, 1], dtype=tf.float32)),
             y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32)))),
    ('keras_multiple_output',
     _test_model_fn(
         model_examples.build_multiple_outputs_keras_model,
         tf.keras.losses.MeanSquaredError,
         collections.OrderedDict(
             x=(tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32)),
             y=(tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32))))),
]


class SerializationTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(_TEST_MODEL_FNS)
  def test_round_trip(self, model_fn):
    model = model_fn()
    test_dir = os.path.join(self.get_temp_dir(), 'roundtrip_test')
    try:
      # Clear the any previous output.
      tf.io.gfile.rmtree(test_dir)
    except tf.errors.OpError:
      pass
    serialization.save(model, test_dir)
    loaded_model = serialization.load(test_dir)
    self.assertIsInstance(loaded_model, model_lib.Model)
    self.assertEqual(model.input_spec, loaded_model.input_spec)
    # Assert we can save the loaded_model again.
    serialization.save(loaded_model, test_dir)

    # Build an arbitrary batch for testing functions call.s
    def build_ones(t):
      return np.ones(
          shape=[dim if dim else 1 for dim in t.shape],
          dtype=t.dtype.as_numpy_dtype)

    test_batch = tf.nest.map_structure(build_ones, loaded_model.input_spec)
    if isinstance(test_batch, dict):
      predict_input = test_batch['x']
    else:
      predict_input = test_batch[0]

    # Assert that the models produce the same results.
    for training in [True, False]:
      self.assertAllClose(
          model.predict_on_batch(predict_input, training),
          loaded_model.predict_on_batch(predict_input, training))
      model_result = model.forward_pass(test_batch, training)
      loaded_model_result = loaded_model.forward_pass(test_batch, training)
      self.assertAllClose(
          attr.asdict(model_result), attr.asdict(loaded_model_result))

  @parameterized.named_parameters(_TEST_MODEL_FNS)
  def test_saved_model_to_tflite(self, model_fn):
    model = model_fn()
    test_dir = os.path.join(self.get_temp_dir(), 'tflite_test')
    try:
      # Clear the any previous output.
      tf.io.gfile.rmtree(test_dir)
    except tf.errors.OpError:
      pass
    serialization.save(model, test_dir)
    tflite_flatbuffer = tf.lite.TFLiteConverter.from_saved_model(
        test_dir,
        signature_keys=[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                       ]).convert()
    self.assertNotEmpty(tflite_flatbuffer)


if __name__ == '__main__':
  test_case.main()
