# Copyright 2019, The TensorFlow Federated Authors.
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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning.framework import encoding_utils
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def _test_encoder_fn():
  """Returns an example mapping of tensor to encoder, determined by shape."""
  encoder_constructor = te.encoders.as_simple_encoder
  identity_encoder = te.encoders.identity()
  test_encoder = te.core.EncoderComposer(
      te.testing.PlusOneOverNEncodingStage()).make()

  def encoder_fn(tensor):
    if np.prod(tensor.shape) > 1:
      encoder = encoder_constructor(test_encoder,
                                    tf.TensorSpec(tensor.shape, tensor.dtype))
    else:
      encoder = encoder_constructor(identity_encoder,
                                    tf.TensorSpec(tensor.shape, tensor.dtype))
    return encoder

  return encoder_fn


class EncodingUtilsTest(test_case.TestCase, parameterized.TestCase):
  """Tests for utilities for building StatefulFns."""

  def test_broadcast_process_from_model(self):
    model_fn = model_examples.LinearRegression
    broadcast_process = (
        encoding_utils.build_encoded_broadcast_process_from_model(
            model_fn, _test_encoder_fn()))
    self.assertIsInstance(broadcast_process, measured_process.MeasuredProcess)


if __name__ == '__main__':
  test_case.main()
