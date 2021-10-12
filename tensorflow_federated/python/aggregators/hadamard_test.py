# Copyright 2019, The TensorFlow Authors. All Rights Reserved.
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

from tensorflow_federated.python.aggregators import hadamard


class FastWalshHadamardTransformTests(tf.test.TestCase, parameterized.TestCase):
  """Tests for `fast_walsh_hadamard_transform` method."""

  @parameterized.named_parameters(('2', 2), ('4', 4), ('8', 8), ('16', 16))
  def test_is_rotation(self, dim):
    """Tests the transform acts as a rotation."""
    x = tf.random.normal([1, dim])
    hx = hadamard.fast_walsh_hadamard_transform(x)
    # Check that x and hx are not the same, but have equal norm.
    self.assertGreater(np.linalg.norm(x - hx), 1e-3)
    self.assertAllClose(np.linalg.norm(x), np.linalg.norm(hx))

  @parameterized.named_parameters(('1', 1), ('2', 2), ('5', 5), ('11', 11))
  def test_apply_twice_equals_identity(self, first_dim):
    """Tests applying the transform twice is equal to identity."""
    x = tf.random.normal([first_dim, 8])
    hx = hadamard.fast_walsh_hadamard_transform(x)
    hhx = hadamard.fast_walsh_hadamard_transform(hx)
    self.assertAllEqual(x.shape, hhx.shape)
    self.assertAllClose(x, hhx)

  @parameterized.named_parameters(('1', [1]), ('1x4x4', [1, 4, 4]),
                                  ('1x1x1x4', [1, 1, 1, 4]))
  def test_illegal_inputs_shape(self, dims):
    """Tests incorrect rank of the input."""
    x = tf.random.normal(dims)
    with self.assertRaisesRegex(hadamard.TensorShapeError,
                                'Number of dimensions of x must be 2.'):
      hadamard.fast_walsh_hadamard_transform(x)

  @parameterized.named_parameters(('1x3', [1, 3]), ('1x7', [1, 7]),
                                  ('1x9', [1, 9]), ('4x3', [4, 3]))
  def test_illegal_inputs_static_power_of_two(self, dims):
    """Tests incorrect static shape of the rank 2 input."""
    x = tf.random.normal(dims)
    with self.assertRaisesRegex(hadamard.TensorShapeError,
                                'The dimension of x must be a power of two.'):
      hadamard.fast_walsh_hadamard_transform(x)

  def test_illegal_inputs_dynamic_power_of_two(self):
    """Tests incorrect dynamic shape of the rank 2 input."""

    # Explicit drop to graph mode for non-statically known shape to be possible.
    @tf.function
    def test_fn():
      rand = tf.random.uniform((), maxval=3, dtype=tf.int32) + 1
      # The created x has shape (3, 3) or (3, 9) or (3, 27), chosen randomly and
      # thus statically not known. In all cases, it is not a power of two.
      x = tf.random.normal((3, 3**rand))
      return hadamard.fast_walsh_hadamard_transform(x)

    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        'The dimension of x must be a power of two.'):
      test_fn()

  @parameterized.named_parameters(('1x1', [1, 1]), ('4x1', [4, 1]),
                                  ('2x2', [2, 2]), ('1x4', [1, 4]),
                                  ('1x8', [1, 8]))
  def test_static_input_output_shape(self, dims):
    """Tests static output shape is identical to static input shape."""
    x = tf.random.normal(dims)
    hx = hadamard.fast_walsh_hadamard_transform(x)
    hhx = hadamard.fast_walsh_hadamard_transform(hx)
    self.assertEqual(dims, hx.shape.as_list())
    self.assertEqual(dims, hhx.shape.as_list())
    self.assertAllClose(x, hhx)

  def test_dynamic_input_shape(self):
    """Tests dynamic input shape."""

    # Explicit drop to graph mode for non-statically known shape to be possible.
    @tf.function
    def test_fn():
      rand = tf.random.uniform((), maxval=4, dtype=tf.int32)
      x = tf.random.normal((3, 2**rand))
      hx = hadamard.fast_walsh_hadamard_transform(x)
      hhx = hadamard.fast_walsh_hadamard_transform(hx)
      return x, hhx

    x, hhx = test_fn()
    self.assertAllEqual(x.shape, hhx.shape)
    self.assertAllClose(x, hhx)

  def test_dynamic_input_shape_dim_one(self):
    """Tests input shape where the second dimension is 1, dynamically known."""

    # Explicit drop to graph mode for non-statically known shape to be possible.
    @tf.function
    def test_fn():
      rand = tf.random.uniform((), maxval=1, dtype=tf.int32)
      x = tf.random.normal((3, 2**rand))
      hx = hadamard.fast_walsh_hadamard_transform(x)
      hhx = hadamard.fast_walsh_hadamard_transform(hx)
      return x, hhx

    x, hhx_tf = test_fn()
    self.assertAllEqual(x.shape, hhx_tf.shape)
    self.assertAllClose(x, hhx_tf)

  def test_spreads_information(self):
    """Tests that input is 'spread' in the output space."""
    x = tf.one_hot(indices=[17], depth=256)
    hx = hadamard.fast_walsh_hadamard_transform(x)

    # The hx should contain only values approximately 1/16 or -1/16.
    min_hx = tf.math.reduce_min(hx)
    max_hx = tf.math.reduce_max(hx)
    min_abs_hx = tf.math.reduce_min(tf.math.abs(hx))
    self.assertAllClose(-1.0/16.0, min_hx)
    self.assertAllClose(1.0/16.0, max_hx)
    self.assertAllClose(1.0/16.0, min_abs_hx)

if __name__ == '__main__':
  tf.test.main()
