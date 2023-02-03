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
"""TensorFlow utility for fast Walsh-Hadamard transform."""

import math
import tensorflow as tf


class TensorShapeError(ValueError):
  pass


def fast_walsh_hadamard_transform(x):
  """Applies the fast Walsh-Hadamard transform.

  See https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform.

  This method uses a composition of existing TensorFlow operations to implement
  the transform.

  The input must be a rank-2 tensor with shape `[a, b]`, where `b` must be a
  power of two, not required to be statically known. The transform will be
  applied for each of the `a` dimensions. That is, the returned tensor y has
  shape `[a, b]` where `y[i, :]` is the product `x[i, :]*H`, where `H` is the
  (normalized) Hadamard matrix.

  Args:
    x: A tensor of shape `[a, b]`.

  Returns:
    A transformed tensor of shape `[a, b]`.

  Raises:
    TensorShapeError: If the input is not rank 2 tensor, or if the second
      dimension is statically known and is not a power of two.
    tf.errors.InvalidArgumentError: If the second dimension is not statically
      known and is not a power of two. Note that in graph execution, this error
      is not raised during the execution of the Python function, but during
      execution of the resulting computation.
  """
  x = tf.convert_to_tensor(x)
  if x.shape.ndims != 2:
    raise TensorShapeError(
        f'Number of dimensions of x must be 2. Shape of x: {x.shape}'
    )

  original_x_shape = x.shape.as_list()
  dim = x.shape.as_list()[-1]

  if dim is None:  # dim is not statically known.
    dim = tf.shape(x)[-1]
    log2 = tf.cast(
        tf.math.round(tf.math.log(tf.cast(dim, tf.float32)) / tf.math.log(2.0)),
        tf.int32,
    )
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                dim,
                tf.math.pow(2, log2),
                message=(
                    'The dimension of x must be a power of two.'
                    'Provided dimension is: %s'
                )
                % dim,
            )
        ]
    ):
      x = tf.identity(x)
  else:  # dim is statically known.
    if not (dim and (dim & (dim - 1)) == 0):
      raise TensorShapeError(
          'The dimension of x must be a power of two. '
          f'Provided dimension is: {dim}'
      )
    log2 = int(math.ceil(math.log2(dim)))
    if dim == 1:  # Equivalent to identity.
      return tf.identity(x)

  h_2x2 = tf.constant([[1.0, 1.0], [1.0, -1.0]], dtype=x.dtype)
  permutation = tf.constant([0, 2, 1])

  def _hadamard_step(x, dim):
    """A single step in the fast Walsh-Hadamard transform."""
    x_shape = x.shape.as_list()
    x = tf.reshape(x, [-1, 2])
    x = tf.matmul(x, h_2x2)
    x = tf.reshape(x, [-1, dim // 2, 2])
    x = tf.transpose(x, perm=permutation)
    x.set_shape(x_shape)  # Failed shape inference in tf.while_loop.
    return x

  def _fwht(x, dim, log2):
    x = tf.reshape(x, [-1, 2, dim // 2])
    index = tf.constant(0)
    cond = lambda i, x: tf.less(i, log2)
    body = lambda i, x: [i + 1, _hadamard_step(x, dim)]
    index, x = tf.while_loop(cond, body, [index, x])
    return x

  x = tf.cond(
      tf.equal(dim, 1), lambda: tf.identity(x), lambda: _fwht(x, dim, log2)
  )

  x = tf.reshape(x, [-1, dim])
  x /= tf.sqrt(tf.cast(dim, x.dtype))  # Normalize.
  x.set_shape(original_x_shape)  # Failed shape inference after tf.while_loop.
  return x
