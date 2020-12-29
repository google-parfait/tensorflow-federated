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
"""Defines experimental functions/classes for constructing TFF computations."""

from tensorflow_federated.experimental.python.core.impl.wrappers import computation_wrapper_instances

jax_computation = computation_wrapper_instances.jax_wrapper
jax_computation.__doc__ = (
    """Decorates/wraps Python functions containing JAX code as TFF computations.

  This wrapper can be used in a similar manner to `tff.tf_computation`, with
  exception of the following:

  * The code in the wrapped Python function must be JAX code that can be
    compiled to XLA (e.g., code that one would expect to be able to annotate
    with `@jax.jit`).

  * The inputs and outputs must be tensors, or (possibly recursively) nested
    structures of tensors. Sequences are currently not supported.

  Example:

  ```
  @tff.experimental.jax_computation(tf.int32)
  def comp(x):
    return jax.numpy.add(x, np.int32(10))
  ```
  """)
