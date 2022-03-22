# Copyright 2022, The TensorFlow Federated Authors.
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
"""Function wrapper ensuring sequential execution.

The SequentialWrapper adds additional control dependencies to function calls
which reduce parallelism and ensure that calls to the function are made
in sequence.

This is handy to keep overall memory low when processing functions with a large
temporary memory pressure. For example the function can be used when loading
checkpoint values from tensors and assigning them into variables or when
checking for NaN values.
"""

from typing import Any, Callable

import tensorflow as tf


class SequentialWrapper:
  """Function wrapper ensuring sequential processing of the TF.

  Every consequent invocation of the function will have control dependencies
  on the output of the previous call.
  """

  def __init__(self, wrapped_fn: Callable[..., Any]):
    """Instantiates SequentialWrapper.

    Args:
      wrapped_fn: A function which should be applied sequentially.
    """
    self._wrapped_fn = wrapped_fn
    self._dependencies = []

  def __call__(self, *args, **kwargs):
    with tf.control_dependencies(self._dependencies):
      result = self._wrapped_fn(*args, **kwargs)
    self._dependencies = tf.nest.flatten([result])
    return result
