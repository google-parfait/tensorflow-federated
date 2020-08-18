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
"""Dataset reduce functions for federated optimization algorithms."""

from typing import Callable, Iterable, Union

import tensorflow as tf

_ReduceFnCallable = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def _dataset_reduce_fn(reduce_fn: _ReduceFnCallable,
                       dataset: tf.data.Dataset) -> tf.Tensor:
  return dataset.reduce(initial_state=tf.constant(0), reduce_func=reduce_fn)


def _for_iter_dataset_fn(
    reduce_fn: _ReduceFnCallable,
    dataset: Iterable,  # pylint: disable=g-bare-generic
) -> tf.Tensor:
  """Performs dataset reduce for simulation performance."""
  # TODO(b/162683412): use `tf.data.Dataset` instead of `Iterable` for pytype.
  num_examples_sum = tf.constant(0, dtype=tf.int32)
  # TODO(b/155208489): this is a workaround for GPU simulation because
  # `tf.device` does not cross the boundary of dataset ops. TF use a different
  # set of ops when we explicitly use `iter` for dataset.
  for batch in iter(dataset):
    num_examples_sum = reduce_fn(num_examples_sum, batch)
  return num_examples_sum


def build_dataset_reduce_fn(
    simulation_flag: bool = True
) -> Callable[[_ReduceFnCallable, Union[tf.data.Dataset, Iterable]], tf.Tensor]:  # pylint: disable=g-bare-generic
  # TODO(b/162683412): remove `Iterable` after pytype fix.
  """Retruns a reduce loop function on input dataset."""
  if simulation_flag:
    return _for_iter_dataset_fn
  else:
    return _dataset_reduce_fn
