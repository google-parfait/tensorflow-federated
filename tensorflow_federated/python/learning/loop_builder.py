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

from collections.abc import Callable, Iterable
import enum
from typing import Any, Union

import tensorflow as tf

_ReduceFnCallable = Callable[[Any, tf.Tensor], Any]


@enum.unique
class LoopImplementation(enum.Enum):
  """An enum to specify which implementation of the training loop to use.

  Attributes:
    DATASET_ITERATOR: A training loop that uses Dataset iterator ops. This is
      required when running on hosts with multiple GPUs to effectively leverage
      all GPUs on the host.
    DATASET_REDUCE: A training loop that uses the DatasetReduce op. This is
      required when running on hosts with TPUs to allow the MLIR Bridge to
      compile the reduction function to XLA HLO for TPUs.
    SLICE_FOLDL: A training loop that expects a nested structure of tensors
      representing all the steps of the dataset such that iterating over the 0th
      dimension is equivalent to iterating over the dataset. These tensors
      should be in the shape `[<# of batches>, <batch size>, <... examples shape
      ... >]`, so that the size of the dataset is `# of batches * batch size`.
      Note: this loop implementation requires either padding out the last batch
        or discarding partial batches.
  """

  DATASET_ITERATOR = enum.auto()
  DATASET_REDUCE = enum.auto()
  SLICE_FOLDL = enum.auto()


def _dataset_reduce_fn(
    reduce_fn: _ReduceFnCallable,
    dataset: tf.data.Dataset,
    initial_state_fn: Callable[[], Any] = lambda: tf.constant(0),
) -> Any:
  return dataset.reduce(initial_state=initial_state_fn(), reduce_func=reduce_fn)


def _for_iter_dataset_fn(
    reduce_fn: _ReduceFnCallable,
    dataset: tf.data.Dataset,
    initial_state_fn: Callable[[], Any] = lambda: tf.constant(0),
) -> Any:
  """Performs dataset reduce for simulation performance."""
  # TODO: b/155208489 - this is a workaround for GPU simulation because
  # `tf.device` does not cross the boundary of dataset ops. TF use a different
  # set of ops when we explicitly use `iter` for dataset.
  update_state = initial_state_fn()
  for batch in iter(dataset):
    update_state = reduce_fn(update_state, batch)
  return update_state


def _slice_foldl_fn(
    reduce_fn: _ReduceFnCallable,
    dataset_as_arrays: Any,
    initial_state_fn: Callable[[], Any] = lambda: tf.constant(0),
) -> Any:
  """Applies a reduction across the slices of `dataset_as_arrays`."""
  return tf.foldl(
      fn=reduce_fn,
      elems=dataset_as_arrays,
      initializer=initial_state_fn(),
      # Since `reduce_fn` is generally a training step, future steps depend
      # on the model weights updated in the previous step setting up a data
      # dependency on execution. Parallelism here will allow any next step
      # "pre-work" to occur, but we limited to 2 steps at a time to put a cap on
      # memory usage.
      parallel_iterations=2,
  )


def build_training_loop(
    loop_implementation: LoopImplementation,
) -> Callable[
    [_ReduceFnCallable, Union[tf.data.Dataset, Iterable[Any]], Any], tf.Tensor
]:
  # TODO: b/162683412 - remove `Iterable` after pytype fix.
  """Returns a reduce loop function on input dataset."""
  if loop_implementation == LoopImplementation.DATASET_ITERATOR:
    return _for_iter_dataset_fn
  elif loop_implementation == LoopImplementation.DATASET_REDUCE:
    return _dataset_reduce_fn
  elif loop_implementation == LoopImplementation.SLICE_FOLDL:
    return _slice_foldl_fn
  else:
    raise NotImplementedError(
        f"Unknown implementation requested: {loop_implementation}"
    )
