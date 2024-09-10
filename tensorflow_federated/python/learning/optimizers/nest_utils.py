# Copyright 2024, The TensorFlow Federated Authors.
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
"""Utilities for working with nested structures."""

import tensorflow as tf


def _tuplify(x):
  if isinstance(x, tuple):
    return x
  else:
    return (x,)


def map_at_leaves(f, *args, num_outputs: int = 0) -> ...:
  """Applies a function leaf-wise to arguments of matching nested structure.

  For example, if `f(X, Y, Z) -> A, B` then for some nested structure
  `Struct` we will have `map_at_leaves(f, Struct[X], Struct[Y], Struct[Z]) ->
  Struct[A], Struct[B]`. Unlike `tf.nest.map_structure`, this will yield a
  number of outputs matching the number of outputs of `f`.

  Args:
    f: A callable.
    *args: Some number of args of matching nested structure.
    num_outputs: An optional integer specifying the number of outputs. This is
      only used when the leaf structure of each member of *args is empty and
      therefore the number of outputs must be inferred.

  Returns:
    Some number of structures, matching the nested structure of each member of
    args, the number of which matches the number of outputs of `f`.
  """
  if len(args) > 1:
    for arg in args[1:]:
      tf.nest.assert_same_structure(args[0], arg)

  if not tf.nest.flatten(args[0]):
    # In the event that all leavesare empty, no need to do flattening and
    # transposition. We simply return the (empty) structures. We will return
    # `num_outputs` structures matching `args[0]`.;
    if num_outputs < 1:
      raise ValueError(
          f'Found empty arguments of structure {args[0]}, but `num_outputs` was'
          f' set to non-positive value {num_outputs}. The behavior of'
          ' `map_at_leaves` is undefined in such settings. If intending to use'
          ' with empty structures, please specify `num_outputs`.'
      )
    # If there is only one output, return an empty structure. Otherwise return a
    # tuple.
    if num_outputs == 1:
      return args[0]
    else:
      return (args[0],) * num_outputs

  flat_args = [tf.nest.flatten(arg) for arg in args]
  # Apply the function to each set of corresponding flattened elements.
  # We have to tuplify the result of fun if it returns a singleton.
  flat_results = [_tuplify(f(*elements)) for elements in zip(*flat_args)]
  # Transpose the list of result tuples (or single-element tuples).
  flat_results = zip(*flat_results)
  # Unflatten each result list using the first arg as a template.
  results = tuple(tf.nest.pack_sequence_as(args[0], x) for x in flat_results)
  # If the result is a singleton, return it. Otherwise, return a tuple.
  if len(results) == 1:
    return results[0]
  else:
    return results
