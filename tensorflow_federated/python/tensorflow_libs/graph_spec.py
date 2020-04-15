# Lint as: python3
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
"""TFF-independent data class for the data TFF keeps in its TF protos."""
from typing import Optional, Sequence

import attr
import tensorflow as tf


def _check_names_are_strings(instance, attribute, value):
  for name in value:
    if not isinstance(name, str):
      raise TypeError('Each entry in {} must be of string type; '
                      'encountered an element of type {}'.format(
                          attribute, type(name)))


@attr.s(frozen=True, eq=False)
class GraphSpec:
  """Container class for validating input to graph merging functions.

  Mirrors the serialization format of TFF, the main difference being that
  `GraphSpec` takes an already flattened list of names as `in_names` and
  `out_names`, as opposed to the single binding taken by TFF serialization.

  Attributes:
    graph_def: Instance of `tf.compat.v1.GraphDef`.
    init_op: Either a string name of the init op in the graph represented by
      `graph_def`, or `None` if there is no such init op.
    in_names: A Python `list` or `tuple of string names corresponding to the
      input objects in `graph_def`. Must be empty if there are no such inputs.
    out_names: An iterable of string names of the output tensors in `graph_def`,
      subject to the same restrictions as `in_names`.
  """
  graph_def: tf.compat.v1.GraphDef = attr.ib(
      validator=attr.validators.instance_of(tf.compat.v1.GraphDef))
  init_op: Optional[str] = attr.ib(
      validator=attr.validators.instance_of((str, type(None))))
  in_names: Sequence[str] = attr.ib(validator=_check_names_are_strings)
  out_names: Sequence[str] = attr.ib(validator=_check_names_are_strings)
