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
"""Configuration classes for client datasets."""

from typing import Optional

import attrs


def _check_positive(instance, attribute, value):
  del instance  # Unused.
  if value <= 0:
    raise ValueError(f'{attribute.name} must be positive. Found {value}.')


@attrs.define(eq=False, frozen=True)
class ClientSpec:
  """Contains information for configuring clients within a training task.

  Attributes:
    num_epochs: An integer representing the number of passes each client
      performs over its entire local dataset.
    batch_size: An integer representing the batch size used when iterating over
      client datasets.
    max_elements: An optional positive integer governing the maximum number of
      examples used by each client. By default, this is set to `None` in which
      case clients use their full dataset. The maximun number of examples is
      taken after the entire local dataset is repeated for the number of passes
      defined by `num_epochs`.
    shuffle_buffer_size: An optional positive integer specifying the shuffle
      buffer size to use. If set to `None`, a default value suitable for the
      task's dataset will be used. If set to `1`, no shuffling occurs.
  """

  num_epochs: int = attrs.field(
      validator=[attrs.validators.instance_of(int), _check_positive],
      converter=int,
  )
  batch_size: int = attrs.field(
      validator=[attrs.validators.instance_of(int), _check_positive],
      converter=int,
  )
  max_elements: Optional[int] = attrs.field(
      default=None,
      validator=[
          attrs.validators.optional(attrs.validators.instance_of(int)),
          attrs.validators.optional(_check_positive),
      ],
  )
  shuffle_buffer_size: Optional[int] = attrs.field(
      default=None,
      validator=[
          attrs.validators.optional(attrs.validators.instance_of(int)),
          attrs.validators.optional(_check_positive),
      ],
  )
