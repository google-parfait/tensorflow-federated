# Copyright 2023, The TensorFlow Federated Authors.
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
"""Utilities for working with shapes.

The shape of an `Array` may be one of the following:

*   Fully-defined: Has a known number of dimensions and a known size for each
    dimension (e.g. (2, 3)).
*   Partially-defined: Has a known number of dimensions, and an unknown size for
    one or more dimension (e.g. (2, None)).
*   Unknown: Has an unknown number of dimensions (e.g. None).
*   Scalar: Has no dimensions (e.g. ()).
"""

from collections.abc import Sequence
import functools
import operator
from typing import Optional, Union

from tensorflow_federated.proto.v0 import array_pb2
from tensorflow_federated.proto.v0 import data_type_pb2  # pylint: disable=unused-import  # b/330931277


_EmptyTuple = tuple[()]
_ArrayShapeLike = Union[Sequence[Optional[int]], None, _EmptyTuple]

# ArrayShape is the Python representation of the `ArrayShape` protobuf, and is
# the shape of an `Array`.
ArrayShape = Union[tuple[Optional[int], ...], None, _EmptyTuple]


def from_proto(shape_pb: array_pb2.ArrayShape) -> ArrayShape:
  """Returns a `tff.types.ArrayShape` for the `shape_pb`."""
  if shape_pb.unknown_rank:
    return None
  else:
    return tuple(d if d >= 0 else None for d in shape_pb.dim)


def to_proto(shape: ArrayShape) -> array_pb2.ArrayShape:
  """Returns an `ArrayShape` for the `shape`."""
  if shape is not None:
    dims = [d if d is not None else -1 for d in shape]
    return array_pb2.ArrayShape(dim=dims)
  else:
    return array_pb2.ArrayShape(unknown_rank=True)


def is_shape_fully_defined(shape: ArrayShape) -> bool:
  """Returns `True` if `shape` is fully defined, False otherwise.

  Args:
    shape: A `tff.types.ArrayShape`.
  """
  return shape is not None and all(dim is not None for dim in shape)


def is_shape_scalar(shape: ArrayShape) -> bool:
  """Returns `True` if `shape` is scalar, False otherwise.

  Args:
    shape: A `tff.types.ArrayShape`.
  """
  return shape is not None and not shape


def is_compatible_with(target: ArrayShape, other: ArrayShape) -> bool:
  """Returns `True` if `target` is compatible with `other`, otherwise `False`.

  Two shapes are compatible if there exists a fully-defined shape that both
  shapes can represent. For example:

  * `None` is compatible with all shapes.
  * `(None, None)` is compatible with all two-dimensional shapes, and also
  `None`.
  * `(2, None)` is compatible with all two-dimensional shapes with size 2 in the
  0th dimension, and also `(None, None)` and `None`.
  * `(2, 3) is compatible with itself, and also `(32, None)`, `(None, 3]),
  `(None, None)`, and `None`.

  Args:
    target: A `tff.types.ArrayShape`.
    other: Another `tff.types.ArrayShape`.
  """
  if target is None or other is None:
    return True

  if len(target) != len(other):
    return False

  return all(x is None or y is None or x == y for x, y in zip(target, other))


def num_elements_in_shape(shape: ArrayShape) -> Optional[int]:
  """Returns the number of elements in `shape`, or `None` if not fully defined.

  Args:
    shape: A `tff.types.ArrayShape`.
  """
  if is_shape_fully_defined(shape):
    return functools.reduce(operator.mul, shape, 1)
  else:
    return None
