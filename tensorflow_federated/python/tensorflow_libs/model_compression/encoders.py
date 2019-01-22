# Copyright 2019, The TensorFlow Federated Authors.
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
"""Classes responsible for composing encoding stages."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six

from tensorflow_federated.python.tensorflow_libs.model_compression.core import encoding_stage


class Encoder(object):
  """Placeholder for the implementation of the `Encoder` class."""

  def __init__(self, stage, children):
    """Constructor of the `Encoder`.

    If `stage` is an `EncodingStageInterface`, the constructor will wrap it as
    an `AdaptiveEncodingStageInterface` with no state.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`.
      children: A dictionary of `Encoder` objects.
    """
    stage = encoding_stage.as_adaptive_encoding_stage(stage)
    self.stage = stage
    self.children = children
    super(Encoder, self).__init__()


class EncoderComposer(object):
  """Class for composing and creating `Encoder`.

  This class is a utility for separating the creation of the `Encoder` from its
  intended functionality. Each instance of `EncoderComposer` corresponds to a
  node in a tree of encoding stages to be used for encoding.

  The `make` method converts this object to an `Encoder`, which exposes the
  encoding functionality.
  """

  def __init__(self, stage):
    """Constructor for the `EncoderComposer`.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`.
    """
    if not isinstance(stage, (encoding_stage.EncodingStageInterface,
                              encoding_stage.AdaptiveEncodingStageInterface)):
      raise TypeError('The input argument is %s but must be an instance of '
                      'EncodingStageInterface or of '
                      'AdaptiveEncodingStageInterface' % type(stage))
    self._stage = stage
    self._children = {}
    super(EncoderComposer, self).__init__()

  def add_child(self, stage, key):
    """Adds a child node.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`.
      key: `string`, specifying the output key of the encoding stage controlled
        by this object, to be further encoded by `stage`.

    Returns:
      An `EncoderComposer`, the newly created child node.

    Raises:
      KeyError: If `key` is not a compressible tensor of the encoding stage
        controlled by this object, or if it already has a child.
    """
    if key not in self._stage.compressible_tensors_keys:
      raise KeyError('The specified key is either not compressible or not '
                     'returned by the current encoding stage.')
    if key in self._children:
      raise KeyError('The specified key is already used.')
    new_builder = EncoderComposer(stage)
    self._children[key] = new_builder
    return new_builder

  def add_parent(self, stage, key):
    """Adds a parent node.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`.
      key: `string`, specifying the output key of `stage` to be further encoded
        by the encoding stage controlled by this object.

    Returns:
      An `EncoderComposer`, the newly created parent node.

    Raises:
      KeyError: If `key` is not a compressible tensor of `stage`.
    """
    if key not in stage.compressible_tensors_keys:
      raise KeyError('The specified key is either not compressible or not '
                     'returned by the encoding stage.')
    new_builder = EncoderComposer(stage)
    new_builder._children[key] = self  # pylint: disable=protected-access
    return new_builder

  def make(self):
    """Recursively creates the `Encoder`.

    This method also recursively creates all children, but not parents.

    Returns:
      An `Encoder` composing the encoding stages.
    """
    children = {}
    for k, v in six.iteritems(self._children):
      children[k] = v.make()
    return Encoder(self._stage, children)
