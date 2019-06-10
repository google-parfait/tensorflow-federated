# Lint as: python3
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
"""A base Python interface for values embedded in executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow_federated.python.core.api import typed_object


@six.add_metaclass(abc.ABCMeta)
class ExecutorValue(typed_object.TypedObject):
  """Represents the abstract interface for values embedded within executors.

  The embedded values may represent computations in-flight that may materialize
  in the future or fail before they materialize.
  """

  # TODO(b/134543154): Populate this with additional abstract properties to
  # reflect asynchrony, failures, etc.

  pass
