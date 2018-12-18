# Copyright 2018, The TensorFlow Federated Authors.
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
"""General purpose test utils for TFF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
import tensorflow as tf

from tensorflow.python import tf2


class TffTestCase(tf.test.TestCase, absltest.TestCase):
  """A subclass of tf.test.TestCase that enables select TF 2.0 features.

  N.B. This TestCase should only be used if TensorFlow code is actually
  being tested; use absltest.TestCase otherwise.

  Eventually, we may fully enable TF 2.0 here.
  """

  def __init__(self, *args, **kwargs):
    super(TffTestCase, self).__init__(*args, **kwargs)
    self._setup_tf()

  def setUp(self):
    super(TffTestCase, self).setUp()
    self._setup_tf()

  def _setup_tf(self):
    tf.enable_resource_variables()  # Required since we use defuns.
    tf2.enable()  # Switches TensorArrayV2 and control flow V2
    tf.enable_v2_tensorshape()
