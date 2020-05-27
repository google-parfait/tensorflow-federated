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
"""Tests for static_assert.py."""

from absl.testing import absltest

import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.test import static_assert

tf.compat.v1.enable_v2_behavior()


@computations.federated_computation
def no_aggregation():
  return ()


@computations.federated_computation
def secure_aggregation():
  data_at_clients = intrinsics.federated_value(1, placements.CLIENTS)
  bitwidth = 1
  return intrinsics.federated_secure_sum(data_at_clients, bitwidth)


@computations.federated_computation
def unsecure_aggregation():
  data_at_clients = intrinsics.federated_value(1, placements.CLIENTS)
  return intrinsics.federated_sum(data_at_clients)


@computations.federated_computation
def secure_and_unsecure_aggregation():
  return (secure_aggregation, unsecure_aggregation)


class AssertContainsSecAggTest(absltest.TestCase):

  def test_fails_on_noagg(self):
    with self.assertRaises(AssertionError):
      static_assert.assert_contains_secure_aggregation(no_aggregation)

  def test_passes_on_secagg(self):
    static_assert.assert_contains_secure_aggregation(secure_aggregation)

  def test_fails_on_unsecagg(self):
    with self.assertRaises(AssertionError):
      static_assert.assert_contains_secure_aggregation(unsecure_aggregation)

  def test_passes_on_bothagg(self):
    static_assert.assert_contains_secure_aggregation(
        secure_and_unsecure_aggregation)


class AssertNotContainsSecAggTest(absltest.TestCase):

  def test_passes_on_noagg(self):
    static_assert.assert_not_contains_secure_aggregation(no_aggregation)

  def test_fails_on_secagg(self):
    with self.assertRaises(AssertionError):
      static_assert.assert_not_contains_secure_aggregation(secure_aggregation)

  def test_passes_on_unsecagg(self):
    static_assert.assert_not_contains_secure_aggregation(unsecure_aggregation)

  def test_fails_on_bothagg(self):
    with self.assertRaises(AssertionError):
      static_assert.assert_not_contains_secure_aggregation(
          secure_and_unsecure_aggregation)


class AssertContainsUnsecAggTest(absltest.TestCase):

  def test_fails_on_noagg(self):
    with self.assertRaises(AssertionError):
      static_assert.assert_contains_unsecure_aggregation(no_aggregation)

  def test_fails_on_secagg(self):
    with self.assertRaises(AssertionError):
      static_assert.assert_contains_unsecure_aggregation(secure_aggregation)

  def test_passes_on_unsecagg(self):
    static_assert.assert_contains_unsecure_aggregation(unsecure_aggregation)

  def test_passes_on_bothagg(self):
    static_assert.assert_contains_unsecure_aggregation(
        secure_and_unsecure_aggregation)


class AssertNotContainsUnsecAggTest(absltest.TestCase):

  def test_passes_on_noagg(self):
    static_assert.assert_not_contains_unsecure_aggregation(no_aggregation)

  def test_passes_on_secagg(self):
    static_assert.assert_not_contains_unsecure_aggregation(secure_aggregation)

  def test_fails_on_unsecagg(self):
    with self.assertRaises(AssertionError):
      static_assert.assert_not_contains_unsecure_aggregation(
          unsecure_aggregation)

  def test_fails_on_bothagg(self):
    with self.assertRaises(AssertionError):
      static_assert.assert_not_contains_unsecure_aggregation(
          secure_and_unsecure_aggregation)


if __name__ == '__main__':
  absltest.main()
