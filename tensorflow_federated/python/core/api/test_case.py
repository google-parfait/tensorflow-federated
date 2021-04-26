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
"""Base class for TFF test cases."""
from typing import List, Set

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.impl.types import computation_types


def _filter_node_list_to_ops(nodedefs: List[tf.compat.v1.NodeDef],
                             ops_to_find: Set[str]) -> List[str]:
  found_ops = []
  for node in nodedefs:
    if node.op in ops_to_find:
      found_ops.append(node.op)
  return found_ops


class TestCase(tf.test.TestCase, absltest.TestCase):
  """Base class for TensroFlow Federated tests."""

  def setUp(self):
    super().setUp()
    tf.keras.backend.clear_session()

  def assert_type_assignable_from(self, target_type, source_type):
    # Reraise the exception outside of `except` so as to avoid setting
    # the `.__cause__` on the final error (repeating everything).
    message = None
    try:
      target_type.check_assignable_from(source_type)
    except computation_types.TypeNotAssignableError as e:
      message = e.message
    if message is not None:
      self.fail(message)

  def assert_types_equivalent(self, first_type, second_type):
    message = None
    try:
      first_type.check_equivalent_to(second_type)
    except computation_types.TypesNotEquivalentError as e:
      message = e.message
    if message is not None:
      self.fail(message)

  def assert_types_identical(self, first_type, second_type):
    message = None
    try:
      first_type.check_identical_to(second_type)
    except computation_types.TypesNotIdenticalError as e:
      message = e.message
    if message is not None:
      self.fail(message)

  def assert_type_string(self, type_signature, expected_string):
    self.assertEqual(type_signature.compact_representation(), expected_string)

  def assert_nested_struct_eq(self, x, y):
    """Asserts that nested structures 'x' and 'y' are the same.

    Args:
      x: One nested structure.
      y: Another nested structure.

    Raises:
      ValueError: if the structures are not the same.
    """
    try:
      tf.nest.assert_same_structure(x, y)
    except ValueError:
      self.fail('Expected structures to have the same shape.')
    xl = tf.nest.flatten(x)
    yl = tf.nest.flatten(y)
    if len(xl) != len(yl):
      self.fail('The sizes of structures {} and {} mismatch.'.format(
          str(len(xl)), str(len(yl))))
    for xe, ye in zip(xl, yl):
      if xe != ye:
        self.fail('Mismatching elements {} and {}.'.format(str(xe), str(ye)))

  def assert_graph_contains_ops(self, graphdef: tf.compat.v1.GraphDef,
                                ops: Set[str]):
    found_ops = _filter_node_list_to_ops(graphdef.node, ops)
    remaining_unfound_ops = set(ops) - set(found_ops)
    for function in graphdef.library.function:
      ops_in_function = _filter_node_list_to_ops(function.node_def,
                                                 remaining_unfound_ops)
      remaining_unfound_ops = remaining_unfound_ops - set(ops_in_function)
    if remaining_unfound_ops:
      self.fail(f'Expected to encounter the ops {ops}, but failed to find '
                f'{remaining_unfound_ops}.')

  def assert_graph_does_not_contain_ops(self, graphdef: tf.compat.v1.GraphDef,
                                        forbidden_ops: Set[str]):
    found_ops = _filter_node_list_to_ops(graphdef.node, forbidden_ops)
    if found_ops:
      self.fail(
          f'Ops {forbidden_ops} are forbidden, but encountered {found_ops} '
          'in GraphDef.')
    for function in graphdef.library.function:
      found_ops = _filter_node_list_to_ops(function.node_def, forbidden_ops)
      if found_ops:
        self.fail(
            f'Ops {forbidden_ops} are forbidden, but encountered {found_ops} in GraphDef.'
        )


def main():
  """Runs all unit tests with TF 2.0 features enabled.

  This function should only be used if TensorFlow code is being tested.
  """
  tf.test.main()
