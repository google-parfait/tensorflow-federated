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
"""Classes/functions for statically asserting properties of TFF computations."""

from typing import Optional

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.computation import computation_impl


def _raise_expected_none(
    calls: list[building_blocks.Call], kind: str
) -> Optional[str]:
  assert len(calls) != 0  # pylint: disable=g-explicit-length-test
  msg = 'Expected no {} aggregations, found {}:'.format(kind, len(calls))
  msg += ''.join(('\n\t' + call.compact_representation() for call in calls))
  raise AssertionError(msg)


def assert_contains_secure_aggregation(comp):
  """Asserts that `comp` contains at least one secure aggregation call.

  Args:
    comp: A `tff.Computation`, often a function annotated with
      `tff.federated_computation` or `tff.tf_computation`. Note that polymorphic
      functions (those without the types of their arguments explicitly
      specified) will not yet be `tff.Computation`s.

  Raises:
    AssertionError if `comp` does not contain a secure aggregation call.
    ValueError if `comp` contains a call whose target function cannot be
      identified. This may result from calls to references or other
      indirect structures.
  """
  py_typecheck.check_type(comp, computation_impl.ConcreteComputation)
  comp = comp.to_building_block()
  calls = tree_analysis.find_secure_aggregation_in_tree(comp)
  if len(calls) == 0:  # pylint: disable=g-explicit-length-test
    raise AssertionError(
        'Expected secure aggregation, but none were found in: {}'.format(
            comp.compact_representation()
        )
    )


def assert_not_contains_secure_aggregation(comp):
  """Asserts that `comp` contains no secure aggregation calls.

  Args:
    comp: A `tff.Computation`, often a function annotated with
      `tff.federated_computation` or `tff.tf_computation`. Note that polymorphic
      functions (those without the types of their arguments explicitly
      specified) will not yet be `tff.Computation`s.

  Raises:
    AssertionError if `comp` contains a secure aggregation call.
    ValueError if `comp` contains a call whose target function cannot be
      identified. This may result from calls to references or other
      indirect structures.
  """
  py_typecheck.check_type(comp, computation_impl.ConcreteComputation)
  comp = comp.to_building_block()
  calls = tree_analysis.find_secure_aggregation_in_tree(comp)
  if len(calls) != 0:  # pylint: disable=g-explicit-length-test
    _raise_expected_none(calls, 'secure')


def assert_contains_unsecure_aggregation(comp):
  """Asserts that `comp` contains at least one unsecure aggregation call.

  Args:
    comp: A `tff.Computation`, often a function annotated with
      `tff.federated_computation` or `tff.tf_computation`. Note that polymorphic
      functions (those without the types of their arguments explicitly
      specified) will not yet be `tff.Computation`s.

  Raises:
    AssertionError if `comp` does not contain an unsecure aggregation call.
    ValueError if `comp` contains a call whose target function cannot be
      identified. This may result from calls to references or other
      indirect structures.
  """
  py_typecheck.check_type(comp, computation_impl.ConcreteComputation)
  comp = comp.to_building_block()
  calls = tree_analysis.find_unsecure_aggregation_in_tree(comp)
  if len(calls) == 0:  # pylint: disable=g-explicit-length-test
    raise AssertionError(
        'Expected unsecure aggregation, but none were found in:\n{}'.format(
            comp.compact_representation()
        )
    )


def assert_not_contains_unsecure_aggregation(comp):
  """Asserts that `comp` contains no unsecure aggregation calls.

  Args:
    comp: A `tff.Computation`, often a function annotated with
      `tff.federated_computation` or `tff.tf_computation`. Note that polymorphic
      functions (those without the types of their arguments explicitly
      specified) will not yet be `tff.Computation`s.

  Raises:
    AssertionError if `comp` contains an unsecure aggregation call.
    ValueError if `comp` contains a call whose target function cannot be
      identified. This may result from calls to references or other
      indirect structures.
  """
  py_typecheck.check_type(comp, computation_impl.ConcreteComputation)
  comp = comp.to_building_block()
  calls = tree_analysis.find_unsecure_aggregation_in_tree(comp)
  if len(calls) != 0:  # pylint: disable=g-explicit-length-test
    _raise_expected_none(calls, 'unsecure')
