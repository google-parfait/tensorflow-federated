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
"""Defines a template for a process that can compute an estimate."""

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process


class EstimateNotAssignableError(TypeError):
  """`TypeError` for estimate not being assignable to expected type."""
  pass


class EstimationProcess(iterative_process.IterativeProcess):
  """A stateful process that can compute an estimate of some value.

  This class inherits the constraints documented by
  `tff.templates.IterativeProcess`.

  A `tff.templates.EstimationProcess` is an `tff.templates.IterativeProcess`
  that in addition to the `initialize` and `next` functions, has a
  `report` function that returns the result of some computation based on the
  state of the process. The argument of `report` must be of the same type as the
  state, that is, the type of object returned by `initialize`.
  """

  def __init__(self, initialize_fn: computation_base.Computation,
               next_fn: computation_base.Computation,
               report_fn: computation_base.Computation):
    """Creates a `tff.templates.EstimationProcess`.

    Args:
      initialize_fn: A no-arg `tff.Computation` that creates the initial state
        of the computation.
      next_fn: A `tff.Computation` that represents the iterated function. If
        `initialize_fn` returns a type `T`, then `next_fn` must either return a
        type `U` which is compatible with `T` or multiple values where the first
        type is `U`, and accept either a single argument of type `U` or multiple
        arguments where the first argument must be of type `U`.
      report_fn: A `tff.Computation` that represents the estimation based on
        state. Its input argument must be assignable from return type of
        `initialize_fn`.

    Raises:
      TypeError: If `initialize_fn`, `next_fn` and `report_fn` are not
        instances of `tff.Computation`.
      TemplateInitFnParamNotEmptyError: If `initialize_fn` has any input
        arguments.
      TemplateStateNotAssignableError: If the `state` returned by either
        `initialize_fn` or `next_fn` is not assignable to the first input
        argument of `next_fn` and `report_fn`.
    """
    super().__init__(initialize_fn, next_fn)

    py_typecheck.check_type(report_fn, computation_base.Computation)
    report_fn_arg_type = report_fn.type_signature.parameter
    if not report_fn_arg_type.is_assignable_from(self.state_type):
      raise errors.TemplateStateNotAssignableError(
          f'The state type of the process must be assignable to the '
          f'input argument of `report_fn`, but the state type is: '
          f'{self.state_type}\n'
          f'and the argument of `report_fn` is:\n'
          f'{report_fn_arg_type}')

    self._report_fn = report_fn

  @property
  def report(self) -> computation_base.Computation:
    """A `tff.Computation` that computes the current estimate from `state`.

    Given a `state` controlled by this process, computes and returns the most
    recent estimate of the estimated value.

    Returns:
      A `tff.Computation`.
    """
    return self._report_fn

  def map(self, map_fn: computation_base.Computation):
    """Applies `map_fn` to the estimate function of the process.

    This method will return a new instance of `EstimationProcess` with the same
    `initailize` and `next` functions, and its `report` function replaced by
    `map_fn(report(state))`.

    Args:
      map_fn: A `tff.Computation` to apply to the result of the `report`
        function of the process. Must accept the return type of `report`.

    Returns:
      An `EstimationProcess`.

    Raises:
      EstimateNotAssignableError: If the return type of `report` is not
        assignable to the expected input type of `map_fn`.
    """
    py_typecheck.check_type(map_fn, computation_base.Computation)

    estimate_type = self.report.type_signature.result
    map_fn_arg_type = map_fn.type_signature.parameter

    if not map_fn_arg_type.is_assignable_from(estimate_type):
      raise EstimateNotAssignableError(
          f'The return type of `report` of this process must be '
          f'assignable to the input argument of `map_fn`, but '
          f'`report` returns type:\n{estimate_type}\n'
          f'and the argument of `map_fn` is:\n{map_fn_arg_type}')

    try:
      transformed_report_fn = computations.tf_computation(
          lambda state: map_fn(self.report(state)), self.state_type)
    except TypeError:
      # Raised if the computation operates in federated types. However, there is
      # currently no way to distinguish these using the public API.
      transformed_report_fn = computations.federated_computation(
          lambda state: map_fn(self.report(state)), self.state_type)

    return EstimationProcess(
        initialize_fn=self.initialize,
        next_fn=self.next,
        report_fn=transformed_report_fn)
