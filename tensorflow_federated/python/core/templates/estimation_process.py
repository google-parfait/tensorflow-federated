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

from typing import Optional

import federated_language

from tensorflow_federated.python.common_libs import py_typecheck
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

  def __init__(
      self,
      initialize_fn: federated_language.framework.Computation,
      next_fn: federated_language.framework.Computation,
      report_fn: federated_language.framework.Computation,
      next_is_multi_arg: Optional[bool] = None,
  ):
    """Creates a `tff.templates.EstimationProcess`.

    Args:
      initialize_fn: A no-arg `federated_language.Computation` that returns the
        initial state of the estimation process. Let the type of this state be
        called `S`.
      next_fn: A `federated_language.Computation` that represents the iterated
        function. The first or only argument must match the state type `S`. The
        first or only return value must also match state type `S`.
      report_fn: A `federated_language.Computation` that represents the
        estimation based on state. Its input argument must match the state type
        `S`.
      next_is_multi_arg: An optional boolean indicating that `next_fn` will
        receive more than just the state argument (if `True`) or only the state
        argument (if `False`). This parameter is primarily used to provide
        better error messages.

    Raises:
      TypeError: If `initialize_fn`, `next_fn` and `report_fn` are not
        instances of `federated_language.Computation`.
      TemplateInitFnParamNotEmptyError: If `initialize_fn` has any input
        arguments.
      TemplateStateNotAssignableError: If the `state` returned by either
        `initialize_fn` or `next_fn` is not assignable to the first input
        argument of `next_fn` and `report_fn`.
    """
    super().__init__(
        initialize_fn, next_fn, next_is_multi_arg=next_is_multi_arg
    )

    py_typecheck.check_type(report_fn, federated_language.framework.Computation)
    report_fn_arg_type = report_fn.type_signature.parameter
    if report_fn_arg_type is None or not report_fn_arg_type.is_assignable_from(
        self.state_type
    ):
      raise errors.TemplateStateNotAssignableError(
          'The state type of the process must be assignable to the '
          'input argument of `report_fn`, but the state type is: '
          f'{self.state_type}\n'
          'and the argument of `report_fn` is:\n'
          f'{report_fn_arg_type}'
      )

    self._report_fn = report_fn

  @property
  def report(self) -> federated_language.framework.Computation:
    """A `federated_language.Computation` that computes the current estimate from `state`.

    Given a `state` controlled by this process, computes and returns the most
    recent estimate of the estimated value.

    Returns:
      A `federated_language.Computation`.
    """
    return self._report_fn

  def map(self, map_fn: federated_language.framework.Computation):
    """Applies `map_fn` to the estimate function of the process.

    This method will return a new instance of `EstimationProcess` with the same
    `initailize` and `next` functions, and its `report` function replaced by
    `map_fn(report(state))`.

    Args:
      map_fn: A `federated_language.Computation` to apply to the result of the
        `report` function of the process. Must accept the return type of
        `report`.

    Returns:
      An `EstimationProcess`.

    Raises:
      EstimateNotAssignableError: If the return type of `report` is not
        assignable to the expected input type of `map_fn`.
    """
    py_typecheck.check_type(map_fn, federated_language.framework.Computation)

    estimate_type = self.report.type_signature.result
    map_fn_arg_type = map_fn.type_signature.parameter

    if map_fn_arg_type is None or not map_fn_arg_type.is_assignable_from(
        estimate_type
    ):
      raise EstimateNotAssignableError(
          'The return type of `report` of this process must be '
          'assignable to the input argument of `map_fn`, but '
          f'`report` returns type:\n{estimate_type}\n'
          f'and the argument of `map_fn` is:\n{map_fn_arg_type}'
      )

    transformed_report_fn = federated_language.federated_computation(
        lambda state: map_fn(self.report(state)), self.state_type
    )

    return EstimationProcess(
        initialize_fn=self.initialize,
        next_fn=self.next,
        report_fn=transformed_report_fn,
    )
