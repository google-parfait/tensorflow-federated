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
"""Defines a template for stateful processes used for learning-oriented tasks."""

import attr

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process


class LearningProcessPlacementError(TypeError):
  """Raises when a learning process does not have expected placements."""
  pass


class LearningProcessSequenceTypeError(TypeError):
  """Raises when a learning process does not have the expected sequence type."""


class LearningProcessOutputError(TypeError):
  """Raises when a learning process does not have the expected output type."""


class ReportFnTypeSignatureError(TypeError):
  """Raises when the type signature `report_fn` is not correct."""


@attr.s(frozen=True, eq=False, slots=True)
class LearningProcessOutput:
  """A structure containing the output of a `LearningProcess.next` computation.

  Attributes:
    state: A structure that will be passed to invocation of
      `LearningProcess.next`. Not intended for inspection externally, contains
      implementation details of the process.
    metrics: Metrics derived from the applying the process to a given state and
      input. This output is intended for surfacing values to better understand
      the progress of a learning process, and is not directly intended to be
      used as input to `LearningProcess.next`. For example, this output may
      include classical metrics surrounding the quality of a machine learning
      model (eg. the average accuracy of a model across the participating
      clients), or operational metrics concerning systems-oriented information
      (eg. timing information and the amount of communication occurring between
      clients and server).
  """
  state = attr.ib()
  metrics = attr.ib()


class LearningProcess(iterative_process.IterativeProcess):
  """A stateful process for learning tasks that produces metrics.

  This class inherits the constraints documented by
  `tff.templates.IterativeProcess`, including an `initialize` and `next`
  attribute. The `LearningProcess` also contains an additional `report`
  attribute.

  All of `initialize`, `next` and `report`  must be `tff.Computation`s, with the
  following type signatures:
    - initialize: `( -> S@SERVER)`
    - next: `(<S@SERVER, {D*}@CLIENTS> -> <state=S@SERVER, metrics=M@SERVER>)`
    - report: `(S -> R)`
  where `{D*}@CLIENTS` represents the sequence of data at a client, with `D`
  denoting the type of a single member of that sequence, and `R` representing
  the (unplaced) output type of the `report` function.

  For example, given a LearningProcess `process` and client data `data`, we
  could call the following to initialize, update the state three times, and get
  a report of the resulting state:
  ```
  state = process.initialize()
  for _ in range(3):
    state, metrics = process.next(state, data)
  report = process.report(state)
  """

  def __init__(self, initialize_fn: computation_base.Computation,
               next_fn: computation_base.Computation,
               report_fn: computation_base.Computation):
    """Creates a `tff.templates.AggregationProcess`.

    Args:
      initialize_fn: A no-arg `tff.Computation` that creates the initial state
        of the learning process.
      next_fn: A `tff.Computation` that defines an iterated function. Given that
        `initialize_fn` returns a type `S@SERVER`, the `next_fn` must return a
        `LearningProcessOutput` where the `state` attribute matches the type
        `S@SERVER`, and accepts two argument of types `S@SERVER` and
        `{D*}@CLIENTS`.
     report_fn: A `tff.Computation` that accepts an input `S` where the output
       of `initialize_fn` is of type `S@SERVER`. This computation is used to
       create a representation of the state that can be used for downstream
       tasks without requiring access to the entire server state. For example,
       `report_fn` could be used to extract model weights for computing metrics
       on held-out data.

    Raises:
      TypeError: If `initialize_fn` and `next_fn` are not instances of
        `tff.Computation`.
      TemplateInitFnParamNotEmptyError: If `initialize_fn` has any input
        arguments.
      TemplateStateNotAssignableError: If the `state` returned by either
        `initialize_fn` or `next_fn` is not assignable to the first input
        argument of `next_fn`.
      TemplateNextFnNumArgsError: If `next_fn` does not have at exactly two
        input arguments.
      LearningProcessPlacementError: If the placements of `initialize_fn` and
        `next_fn` do not match the expected type placements.
      LearningProcessOutputError: If `next_fn` does not return a
        `LearningProcessOutput`.
      LearningProcessSequenceTypeError: If the second argument to `next_fn` is
        not a sequence type.
    """
    super().__init__(initialize_fn, next_fn)

    init_fn_result = initialize_fn.type_signature.result
    if init_fn_result.placement != placements.SERVER:
      raise LearningProcessPlacementError(
          f'The result of `initialize_fn` must be placed at `SERVER` but found '
          f'placement {init_fn_result.placement}.')

    next_result_type = next_fn.type_signature.result
    if not (isinstance(next_result_type, computation_types.StructWithPythonType)
            and next_result_type.python_container is LearningProcessOutput):
      raise LearningProcessOutputError(
          f'The `next_fn` of a `LearningProcess` must return a '
          f'`LearningProcessOutput` object, but returns {next_result_type!r}')

    # We perform a more strict type check on the inputs to `next_fn` than in the
    # base class.
    next_fn_param = next_fn.type_signature.parameter
    if not next_fn_param.is_struct() or len(next_fn_param) != 2:
      raise errors.TemplateNextFnNumArgsError(
          f'The `next_fn` must have two input arguments, but found an input '
          f'of type {next_fn_param}.')
    if next_fn_param[1].placement != placements.CLIENTS:
      raise LearningProcessPlacementError(
          f'The second input argument of `next_fn` must be placed at `CLIENTS`,'
          f' but found placement {next_fn_param[1].placement}.')
    if not next_fn_param[1].member.is_sequence():
      raise LearningProcessSequenceTypeError(
          f'The member type of the second input argument to `next_fn` must be a'
          f' `tff.SequenceType` but found {next_fn_param[1].member} instead.')

    next_fn_result = next_fn.type_signature.result
    if next_fn_result.metrics.placement != placements.SERVER:
      raise LearningProcessPlacementError(
          f'The result of `next_fn` must be placed at `SERVER` but found '
          f'placement {next_fn_result.metrics.placement} for `metrics`.')

    py_typecheck.check_type(report_fn, computation_base.Computation)

    report_fn_type = report_fn.type_signature
    if report_fn_type.is_federated():
      raise LearningProcessPlacementError(
          f'The `report_fn` must not be a federated computation, '
          f'but found `report_fn` with type signature:\n'
          f'{report_fn_type}')

    report_fn_param = report_fn.type_signature.parameter
    state_type_without_placement = initialize_fn.type_signature.result.member
    if not report_fn_param.is_assignable_from(state_type_without_placement):
      raise ReportFnTypeSignatureError(
          f'The input type of `report_fn` must be assignable from '
          f'the member type of the output of `initialize_fn`, but found input '
          f'type {report_fn_param}, which is not assignable from '
          f'{state_type_without_placement}.')

    self._report_fn = report_fn

  @property
  def initialize(self) -> computation_base.Computation:
    """A `tff.Computation` that initializes the process.

    This computation must have no input arguments, and its output must be the
    initial state of the iterative process, placed at `SERVER`.

    Returns:
      A `tff.Computation`.
    """
    return super().initialize

  @property
  def next(self) -> computation_base.Computation:
    """A `tff.Computation` that runs one iteration of the process.

    The first argument of this computation should always be the current state
    (originally produced by the `initialize` attribute), the second argument
    must be a `tff.SequenceType` placed at `CLIENTS`. The return type must be
    a `LearningProcessOutput`, with each field placed at `SERVER`.

    Returns:
      A `tff.Computation`.
    """
    return super().next

  @property
  def report(self) -> computation_base.Computation:
    """A `tff.Computation` that creates a representation of the process state.

    This computation should accept the unplaced member of the current state of
    the process (originally produced by the `initialize` attribute), and returns
    an unplaced representation of that state.

    Returns:
      A `tff.Computation`.
    """
    return self._report_fn
