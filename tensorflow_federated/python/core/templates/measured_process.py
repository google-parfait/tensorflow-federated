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
"""Define a template for a stateful process that produces metrics."""

from typing import Optional

import attr

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process


@attr.s(frozen=True, eq=False, slots=True)
class MeasuredProcessOutput:
  """A structure containing the output of a `MeasuredProcess.next` computation.

  Attributes:
    state: A structure that will be passed to invocation of
      `MeasuredProcess.next`. Not intended for inspection externally, contains
      implementation details of the process.
    result: The result of the process given the current input and state. Using
      the rules of composition, either passed to input arguments of chained a
      `MeasuredProcess`, or concatenated with outputs of parallel
      `MeasuredProcess`es.
    measurements: Metrics derived from the computation of `result`. Intended for
      surfacing values to track the progress of a process that are not sent to
      chained `MeasuredProcess`es.
  """
  state = attr.ib()
  result = attr.ib()
  measurements = attr.ib()


# The type signature of the result of MeasuredProcess must be a named tuple with
# the following names in the same order.
_RESULT_FIELD_NAMES = [f.name for f in attr.fields(MeasuredProcessOutput)]


class MeasuredProcess(iterative_process.IterativeProcess):
  """A stateful process that produces metrics.

  This class inherits the constraints documented by
  `tff.templates.IterativeProcess`.

  A `tff.templates.MeasuredProcess` is a `tff.templates.IterativeProcess` whose
  `next` computation returns a `tff.templates.MeasuredProcessOutput`.

  Unlike `tff.templates.IterativeProcess`, the more generic but less-defined
  template, arbitrary `tff.templates.MeasuredProcess`es can be composed together
  as follows:

  *Guidance for Composition*
  Two `MeasuredProcess`es _F(x)_ and _G(y)_ can be composed into a new
  `MeasuredProcess` called _C_ with the following properties:
    - `C.state` is the concatenation `<F.state, G.state>`.
    - `C.next(C.state, x).result ==
       G.next(G.state, F.next(F.state, x).result).result`
    - `C.measurements` is the concatenation `<F.measurements, G.measurements>`.

  The resulting composition _C_ would have the following type signatures:
    initialize: `( -> <F.initialize, G.initialize>)`
    next: `(<<F.state, G.state>, F.input> -> <state=<F.state, G.State>,
      result=G.result, measurements=<F.measurements, G.measurements>)`

  Note that the guidance for composition is not strict and details are allowed
  to differ.
  """

  def __init__(self,
               initialize_fn: computation_base.Computation,
               next_fn: computation_base.Computation,
               next_is_multi_arg: Optional[bool] = None):
    """Creates a `tff.templates.MeasuredProcess`.

    Args:
      initialize_fn: A no-arg `tff.Computation` that returns the initial state
        of the measured process. Let the type of this state be called `S`.
      next_fn: A `tff.Computation` that represents the iterated function. The
        first or only argument must match the state type `S`. The return value
        must be a `MeasuredProcessOutput` whose `state` member matches the
        state type `S`.
      next_is_multi_arg: An optional boolean indicating that `next_fn` will
        receive more than just the state argument (if `True`) or only the state
        argument (if `False`). This parameter is primarily used to provide
        better error messages.

    Raises:
      TypeError: If `initialize_fn` and `next_fn` are not instances of
        `tff.Computation`.
      TemplateInitFnParamNotEmptyError: If `initialize_fn` has any input
        arguments.
      TemplateStateNotAssignableError: If the `state` returned by either
        `initialize_fn` or `next_fn` is not assignable to the first input
        argument of `next_fn`.
      TemplateNotMeasuredProcessOutputError: If `next_fn` does not return a
        `MeasuredProcessOutput`.
    """
    super().__init__(initialize_fn, next_fn, next_is_multi_arg)
    next_result_type = next_fn.type_signature.result
    if not (isinstance(next_result_type, computation_types.StructWithPythonType)
            and next_result_type.python_container is MeasuredProcessOutput):
      raise errors.TemplateNotMeasuredProcessOutputError(
          f'The `next_fn` of a `MeasuredProcess` must return a '
          f'`MeasuredProcessOutput` object, but returns {next_result_type!r}')

    # Perform a more strict type check on state than the base class. Base class
    # ensures that state returned by initialize_fn is accepted as input argument
    # of next_fn, and that this is in the returned structure. For
    # MeasuredProcess, this explicitly needs to be in the state attribute. See
    # `test_measured_process_output_as_state_raises` for an example.
    state_type = self.state_type
    if not state_type.is_assignable_from(next_fn.type_signature.result.state):
      raise errors.TemplateStateNotAssignableError(
          f'The state attrubute of returned MeasuredProcessOutput must be '
          f'assignable to its first input argument, but found\n'
          f'`next_fn` which returns MeasuredProcessOutput with state attribute '
          f'of type:\n{next_result_type}\n'
          f'which does not match its first input argument:\n{state_type}')

  @property
  def next(self) -> computation_base.Computation:
    """A `tff.Computation` that runs one iteration of the process.

    Its first argument should always be the current state (originally produced
    by `tff.templates.MeasuredProcess.initialize`), and the return type must be
    a `tff.templates.MeasuredProcessOutput`.

    Returns:
      A `tff.Computation`.
    """
    return super().next
