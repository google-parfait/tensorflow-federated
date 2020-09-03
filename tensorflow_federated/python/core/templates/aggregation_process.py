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
"""A `MeasuredProcess` that aggregates values from `CLIENTS` to `SERVER`."""

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.templates import measured_process

_STATE_PARAM_INDEX = 0
_INPUT_PARAM_INDEX = 1


class AggregationProcess(measured_process.MeasuredProcess):
  """A `tff.templates.MeasuredProcess` for aggregations.

  A `tff.templates.AggregationProcess` is a `tff.templates.MeasuredProcess`
  that formalizes the process for aggregation.

  Both `initialize` and `next` properties inherit composition pattern from the
  base class, and must be `tff.Computation`s with the following type signatures:
    initialize: `( -> S@SERVER)`
    next: `(<S@SERVER, V@CLIENTS, *> ->
            <state=S@SERVER, result=V@SERVER, measurements=M@SERVER>)`
  where `*` represents optional other arguments placed at `CLIENTS`.
  """

  def __init__(self, initialize_fn: computation_base.Computation,
               next_fn: computation_base.Computation):
    """Creates a `tff.templates.AggregationProcess`.

    Args:
      initialize_fn: A no-arg `tff.Computation` that creates the initial state
        of the aggregation process.
      next_fn: A `tff.Computation` that defines an iterated function.

    Raises:
      TypeError: `initialize_fn` and `next_fn` are not compatible function
        types, or do not have valid federated type signature.
    """
    # Calling super class __init__ first ensures that
    # next_fn.type_signature.result is a `MeasuredProcessOutput`, make our
    # validation here easier as that must be true.
    super().__init__(initialize_fn, next_fn)
    if (not initialize_fn.type_signature.result.is_federated() or
        initialize_fn.type_signature.result.placement != placements.SERVER):
      raise TypeError(
          f'The return type of initialize_fn must be federated and placed at '
          f'SERVER, but found {initialize_fn.type_signature.result}')

    next_fn_param = next_fn.type_signature.parameter
    next_fn_result = next_fn.type_signature.result
    if not next_fn_param.is_struct() or len(next_fn_param) < 2:
      raise TypeError(f'The next_fn must have at least two input arguments, '
                      f'but has input signature of {next_fn_param}.')

    if (not next_fn_param[_STATE_PARAM_INDEX].is_federated() or
        next_fn_param[_STATE_PARAM_INDEX].placement != placements.SERVER):
      raise TypeError(
          f'The first argument of next_fn must be federated and placed at '
          f'SERVER, but found {next_fn_param[_STATE_PARAM_INDEX]}')
    if (not next_fn_param[_INPUT_PARAM_INDEX].is_federated() or
        next_fn_param[_INPUT_PARAM_INDEX].placement != placements.CLIENTS):
      raise TypeError(
          f'The second argument of next_fn must be federated and placed at '
          f'CLIENTS, but found {next_fn_param[_INPUT_PARAM_INDEX]}')

    if next_fn_result.state.placement != placements.SERVER:
      raise TypeError(
          f'The "state" attribute of return type of next_fn must be placed at '
          f'SERVER, but found {next_fn_result.state}.')
    if next_fn_result.result.placement != placements.SERVER:
      raise TypeError(
          f'The "result" attribute of return type of next_fn must be placed at '
          f'SERVER, but found {next_fn_result.result}.')
    if next_fn_result.measurements.placement != placements.SERVER:
      raise TypeError(
          f'The "measurements" attribute of return type of next_fn must be '
          f'placed at SERVER, but found '
          f'{next_fn_result.measurements}.')

    if (next_fn_param[_INPUT_PARAM_INDEX].member !=
        next_fn_result.result.member):
      raise TypeError(
          f'The second argument of next_fn must be of the same non-federated '
          f'type as the "result" attrubute of the returned structure, but '
          f'instead found: Second argument of next_fn of type: '
          f'{next_fn_param[_INPUT_PARAM_INDEX].member} and "result" attrubute '
          f'of the returned structure of type: '
          f'{next_fn_result.result.member}')

  @property
  def next(self) -> computation_base.Computation:
    """A `tff.Computation` that runs one iteration of the process.

    Its first argument should always be the current state (originally produced
    by the `initialize` attribute), the second argument must be the input placed
    at `CLIENTS`, and the return type must be a structure matching the type
    signature `<state=A@SERVER, result=B@SERVER, measurements=C@SERVER>`.

    Returns:
      A `tff.Computation`.
    """
    return super().next
