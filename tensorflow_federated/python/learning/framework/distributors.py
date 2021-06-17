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
"""Abstractions for distributors.

This module is a minimal stab at structure which will probably live in
`tff.distributors` and `tff.templates` later on.
"""

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process


class DistributionProcess(measured_process.MeasuredProcess):
  """A stateful process that distributes values.

  A `DistributionProcess` is a `tff.templates.MeasuredProcess` that formalizes
  the type signature of `initialize_fn` and `next_fn` for distribution.

  The `initialize_fn` and `next_fn` must have the following type signatures:
  ```
    - initialize_fn: ( -> S@SERVER)
    - next_fn: (<S@SERVER, V@SERVER> ->
                <state=S@SERVER, result=V'@CLIENTS, measurements=M@SERVER>)
  ```

  `DistributionProcess` requires `next_fn` with a second input argument, which
  is a value placed at `SERVER` and to be distributed to `CLIENTS`.

  The `result` field of the returned `tff.templates.MeasuredProcessOutput` must
  be placed at `CLIENTS`. Its type singature, `V'`, need not be the same as the
  type signature of the second input argument, `V`. Note these will be
  equivalent for a number of implementations of this process, though.
  """

  def __init__(self, initialize_fn, next_fn):
    super().__init__(initialize_fn, next_fn, next_is_multi_arg=True)

    if not initialize_fn.type_signature.result.is_federated():
      raise errors.TemplateNotFederatedError(
          f'Provided `initialize_fn` must return a federated type, but found '
          f'return type:\n{initialize_fn.type_signature.result}\nTip: If you '
          f'see a collection of federated types, try wrapping the returned '
          f'value in `tff.federated_zip` before returning.')
    next_types = (
        structure.flatten(next_fn.type_signature.parameter) +
        structure.flatten(next_fn.type_signature.result))
    if not all([t.is_federated() for t in next_types]):
      offending_types = '\n- '.join(
          [t for t in next_types if not t.is_federated()])
      raise errors.TemplateNotFederatedError(
          f'Provided `next_fn` must be a *federated* computation, that is, '
          f'operate on `tff.FederatedType`s, but found\n'
          f'next_fn with type signature:\n{next_fn.type_signature}\n'
          f'The non-federated types are:\n {offending_types}.')

    if initialize_fn.type_signature.result.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          f'The state controlled by an `DistributionProcess` must be placed at '
          f'the SERVER, but found type: {initialize_fn.type_signature.result}.')
    # Note that state of next_fn being placed at SERVER is now ensured by the
    # assertions in base class which would otherwise raise
    # TemplateStateNotAssignableError.

    next_fn_param = next_fn.type_signature.parameter
    next_fn_result = next_fn.type_signature.result
    if not next_fn_param.is_struct():
      raise errors.TemplateNextFnNumArgsError(
          f'The `next_fn` must have exactly two input arguments, but found '
          f'the following input type which is not a Struct: {next_fn_param}.')
    if len(next_fn_param) != 2:
      next_param_str = '\n- '.join([str(t) for t in next_fn_param])
      raise errors.TemplateNextFnNumArgsError(
          f'The `next_fn` must have exactly two input arguments, but found '
          f'{len(next_fn_param)} input arguments:\n{next_param_str}')
    if next_fn_param[1].placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          f'The second input argument of `next_fn` must be placed at SERVER '
          f'but found {next_fn_param[1]}.')

    if next_fn_result.result.placement != placements.CLIENTS:
      raise errors.TemplatePlacementError(
          f'The "result" attribute of return type of `next_fn` must be placed '
          f'at CLIENTS, but found {next_fn_result.result}.')
    if next_fn_result.measurements.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          f'The "measurements" attribute of return type of `next_fn` must be '
          f'placed at SERVER, but found {next_fn_result.measurements}.')
