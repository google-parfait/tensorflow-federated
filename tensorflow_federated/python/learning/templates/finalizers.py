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
"""Abstractions for finalization in learning algorithms."""

from typing import Optional

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.templates import hparams_base


class FinalizerResultTypeError(TypeError):
  """`TypeError` for finalizer not updating model weights as expected."""


class FinalizerProcess(measured_process.MeasuredProcess):
  """A stateful process for finalization of a round of training.

  A `FinalizerProcess` is a `tff.templates.MeasuredProcess` that formalizes the
  type signature of `initialize_fn` and `next_fn` for the work performed by
  server in a learning process after aggregating model updates from clients.
  """

  def __init__(
      self,
      initialize_fn: computation_base.Computation,
      next_fn: computation_base.Computation,
      *,
      get_hparams_fn: Optional[computation_base.Computation] = None,
      set_hparams_fn: Optional[computation_base.Computation] = None,
  ):
    """Initializes a `FinalizerProcess`.

    The `initialize_fn` and `next_fn` must have the following type signatures:

    ```
      - initialize_fn: ( -> S@SERVER)
      - next_fn: (<S@SERVER,
                   A@SERVER,
                   B@SERVER>
                  ->
                  <state=S@SERVER,
                   result=A@SERVER,
                   measurements=M@SERVER>)
    ```

    Here, `A` represents the server parameters to be updated, while the `B`
    represents an update to the parameter `A`.

    The `result` field of the returned `tff.templates.MeasuredProcessOutput`
    must be placed at `SERVER`, be of type matching that of second input
    argument (`A`) and represents the updated ("finalized") model parameters.

    If provided, the `get_hparams_fn` and `set_hparams_fn` must be non-federated
    computations with the following type signatures:

    ```
      - get_hparams_fn: (S -> H)
      - set_hparams_fn: (<S, H> -> S)
    ```

    Here, `S` must match the state `S` of `initialize_fn` and `next_fn`, and `H`
    represents the hyperparameter type.

    Args:
      initialize_fn: A `tff.Computation` matching the criteria above.
      next_fn: A `tff.Computation` matching the criteria above.
      get_hparams_fn: An optional `tff.Computation` matching the criteria above.
        If not provided, this defaults to a computation that returns an empty
        ordred dictionary, regardless of the contents of the state.
      set_hparams_fn: An optional `tff.Computation` matching the criteria above.
        If not provided, this defaults to a pass-through computation, that
        returns the input state regardless of the hparams passed in.

    Raises:
      TemplateNotFederatedError: If any of the federated computations provided
        do not return a federated type.
      TemplateNextFnNumArgsError: If the `next_fn` has an incorrect number
        of arguments.
      TemplatePlacementError: If any of the federated computations have an
        incorrect placement.
      FinalizerResultTypeError: If the second output of `next_fn` does not meet
        the criteria outlined above.
      GetHparamsTypeError: If the type signature of `get_hparams_fn` does not
        meet the criteria above.
      SetHparamsTypeError: If the type signature of `set_hparams_fn` does not
        meet the criteria above.
    """
    super().__init__(initialize_fn, next_fn, next_is_multi_arg=True)

    if not initialize_fn.type_signature.result.is_federated():
      raise errors.TemplateNotFederatedError(
          'Provided `initialize_fn` must return a federated type, but found '
          f'return type:\n{initialize_fn.type_signature.result}\nTip: If you '
          'see a collection of federated types, try wrapping the returned '
          'value in `tff.federated_zip` before returning.'
      )
    next_types = structure.flatten(
        next_fn.type_signature.parameter
    ) + structure.flatten(next_fn.type_signature.result)
    if not all([t.is_federated() for t in next_types]):
      offending_types = '\n- '.join(
          [t for t in next_types if not t.is_federated()]
      )
      raise errors.TemplateNotFederatedError(
          'Provided `next_fn` must be a *federated* computation, that is, '
          'operate on `tff.FederatedType`s, but found\n'
          f'next_fn with type signature:\n{next_fn.type_signature}\n'
          f'The non-federated types are:\n {offending_types}.'
      )

    if initialize_fn.type_signature.result.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          'The state controlled by an `FinalizerProcess` must be placed at '
          f'the SERVER, but found type: {initialize_fn.type_signature.result}.'
      )
    # Note that state of next_fn being placed at SERVER is now ensured by the
    # assertions in base class which would otherwise raise
    # TemplateStateNotAssignableError.

    next_fn_param = next_fn.type_signature.parameter
    if not next_fn_param.is_struct():
      raise errors.TemplateNextFnNumArgsError(
          'The `next_fn` must have exactly two input arguments, but found '
          f'the following input type which is not a Struct: {next_fn_param}.'
      )
    if len(next_fn_param) != 3:
      next_param_str = '\n- '.join([str(t) for t in next_fn_param])
      raise errors.TemplateNextFnNumArgsError(
          'The `next_fn` must have exactly three input arguments, but found '
          f'{len(next_fn_param)} input arguments:\n{next_param_str}'
      )
    model_weights_param = next_fn_param[1]
    update_from_clients_param = next_fn_param[2]
    if model_weights_param.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          'The second input argument of `next_fn` must be placed at SERVER '
          f'but found {model_weights_param}.'
      )
    if update_from_clients_param.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          'The third input argument of `next_fn` must be placed at SERVER '
          f'but found {update_from_clients_param}.'
      )

    next_fn_result = next_fn.type_signature.result
    if next_fn_result.result.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          'The "result" attribute of the return type of `next_fn` must be '
          f'placed at SERVER, but found {next_fn_result.result}.'
      )
    if not model_weights_param.member.is_assignable_from(
        next_fn_result.result.member
    ):
      raise FinalizerResultTypeError(
          'The second input argument of `next_fn` must match the "result" '
          'attribute of the return type of `next_fn`. Found:\n'
          f'Second input argument: {next_fn_param[1].member}\n'
          f'Result attribute: {next_fn_result.result.member}.'
      )
    if next_fn_result.measurements.placement != placements.SERVER:
      raise errors.TemplatePlacementError(
          'The "measurements" attribute of return type of `next_fn` must be '
          f'placed at SERVER, but found {next_fn_result.measurements}.'
      )

    state_type = initialize_fn.type_signature.result.member
    if get_hparams_fn is not None:
      hparams_base.type_check_get_hparams_fn(get_hparams_fn, state_type)
    else:
      get_hparams_fn = hparams_base.build_basic_hparams_getter(state_type)

    hparams_type = get_hparams_fn.type_signature.result

    if set_hparams_fn is not None:
      hparams_base.type_check_set_hparams_fn(set_hparams_fn, state_type)
    else:
      set_hparams_fn = hparams_base.build_basic_hparams_setter(
          state_type, hparams_type
      )

    self._get_hparams_fn = get_hparams_fn
    self._set_hparams_fn = set_hparams_fn

  @property
  def get_hparams(self) -> computation_base.Computation:
    return self._get_hparams_fn

  @property
  def set_hparams(self) -> computation_base.Computation:
    return self._set_hparams_fn
