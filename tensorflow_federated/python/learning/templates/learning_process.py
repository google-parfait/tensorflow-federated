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

import typing
from typing import Any, NamedTuple, Optional, Union

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.learning.templates import hparams_base


class Error(Exception):
  """Generic module-level error, allows caller to handle all exceptions raised."""


class LearningProcessPlacementError(Error):
  """Raises when a learning process does not have expected placements."""


class LearningProcessOutputError(Error):
  """Raises when a learning process does not have the expected output type."""


class GetModelWeightsTypeSignatureError(Error):
  """Raises when the type signature of `get_model_weights` is not correct."""


class SetModelWeightsTypeSignatureError(Error):
  """Raises when the type signature of `set_model_weights` is not correct."""


class LearningProcessOutput(NamedTuple):
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
  state: Any
  metrics: Any


class LearningProcess(iterative_process.IterativeProcess):
  """A stateful process for learning tasks that produces metrics.

  This class inherits the constraints documented by
  `tff.templates.IterativeProcess`, including an `initialize` and `next`
  attribute. The `LearningProcess` also contains additional attributes,
  including `get_model_weights` and `get_hparams`. The former can be used to
  get out structures suitable for evaluation purposes, while the latter can
  be used to extract hyperparameters from the process. There are also
  corresponding `set_model_weights` and `set_hparams` attributes that can set
  these structures in a given state.

  For example, given a LearningProcess `process` and client data `data`, we
  could call the following to initialize, optionally load other model weights,
  update the state three times, and extract the model weights of the state:

  >>> state = process.initialize()
  >>> # Optional: state = process.set_model_weights(state, other_weights)
  >>> for _ in range(3):
  >>>  state, metrics = process.next(state, data)
  >>> model_weights = process.get_model_weights(state)
  """

  def __init__(
      self,
      initialize_fn: computation_base.Computation,
      next_fn: computation_base.Computation,
      get_model_weights: computation_base.Computation,
      set_model_weights: computation_base.Computation,
      *,
      get_hparams_fn: Optional[computation_base.Computation] = None,
      set_hparams_fn: Optional[computation_base.Computation] = None,
  ):
    """Creates a `tff.learning.templates.LearningProcess`.

    The `initialize_fn`, `next_fn`, `get_model_weights`, and `set_model_weights`
    must have the following type signatures:

    ```
      - initialize_fn: ( -> S@SERVER)
      - next_fn: (<S@SERVER, {D*}@CLIENTS> -> <state=S@SERVER,
      metrics=M@SERVER>)
      - get_model_weights: (S -> W)
      - set_model_weights: (<S, W> -> S)
    ```
    Here, `S` represents the state of the process, and {D*} represents a
    sequence of data. `M` represents the metrics output by the process. Note
    that while `initialize_fn`, and `next_fn` are federated computations,
    `get_model_weights` and `set_model_weights` are unplaced.

    If provided, the `get_hparams_fn` and `set_hparams_fn` must be non-federated
    computations with the following type signatures:

    ```
      - get_hparams_fn: (S -> H)
      - set_hparams_fn: (<S, H> -> S)
    ```
    Here, `S` must match the state `S` of `initialize_fn` and `next_fn`, and `H`
    represents the hyperparameter type.

    Args:
      initialize_fn: A no-arg `tff.Computation` that creates the initial state
        of the learning process.
      next_fn: A `tff.Computation` that defines an iterated function. Given that
        `initialize_fn` returns a type `S@SERVER`, the `next_fn` must return a
        `LearningProcessOutput` where the `state` attribute is assignable from
        values with type `S@SERVER`, and accepts two arguments with types
        assignable from values with type `S@SERVER` and `{D*}@CLIENTS`.
      get_model_weights: A `tff.Computation` that accepts an input `S` whose
        type is assignable from the result of `init_fn`. This computation is
        used to create a representation of the state that can be used for
        downstream tasks without requiring access to the entire server state.
        For example, `get_model_weights` could be used to extract model weights
        suitable for computing evaluation metrics on held-out data.
      set_model_weights: A `tff.Computation` that accepts two inputs `S` and `M`
        where the type of `S` is assignable from values with the type returned
        by `init_fn` and `M` is a representation of the model weights stored in
        `S`. This updates the model weights representation within the state with
        the incoming value and returns a new value of type `S`.
      get_hparams_fn: An optional `tff.Computation` accepting the state `S` and
        returning the hyperparameters `H`. If not provided, this defaults to a
        computation that returns an empty ordered dictionary, regardless of the
        contents of the state.
      set_hparams_fn: An optional `tff.Computation` accepting the state `S` and
        hyperparameters `H` (matching the output of `get_hparams_fn`) and
        returning an updated state `S`. If not provided, this defaults to a
        pass-through computation that returns the input state regardless of the
        hparams passed in.

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
      GetModelWeightsTypeSignatureError: If the input type of get_model_weights
        does not match the process state type.
      SetModelWeightsTypeSignatureError: If the type of the first input or the
        type of the output of set_model_weights does not match the process state
        type.
    """
    super().__init__(initialize_fn, next_fn)

    init_fn_result = initialize_fn.type_signature.result
    if init_fn_result.placement != placements.SERVER:  # pytype: disable=attribute-error
      raise LearningProcessPlacementError(
          'The result of `initialize_fn` must be placed at `SERVER` but found '
          f'placement {init_fn_result.placement}.'  # pytype: disable=attribute-error
      )

    next_result_type = next_fn.type_signature.result
    # TODO: b/224484886 - Downcasting to all handled types.
    next_result_type = typing.cast(
        Union[computation_types.StructWithPythonType], next_result_type
    )
    if not (
        isinstance(next_result_type, computation_types.StructWithPythonType)
        and next_result_type.python_container is LearningProcessOutput
    ):
      raise LearningProcessOutputError(
          'The `next_fn` of a `LearningProcess` must return a '
          f'`LearningProcessOutput` object, but returns {next_result_type!r}'
      )
    # We perform a more strict type check on the inputs to `next_fn` than in the
    # base class.
    # TODO: b/224484886 - Downcasting to all handled types.
    next_fn_param = typing.cast(
        Union[computation_types.StructType], next_fn.type_signature.parameter
    )
    if (
        not isinstance(next_fn_param, computation_types.StructType)
        or len(next_fn_param) != 2
    ):
      raise errors.TemplateNextFnNumArgsError(
          'The `next_fn` must have two input arguments, but found an input '
          f'of type {next_fn_param}.'
      )
    if next_fn_param[1].placement != placements.CLIENTS:
      raise LearningProcessPlacementError(
          'The second input argument of `next_fn` must be placed at `CLIENTS`,'
          f' but found placement {next_fn_param[1].placement}.'
      )

    next_fn_result = next_fn.type_signature.result
    if next_fn_result.metrics.placement != placements.SERVER:  # pytype: disable=attribute-error
      raise LearningProcessPlacementError(
          'The result of `next_fn` must be placed at `SERVER` but found '
          f'placement {next_fn_result.metrics.placement} for `metrics`.'  # pytype: disable=attribute-error
      )

    py_typecheck.check_type(get_model_weights, computation_base.Computation)
    get_model_weights_type = get_model_weights.type_signature
    get_model_weights_param = get_model_weights_type.parameter
    next_fn_state_param = next_fn.type_signature.parameter[0].member  # pytype: disable=unsupported-operands
    if (
        get_model_weights_param is None
        or not get_model_weights_param.is_equivalent_to(next_fn_state_param)
    ):
      raise GetModelWeightsTypeSignatureError(
          'The input type of `get_model_weights` must be assignable from '
          'the member type of the output of `initialize_fn`, but found input '
          f'type {get_model_weights_param}, which is not equivalent to '
          f'{next_fn_state_param}.'
      )
    self._get_model_weights = get_model_weights

    py_typecheck.check_type(set_model_weights, computation_base.Computation)
    set_model_weights_type = set_model_weights.type_signature
    set_model_weights_state_param = set_model_weights_type.parameter[0]  # pytype: disable=unsupported-operands
    if not set_model_weights_state_param.is_equivalent_to(next_fn_state_param):
      raise SetModelWeightsTypeSignatureError(
          'The input type of `set_model_weights` must be assignable from '
          'the member type of the output of `initialize_fn`, but found input '
          f'type {set_model_weights_state_param}, which is not equivalent to '
          f'{next_fn_state_param}.'
      )
    set_model_weights_result = set_model_weights_type.result
    if not next_fn_state_param.is_assignable_from(set_model_weights_result):
      raise SetModelWeightsTypeSignatureError(
          'The output type of `set_model_weights` must be assignable to '
          'the first parameter of `next_fn`, but found input '
          f'type {set_model_weights_result}, which is not assignable to; '
          f'{next_fn_state_param}.'
      )
    self._set_model_weights = set_model_weights

    state_type = initialize_fn.type_signature.result.member  # pytype: disable=attribute-error
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
  def initialize(self) -> computation_base.Computation:
    """A `tff.Computation` that initializes the process.

    This computation must have no input arguments, and its output must be the
    initial state of the learning process, placed at `SERVER`.

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
  def get_model_weights(self) -> computation_base.Computation:
    """A `tff.Computation` returning the model weights of a server state.

    This computation accepts an unplaced state of the process (originally
    produced by the `initialize` attribute), and returns an unplaced
    representation of the model weights of the state. Note that this
    representation need not take the form of a
    `tff.learning.models.ModelWeights` object, and may depend on the specific
    `LearningProcess` in question.

    Returns:
      A `tff.Computation`.
    """
    return self._get_model_weights

  @property
  def set_model_weights(self) -> computation_base.Computation:
    """A `tff.Computation` that sets the model weights of a server state.

    This computation accepts two arguments: an unplaced state of the process
    (originally produced by the `initialize` attribute) and a new structure of
    tensors representing the model weights, and returns new unplaced state with
    the updated model weights. Note that the model weights representation need
    not take the form of a `tff.learning.models.ModelWeights` object, and may
    depend on
    the specific `LearningProcess` in question.

    Returns:
      A `tff.Computation`.
    """
    return self._set_model_weights

  @property
  def get_hparams(self) -> computation_base.Computation:
    """A `tff.Computation` returning the hyperparameters of a server state.

    This computation accepts an unplaced state of the process (originally
    produced by the `initialize` attribute), and returns an unplaced ordered
    dictionary representing the hyperparameters of the state.

    Returns:
      A `tff.Computation`.
    """
    return self._get_hparams_fn

  @property
  def set_hparams(self) -> computation_base.Computation:
    """A `tff.Computation` that sets the hyperparamters of a server state.

    This computation accepts two arguments: an unplaced state of the process
    (originally produced by the `initialize` attribute) and an ordered
    dictionary representing the hyperparameters (matching the output of
    `get_hparams`), and returns a new unplaced state with updated
    hyperparameters.

    Returns:
      A `tff.Computation`.
    """
    return self._set_hparams_fn
