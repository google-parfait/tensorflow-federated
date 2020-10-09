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
"""Defines a template for a process that maintains an estimate."""

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process


class EstimationProcess(iterative_process.IterativeProcess):
  """A `tff.templates.IterativeProcess that maintains an estimate.

  In addition to the `initialize` and `next` functions provided by an
  `IterativeProcess`, an `EstimationProcess` has a `get_estimate` function that
  returns the result of some computation on the process state. The argument
  of `get_estimate` must be of the same type as the state, that is, the type
  of object returned by `initialize`.
  """

  def __init__(self, initialize_fn: computation_base.Computation,
               next_fn: computation_base.Computation,
               get_estimate_fn: computation_base.Computation):
    super().__init__(initialize_fn, next_fn)

    py_typecheck.check_type(get_estimate_fn, computation_base.Computation)
    estimate_fn_arg_type = get_estimate_fn.type_signature.parameter
    if not estimate_fn_arg_type.is_assignable_from(self.state_type):
      raise errors.TemplateStateNotAssignableError(
          f'The state type of the process must be assignable to the '
          f'input argument of `get_estimate_fn`, but the state type is: '
          f'{self.state_type}\n'
          f'and the argument of `get_estimate_fn` is:\n'
          f'{estimate_fn_arg_type}')

    self._get_estimate_fn = get_estimate_fn

  @property
  def get_estimate(self) -> computation_base.Computation:
    """A `tff.Computation` that computes the current estimate from `state`.

    Given a `state` controlled by this process, computes and returns the most
    recent estimate of the estimated quantity.

    Note that this computation operates on types without placements, and thus
    can be used with `state` residing either on `SERVER` or `CLIENTS`.

    Returns:
      A `tff.Computation`.
    """
    return self._get_estimate_fn


def apply(transform_fn: computation_base.Computation,
          arg_process: EstimationProcess):
  """Builds an `EstimationProcess` by applying `transform_fn` to `arg_process`.

  Args:
    transform_fn: A `computation_base.Computation` to apply to the estimate of
      the arg_process.
    arg_process: An `EstimationProcess` to which the transformation will be
      applied.

  Returns:
    An estimation process that applies `transform_fn` to the result of calling
      `arg_process.get_estimate`.
  """
  py_typecheck.check_type(transform_fn, computation_base.Computation)
  py_typecheck.check_type(arg_process, EstimationProcess)

  arg_process_estimate_type = arg_process.get_estimate.type_signature.result
  transform_fn_arg_type = transform_fn.type_signature.parameter

  if not transform_fn_arg_type.is_assignable_from(arg_process_estimate_type):
    raise errors.TemplateStateNotAssignableError(
        f'The return type of `get_estimate` of `arg_process` must be '
        f'assignable to the input argument of `transform_fn`, but '
        f'`get_estimate` returns type:\n{arg_process_estimate_type}\n'
        f'and the argument of `transform_fn` is:\n'
        f'{transform_fn_arg_type}')

  transformed_estimate_fn = computations.tf_computation(
      lambda state: transform_fn(arg_process.get_estimate(state)),
      arg_process.state_type)

  return EstimationProcess(
      initialize_fn=arg_process.initialize,
      next_fn=arg_process.next,
      get_estimate_fn=transformed_estimate_fn)
