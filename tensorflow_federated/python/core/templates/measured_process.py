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

import attr

from tensorflow_federated.python.core.api import computation_types
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

# The type signature of the result of MeasuredProcess must be a named tuple
# with the following names in the same order.
_RESULT_FIELD_NAMES = [f.name for f in attr.fields(MeasuredProcessOutput)]


# TODO(b/150384321): add method for performing the composition; current proposal
# include a stadnalone `measure_process.compose(F, G)`, or implementing
# `G.__call__(F)` to return a new MeasuredProcess.
class MeasuredProcess(iterative_process.IterativeProcess):
  """A `tff.templates.IterativeProcess` with a specific output signature.

  A `tff.templates.MeasuredProcess` is a `tff.templates.IterativeProcess` that
  formalizes the output signature of the `next` property to be a named
  three-tuple `<state,result,measurements>`. This definition enables
  `tff.templates.MeasuredProcess` to be composed following the rules below,
  something that wasn't possible with the more generic, less defined
  `tff.templates.IterativeProcess`.

  *Rules of Composition*
  Given two `MeasuredProcess` _F(x)_ and _G(y)_, a new composition _C_ is
  also a `MeasuredProcess` where:
    - `C.state` is the concatenation `<F.state, G.state>`.
    - `C.result` is the result of _G_ applied to the result of
      _F_: `G(G.state, F(F.state, x).result).result`.
    - `C.measurements` is the concatenation `<F.measurements, G.measurements>`.

  The resulting composition _C_ would have the following type signatures:
    initialize: `( -> <F.initialize, G.initialize>)`
    next: `(<<F.state, G.state>, F.input> -> <state=<F.state, G.State>,
      result=G.result, measurements=<F.measurements, G.measurements>)`
  """

  def __init__(self, initialize_fn, next_fn):
    """Creates a `tff.templates.MeasuredProcess`.

    Args:
      initialize_fn: A no-arg `tff.Computation` that creates the initial state
        of the measured process.
      next_fn: A `tff.Computation` that defines an iterated function. If
        `initialize_fn` returns a type `S`, then `next_fn` must return a
        `MeasuredProcessOutput` where the `state` attribute matches the type
        `S`, and accept either a single argument of type `S` or multiple
        arguments where the first argument must be of type `S`.

    Raises:
      TypeError: `initialize_fn` and `next_fn` are not compatible function
        types, or `next_fn` does not return a `MeasuredProcessOutput`.
    """
    super().__init__(initialize_fn, next_fn)
    next_result_type = next_fn.type_signature.result
    if not (isinstance(next_result_type, computation_types.StructWithPythonType)
            and next_result_type.python_container is MeasuredProcessOutput):
      raise TypeError(
          'MeasuredProcess must return a MeasuredProcessOutput. Received a '
          '({t}): {s}'.format(
              t=type(next_fn.type_signature.result),
              s=next_fn.type_signature.result))

  @property
  def next(self):
    """A `tff.Computation` that runs one iteration of the process.

    Its first argument should always be the current state (originally produced
    by `tff.templates.MeasuredProcess.initialize`), and the return type must be
    a `tff.templates.MeasuredProcessOutput`.

    Returns:
      A `tff.Computation`.
    """
    return self._next_fn
