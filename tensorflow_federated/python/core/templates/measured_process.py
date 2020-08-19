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

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.templates import iterative_process

# The type signature of the result of MeasuredProcess must be a named tuple
# with the following names in the same order.
_RESULT_FIELD_NAMES = ['state', 'result', 'measurements']


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
        `initialize_fn` returns a type `S`, then `next_fn` must return a TFF
        type `<state=S,result=O,measurements=M>`, and accept either a single
        argument of type `S` or multiple arguments where the first argument must
        be of type `S`.

    Raises:
      TypeError: `initialize_fn` and `next_fn` are not compatible function
        types, or `next_fn` does not have a valid output signature.
    """
    super().__init__(initialize_fn, next_fn)
    # Additional type checks for the specialized MeasuredProcess.
    if isinstance(next_fn.type_signature.result,
                  computation_types.FederatedType):
      next_result_type = next_fn.type_signature.result.member
    elif isinstance(next_fn.type_signature.result,
                    computation_types.StructType):
      next_result_type = next_fn.type_signature.result
    else:
      raise TypeError(
          'MeasuredProcess must return a StructType (or '
          'FederatedType containing a StructType) with the signature '
          '<state=A,result=B,measurements=C>. Received a ({t}): {s}'.format(
              t=type(next_fn.type_signature.result),
              s=next_fn.type_signature.result))
    result_field_names = [
        name for (name, _) in structure.iter_elements(next_result_type)
    ]
    if result_field_names != _RESULT_FIELD_NAMES:
      raise TypeError('The return type of next_fn must match type signature '
                      '<state=A,result=B,measurements=C>. Got: {!s}'.format(
                          next_result_type))

  @property
  def next(self):
    """A `tff.Computation` that runs one iteration of the process.

    Its first argument should always be the current state (originally produced
    by `tff.templates.MeasuredProcess.initialize`), and the return type must be
    a named tuple matching the signature `<state=A,result=B,measurements=C>`.

    Returns:
      A `tff.Computation`.
    """
    return self._next_fn
