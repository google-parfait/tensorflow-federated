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
"""Library of compositional helpers for iterative processes."""

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import iterative_process


def compose_dataset_computation(
    dataset_computation: computation_base.Computation,
    process: iterative_process.IterativeProcess,
) -> iterative_process.IterativeProcess:
  """Builds a new iterative process which constructs datasets on clients.

  Given a `tff.Computation` that returns a `tf.data.Dataset`, and a
  `tff.templates.IterativeProcess` where exactly one of the arguments is a
  dataset placed on clients of the same type as returned by the
  `tff.Computation`, this function will construct a new
  `tff.templates.IterativeProcess` whose `next` function accepts a federated set
  of values of the same type as the parameter of the `dataset_computation`, maps
  `dataset_computation` over these values, and proceeds with the body of
  `process.next`.

  For example, if the type signature of `dataset_computation` is:

  ```
  (T -> U*)
  ```
  and the type signature of `process.next` is:

  ```
  (<S, {U*}@CLIENTS> -> <S, V>
  ```

  then the returned `tff.templates.IterativeProcess.next` type signature will
  be:

  ```
  (<S, {T}@CLIENTS> -> <S, V>)
  ```

  This functionality is useful in several settings:

  * We may want to push some dataset preprocessing to happen on the clients, as
    opposed to preprocessing happening on the TFF simultation controller. This
    may be necessary, e.g., in the case that we want to shuffle client
    examples.
  * We may want to *construct* the entire dataset on the clients, given a client
    id. This may be useful in order to speed up distributed simulations, in
    order to remove a linear cost incurred in constructing and serializign the
    datasets on the controller.

  Args:
    dataset_computation: An instance of `tff.Computation` which accepts some
      parameter and returns an element of `tff.SequenceType`.
    process: An instance of `tff.templates.IterativeProcess` whose next function
      accepts exactly one federated dataset, IE, element of type `{B*}@CLIENTS`,
      where `B` is equivalent to the return type of `dataset_computation`.

  Returns:
    A new `tff.templates.IterativeProcess` satisfying the specification above.

  Raises:
    TypeError: If the arguments are of the wrong types, or their TFF type
    signatures are incompatible with the specification of this function.
  """
  py_typecheck.check_type(dataset_computation, computation_base.Computation)
  py_typecheck.check_type(process, iterative_process.IterativeProcess)

  dataset_return_type = dataset_computation.type_signature.result
  federated_param_type = computation_types.FederatedType(
      dataset_computation.type_signature.parameter, placement_literals.CLIENTS)

  if not dataset_return_type.is_sequence():
    raise TypeError(
        'Expected a `tff.SequenceType` to be returned from '
        '`dataset_computation`; found {} instead.'.format(dataset_return_type))

  if dataset_computation.type_signature.parameter is None:
    raise TypeError('Can only construct a new iterative process if '
                    '`dataset_computation` accepts a non-None arg; the '
                    'type {} accepts no argument.'.format(
                        dataset_computation.type_signature))

  init_fn = process.initialize

  if type_analysis.contains(init_fn.type_signature.result,
                            lambda x: x.is_sequence()):
    raise TypeError('Cannot construct a new iterative process if a dataset is '
                    'returned by `initialize`; initialize has result type '
                    '{}.'.format(init_fn.type_signature.result))

  next_fn_param_type = process.next.type_signature.parameter

  # If there is a dataset in the parameter, and `init_fn` does not return a
  # dataset, we know the parameter of next must be a tuple. So we assume that
  # from here on out.
  if not type_analysis.contains(next_fn_param_type, lambda x: x.is_sequence()):
    raise TypeError('IterativeProcess\' next must accept a parameter which '
                    'contains a dataset; the parameter {} contains no '
                    'dataset.'.format(next_fn_param_type))

  dataset_index = None
  new_param_elements = []

  for idx, (elem_name, elem_type) in enumerate(
      structure.iter_elements(next_fn_param_type)):
    if elem_type.is_federated() and elem_type.member.is_equivalent_to(
        dataset_return_type):
      if dataset_index is not None:
        raise TypeError('Cannot accept an iterative process whose next '
                        'function declares more than one sequence parameter; '
                        'received a next function declaring parameter '
                        '{}.'.format(next_fn_param_type))
      dataset_index = idx
      new_param_elements.append((elem_name, federated_param_type))
    else:
      new_param_elements.append((elem_name, elem_type))

  if dataset_index is None:
    raise TypeError('No sequence parameter found in iterative process next '
                    'function. Type signature: {}.'.format(next_fn_param_type))

  new_param_type = computation_types.StructType(new_param_elements)

  @computations.federated_computation(new_param_type)
  def next_fn_taking_client_ids(param):
    datasets_on_clients = intrinsics.federated_map(dataset_computation,
                                                   param[dataset_index])
    original_param = []
    for idx, elem in enumerate(param):
      if idx != dataset_index:
        original_param.append(elem)
      else:
        original_param.append(datasets_on_clients)
    return process.next(original_param)

  return iterative_process.IterativeProcess(
      initialize_fn=init_fn, next_fn=next_fn_taking_client_ids)
