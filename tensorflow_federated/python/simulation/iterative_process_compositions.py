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

import federated_language
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.learning.templates import learning_process


class SequenceTypeNotAssignableError(TypeError):
  pass


class SequenceTypeNotFoundError(TypeError):
  pass


class MultipleMatchingSequenceTypesError(TypeError):
  pass


def compose_dataset_computation_with_computation(
    dataset_computation: federated_language.framework.Computation,
    computation_body: federated_language.framework.Computation,
) -> federated_language.framework.Computation:
  """Builds a new `federated_language.Computation` which constructs datasets on clients.

  Given a `federated_language.Computation` that returns a `tf.data.Dataset`, and
  a
  `federated_language.Computation` where exactly one of the arguments is a
  dataset placed on
  clients of the same type as returned by the `federated_language.Computation`,
  this function
  will construct a new `federated_language.Computation` that accepts a federated
  set of values
  of the same type as the parameter of the `dataset_computation`, maps
  `dataset_computation` over these values, and proceeds with the body of
  `computation_body`.

  For example, if the type signature of `dataset_computation` is:

  ```
  (T -> U*)
  ```

  and the type signature of `computation_body` is:

  ```
  ({U*}@CLIENTS -> V)
  ```

  then the returned `computation_body` type signature will be:

  ```
  ({T}@CLIENTS -> V)
  ```

  This functionality is useful in several settings:

  * We may want to push some dataset preprocessing to happen on the clients, as
    opposed to preprocessing happening on the TFF simultation controller. This
    may be necessary, e.g., in the case that we want to shuffle client
    examples.
  * We may want to *construct* the entire dataset on the clients, given a client
    id. This may be useful in order to speed up distributed simulations, in
    order to remove a linear cost incurred in constructing and serializing the
    datasets on the controller.

  Args:
    dataset_computation: An instance of `federated_language.Computation` which
      accepts some parameter and returns an element of
      `federated_language.SequenceType`.
    computation_body: An instance of `federated_language.Computation` that
      accepts exactly one federated dataset, IE, element of type `{B*}@CLIENTS`,
      where `B` is equivalent to the return type of `dataset_computation`.

  Returns:
    A new `federated_language.Computation` satisfying the specification above.

  Raises:
    TypeError: If the arguments are of the wrong types, their TFF type
    signatures are incompatible with the specification of this function, or if
    `computation_body` declares more than one sequence parameter matching the
    expected dataset type.
  """
  py_typecheck.check_type(
      dataset_computation, federated_language.framework.Computation
  )
  py_typecheck.check_type(
      computation_body, federated_language.framework.Computation
  )

  dataset_return_type = dataset_computation.type_signature.result
  if not isinstance(dataset_return_type, federated_language.SequenceType):
    raise TypeError(
        'Expected a `federated_language.SequenceType` to be returned from '
        '`dataset_computation`; found {} instead.'.format(dataset_return_type)
    )
  # TODO: b/226637447 - This restriction seems unnecessary, and can be removed.
  if dataset_computation.type_signature.parameter is None:
    raise TypeError(
        'Can only construct a new iterative process if '
        '`dataset_computation` accepts a non-None arg; the '
        'type {} accepts no argument.'.format(
            dataset_computation.type_signature
        )
    )

  comp_body_param_type = computation_body.type_signature.parameter

  def is_desired_federated_sequence(t):
    if not isinstance(t, federated_language.FederatedType):
      return False
    return t.member.is_assignable_from(dataset_return_type)

  if is_desired_federated_sequence(comp_body_param_type):
    # Single argument that matches, we compose in a straightforward manner.
    new_param_type = federated_language.FederatedType(
        dataset_computation.type_signature.parameter, federated_language.CLIENTS
    )

    @federated_language.federated_computation(new_param_type)
    def new_computation(param):
      datasets_on_clients = federated_language.federated_map(
          dataset_computation, param
      )
      return computation_body(datasets_on_clients)

    return new_computation
  elif isinstance(comp_body_param_type, federated_language.StructType):
    # If the computation has multiple arguments we need to search over them
    # recursively to find the one that matches the type signature of
    # dataset_computation's result.

    # Tracks the path of the matching type in the computation arguments as a
    # list of indices.
    dataset_index_path = None
    # Federated version of the dataset_computation's argument type signature to
    # use in the final computation type.
    federated_param_type = federated_language.FederatedType(
        dataset_computation.type_signature.parameter, federated_language.CLIENTS
    )
    # Tracks all sequence types encountered in the recursive search for the
    # error message in case the desired argument is not found.
    sequence_types = []

    def build_new_param_type(
        struct_param_type: federated_language.StructType, index_path
    ):
      """Builds a new struct parameter type.

      By recursively finding the field that matches the type signature of
      dataset_computation's result, and replacing it with the federated version.

      Args:
        struct_param_type: An instance of `federated_language.StructType` with a
          field that matches the type signature of dataset_computation's result.
        index_path: An accumulator of indices through nested
          `federated_language.StructType`s for the location of the matching type
          signature.

      Returns:
        A new `federated_language.StructType` satisfying the specification
        above.

      Raises:
        MultipleMatchingSequenceTypesError: If more than one matching type
          signature is found.
      """
      nonlocal dataset_index_path
      new_param_elements = []
      for idx, (elem_name, elem_type) in enumerate(struct_param_type.items()):
        if isinstance(
            elem_type, federated_language.FederatedType
        ) and isinstance(elem_type.member, federated_language.SequenceType):
          sequence_types.append(elem_type.member)

        if is_desired_federated_sequence(elem_type):
          if dataset_index_path is not None:
            raise MultipleMatchingSequenceTypesError(
                'Cannot accept a `computation_body` computation '
                'that declares more than one sequence parameter '
                f'matching the expected dataset type {elem_type}; '
                'received a computation declaring parameter '
                f'{comp_body_param_type}.'
            )
          dataset_index_path = index_path + [idx]
          new_param_elements.append((elem_name, federated_param_type))
        elif isinstance(elem_type, federated_language.StructType):
          new_param_elements.append(
              (elem_name, build_new_param_type(elem_type, index_path + [idx]))
          )
        else:
          new_param_elements.append((elem_name, elem_type))
      return federated_language.StructType(new_param_elements)

    new_param_type = build_new_param_type(comp_body_param_type, [])

    if dataset_index_path is None:
      # Raise a more informative error message in the case that computation_body
      # accepts sequences whose types are not compatible with `elem_type`.
      if sequence_types:
        raise SequenceTypeNotAssignableError(
            'No sequence parameter assignable from expected dataset computation'
            ' result type found in `computation_body`. \nList of sequences in'
            ' argument signature: {}\nExpected sequence type: {}'.format(
                sequence_types, repr(dataset_return_type)
            )
        )
      else:
        raise SequenceTypeNotFoundError(
            'No sequence parameter found in `computation_body`, but '
            'composition with a computation yielding sequences requested.'
            '\nArgument signature: {}\nExpected sequence type: {}'.format(
                repr(comp_body_param_type), repr(dataset_return_type)
            )
        )

    def map_at_path(param, index_path, depth, computation):
      """Builds a new parameter by inserting a `federated_map` computation.

      Args:
        param: An instance of `federated_language.StructType`.
        index_path: A list of indices through nested
          `federated_language.StructType`s specifying the location for the
          insert.
        depth: Tracks index of `index_path` while recursively traversing the
          nested structure of `param`.
        computation: Computation to insert.

      Returns:
        A Python `list` of elements identical to the input `param` with the
        exception that at the possibly-nested part of `param` as identified by
        the indices in `index_path`, `computation` has been applied to the
        original part of `param` via federated_language.federated_map.
      """
      ret_param = []
      for idx, elem in enumerate(param):
        if idx != index_path[depth]:
          ret_param.append(elem)
        elif depth == len(index_path) - 1:
          ret_param.append(federated_language.federated_map(computation, elem))
        else:
          ret_param.append(
              map_at_path(elem, index_path, depth + 1, computation)
          )
      return ret_param

    @federated_language.federated_computation(new_param_type)
    def new_computation(param):
      return computation_body(
          map_at_path(param, dataset_index_path, 0, dataset_computation)
      )

    return new_computation
  else:
    raise TypeError(
        '`computation_body` is not a single argument matching the '
        'signature of `dataset_computation` result signature, nor a struct '
        'of arguments.\n'
        'Argument signature: {}\n'
        'Result signature: {}'.format(
            repr(comp_body_param_type), repr(dataset_return_type)
        )
    )


def compose_dataset_computation_with_iterative_process(
    dataset_computation: federated_language.framework.Computation,
    process: iterative_process.IterativeProcess,
) -> iterative_process.IterativeProcess:
  """Builds a new iterative process which constructs datasets on clients.

  Given a `federated_language.Computation` that returns a `tf.data.Dataset`, and
  a
  `tff.templates.IterativeProcess` where exactly one of the arguments is a
  dataset placed on clients of the same type as returned by the
  `federated_language.Computation`, this function will construct a new
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
    dataset_computation: An instance of `federated_language.Computation` which
      accepts some parameter and returns an element of
      `federated_language.SequenceType`.
    process: An instance of `tff.templates.IterativeProcess` whose next function
      accepts exactly one federated dataset, IE, element of type `{B*}@CLIENTS`,
      where `B` is equivalent to the return type of `dataset_computation`.

  Returns:
    A new `tff.templates.IterativeProcess` satisfying the specification above.

  Raises:
    TypeError: If the arguments are of the wrong types, or their TFF type
    signatures are incompatible with the specification of this function.
  """
  py_typecheck.check_type(
      dataset_computation, federated_language.framework.Computation
  )
  py_typecheck.check_type(process, iterative_process.IterativeProcess)

  dataset_return_type = dataset_computation.type_signature.result
  if not isinstance(dataset_return_type, federated_language.SequenceType):
    raise TypeError(
        'Expected a `federated_language.SequenceType` to be returned from '
        '`dataset_computation`; found {} instead.'.format(dataset_return_type)
    )
  # TODO: b/226637447 - This restriction seems unnecessary, and can be removed.
  if dataset_computation.type_signature.parameter is None:
    raise TypeError(
        'Can only construct a new iterative process if '
        '`dataset_computation` accepts a non-None arg; the '
        'type {} accepts no argument.'.format(
            repr(dataset_computation.type_signature)
        )
    )

  init_fn = process.initialize
  if federated_language.framework.type_contains(
      init_fn.type_signature.result,
      lambda x: isinstance(x, federated_language.SequenceType),
  ):
    raise TypeError(
        'Cannot construct a new iterative process if a dataset is '
        'returned by `initialize`; initialize has result type '
        '{}.'.format(repr(init_fn.type_signature.result))
    )

  new_next_comp = compose_dataset_computation_with_computation(
      dataset_computation, process.next
  )
  return iterative_process.IterativeProcess(
      initialize_fn=init_fn, next_fn=new_next_comp
  )


def compose_dataset_computation_with_learning_process(
    dataset_computation: federated_language.framework.Computation,
    process: learning_process.LearningProcess,
) -> learning_process.LearningProcess:
  """Builds a new learning process which constructs datasets on clients.

  This functionality is identical to
  `tff.simulation.compose_dataset_computation_with_iterative_process`, except
  that all public attributes of the process (except for `initialize` and `next`)
  are also preserved (eg. `LearningProcess.get_model_weights`).

  Args:
    dataset_computation: An instance of `federated_language.Computation` which
      accepts some parameter and returns an element of
      `federated_language.SequenceType`.
    process: An instance of `tff.learning.templates.LearningProcess` whose
      `next` function accepts exactly one federated dataset (ie. something of
      type `{B*}@CLIENTS`, where `B` is equivalent to the return type of
      `dataset_computation`).

  Returns:
    A `tff.learning.templates.LearningProcess`.
  """
  new_iterative_process = compose_dataset_computation_with_iterative_process(
      dataset_computation, process
  )
  new_learning_process = learning_process.LearningProcess(
      initialize_fn=new_iterative_process.initialize,
      next_fn=new_iterative_process.next,
      get_model_weights=process.get_model_weights,
      set_model_weights=process.set_model_weights,
      get_hparams_fn=process.get_hparams,
      set_hparams_fn=process.set_hparams,
  )
  return new_learning_process
