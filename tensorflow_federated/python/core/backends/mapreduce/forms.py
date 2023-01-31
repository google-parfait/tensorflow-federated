# Copyright 2019, The TensorFlow Federated Authors.
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
"""Standardized representation of logic deployable to MapReduce-like systems."""

from collections.abc import Callable
from typing import Optional

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import typed_object


def _check_tensorflow_computation(label, comp):
  py_typecheck.check_type(comp, computation_impl.ConcreteComputation, label)
  comp_proto = computation_impl.ConcreteComputation.get_proto(comp)
  which_comp = comp_proto.WhichOneof('computation')
  if which_comp != 'tensorflow':
    raise TypeError(
        'Expected all computations supplied as arguments to '
        'be plain TensorFlow, found {}.'.format(which_comp)
    )


def _check_lambda_computation(label, comp):
  py_typecheck.check_type(comp, computation_impl.ConcreteComputation, label)
  comp_proto = computation_impl.ConcreteComputation.get_proto(comp)
  which_comp = comp_proto.WhichOneof('computation')
  if which_comp != 'lambda':
    raise TypeError(
        'Expected all computations supplied as arguments to '
        'be Lambda computations, found {}.'.format(which_comp)
    )
  tree_analysis.check_contains_no_unbound_references(comp.to_building_block())
  tree_analysis.check_has_unique_names(comp.to_building_block())


def _check_flattened_intrinsic_args_are_selections(
    value: building_blocks.ComputationBuildingBlock,
    expected_reference_name: str,
):
  """Checks that the flattened args of an intrinsic call are all Selections."""
  inner_values = []
  if value.is_struct():
    inner_values = structure.flatten(value)
  else:
    inner_values = [value]

  for inner_value in inner_values:
    if not inner_value.is_selection():
      raise TypeError(
          'Expected that all arguments to an intrinsic call are selections or '
          '(potentially nested) structs of selections, but found {}'.format(
              inner_value.type_signature
          )
      )
    if (
        inner_value.source.is_reference()
        and inner_value.source.name != expected_reference_name
    ):
      raise TypeError(
          'Expected that all arguments to an intrinsic call are ultimately '
          'selections of the top-level lambda parameter {} but found selection '
          'source {}'.format(expected_reference_name, inner_value.source.name)
      )


def _is_assignable_from_or_both_none(first, second):
  if first is None:
    return second is None
  return first.is_assignable_from(second)


def _is_tuple(type_signature: computation_types.Type, length: int) -> bool:
  return type_signature.is_struct() and len(type_signature) == length


def _check_accepts_tuple(
    label: str, comp: computation_base.Computation, length: int
):
  param_type = comp.type_signature.parameter
  if not _is_tuple(param_type, length):
    raise TypeError(
        f'The `{label}` computation accepts a parameter of type\n{param_type}\n'
        f'that is not a tuple of length {length}.'
    )


def _check_returns_tuple(
    label: str, comp: computation_base.Computation, length: int
):
  result_type = comp.type_signature.result
  if not _is_tuple(result_type, length):
    raise TypeError(
        f'The `{label}` computation returns a result of type\n{result_type}\n'
        f'that is not a tuple of length {length}.'
    )


class BroadcastForm:
  """Standardized representation of server-to-client logic.

  This class is designed to represent computations of the form:

  ```
  server_data_type = self.compute_server_context.type_signature.parameter
  client_data_type = self.client_processing.type_signature.parameter[1]
  @tff.federated_computation(server_data_type, client_data_type)
  def _(server_data, client_data):
    # Select out the bit of server context to send to the clients.
    context_at_server = tff.federated_map(
      self.compute_server_context, server_data)

    # Broadcast the context to the clients.
    context_at_clients = tff.federated_broadcast(context_at_server)

    # Compute some value on the clients based on the server context and
    # the client data.
    return tff.federated_map(
      self.client_processing, (context_at_clients, client_data))
  ```
  """

  def __init__(
      self,
      compute_server_context,
      client_processing,
      server_data_label=None,
      client_data_label=None,
  ):
    for label, comp in (
        ('compute_server_context', compute_server_context),
        ('client_processing', client_processing),
    ):
      _check_tensorflow_computation(label, comp)
    _check_accepts_tuple('client_processing', client_processing, 2)
    client_first_arg_type = client_processing.type_signature.parameter[0]
    server_context_type = compute_server_context.type_signature.result
    if not _is_assignable_from_or_both_none(
        client_first_arg_type, server_context_type
    ):
      raise TypeError(
          'The `client_processing` computation expects an argument tuple with '
          f'type\n{client_first_arg_type}\nas the first element (the context '
          'type from the server), which does not match the result type\n'
          f'{server_context_type}\n of `compute_server_context`.'
      )
    self._compute_server_context = compute_server_context
    self._client_processing = client_processing
    if server_data_label is not None:
      py_typecheck.check_type(server_data_label, str)
    self._server_data_label = server_data_label
    if client_data_label is not None:
      py_typecheck.check_type(server_data_label, str)
    self._client_data_label = client_data_label

  @property
  def compute_server_context(self):
    return self._compute_server_context

  @property
  def client_processing(self):
    return self._client_processing

  @property
  def server_data_label(self):
    return self._server_data_label

  @property
  def client_data_label(self):
    return self._client_data_label

  def summary(self, print_fn=print):
    """Prints a string summary of the `BroadcastForm`.

    Args:
      print_fn: Print function to use. It will be called on each line of the
        summary in order to capture the string summary.
    """
    for label, comp in (
        ('compute_server_context', self.compute_server_context),
        ('client_processing', self.client_processing),
    ):
      # Add sufficient padding to align first column;
      # len('compute_server_context') == 22
      print_fn(
          '{:<22}: {}'.format(
              label, comp.type_signature.compact_representation()
          )
      )


# `work` has a separate output for each aggregation path.
WORK_UPDATE_INDEX = 0
WORK_SECAGG_BITWIDTH_INDEX = 1
WORK_SECAGG_MAX_INPUT_INDEX = 2
WORK_SECAGG_MODULUS_INDEX = 3
WORK_RESULT_LEN = 4


class MapReduceForm(typed_object.TypedObject):
  """Standardized representation of logic deployable to MapReduce-like systems.

  This class docstring describes the purpose of `MapReduceForm` as a data
  structure; for a discussion of the conceptual content of an instance `mrf` of
  `MapReduceForm`, including how precisely it maps to a single federated round,
  see the [package-level docstring](
  https://www.tensorflow.org/federated/api_docs/python/tff/backends/mapreduce).

  This standardized representation can be used to describe a range of
  computations representable as a single round of MapReduce-like processing, and
  deployable to MapReduce-like systems that are only capable of executing plain
  TensorFlow code.

  Processes that do not originate at the server can be described by
  `MapReduceForm`, as well as degenerate cases like computations which use
  exclusively one of the two possible aggregation paths.

  Instances of this class can be generated by TFF's transformation pipeline and
  consumed by a variety of backends that have the ability to orchestrate their
  execution in a MapReduce-like fashion. The latter can include systems that run
  static data pipelines such Apache Beam or Hadoop, but also platforms like that
  which has been described in the following paper:

  "Towards Federated Learning at Scale: System Design"
  https://arxiv.org/pdf/1902.01046.pdf

  It should be noted that not every computation that proceeds in synchronous
  rounds is representable as an instance of this class. In particular, this
  representation is not suitable for computations that involve multiple phases
  of processing, and does not generalize to arbitrary static data pipelines.
  Generalized representations that can take advantage of the full expressiveness
  of Apache Beam-like systems may emerge at a later time, and will be supported
  by a separate set of tools, with a more expressive canonical representation.

  The requirement that the variable constituents of the template be in the form
  of pure TensorFlow code (not arbitrary TFF constructs) reflects the intent
  for instances of this class to be easily converted into a representation that
  can be compiled into a system that does *not* have the ability to interpret
  the full TFF language (as defined in `computation.proto`), but that does have
  the ability to run TensorFlow. Client-side logic in such systems could be
  deployed in a number of ways, e.g., as shards in a MapReduce job, to mobile or
  embedded devices, etc.

  The individual TensorFlow computations that constitute a computation in this
  form are supplied as constructor arguments. Generally, this class will note be
  instantiated by a programmer directly but targeted by a sequence of
  transformations that take a `tff.Computation` and produce the appropriate
  pieces of logic.
  """

  def __init__(
      self,
      type_signature: computation_types.FunctionType,
      prepare: computation_impl.ConcreteComputation,
      work: computation_impl.ConcreteComputation,
      zero: computation_impl.ConcreteComputation,
      accumulate: computation_impl.ConcreteComputation,
      merge: computation_impl.ConcreteComputation,
      report: computation_impl.ConcreteComputation,
      secure_sum_bitwidth: computation_impl.ConcreteComputation,
      secure_sum_max_input: computation_impl.ConcreteComputation,
      secure_modular_sum_modulus: computation_impl.ConcreteComputation,
      update: computation_impl.ConcreteComputation,
  ):
    """Constructs a representation of a MapReduce-like computation.

    Note: All the computations supplied here as arguments must be TensorFlow
    computations, i.e., instances of `tff.Computation` constructed by the
    `tff.tf_computation` decorator/wrapper.

    Args:
      type_signature: The type signature of the corresponding `tff.Computation`
        that is equivalent to the pieces of logic encoded in this data
        structure.
      prepare: The computation that prepares the input for the clients.
      work: The client-side work computation.
      zero: The computation that produces the initial state for accumulators.
      accumulate: The computation that adds a client update to an accumulator.
      merge: The computation to use for merging pairs of accumulators.
      report: The computation that produces the final server-side aggregate for
        the top level accumulator (the global update).
      secure_sum_bitwidth: The computation that produces the bitwidth for
        bitwidth-based secure sums.
      secure_sum_max_input: The computation that produces the maximum input for
        `max_input`-based secure sums.
      secure_modular_sum_modulus: The computation that produces the modulus for
        secure modular sums.
      update: The computation that takes the global update and the server state
        and produces the new server state, as well as server-side output.

    Raises:
      TypeError: If the Python or TFF types of the arguments are invalid or not
        compatible with each other.
      AssertionError: If the manner in which the given TensorFlow computations
        are represented by TFF does not match what this code is expecting (this
        is an internal error that requires code update).
    """
    for label, comp in (
        ('prepare', prepare),
        ('work', work),
        ('zero', zero),
        ('accumulate', accumulate),
        ('merge', merge),
        ('report', report),
        ('secure_sum_bitwidth', secure_sum_bitwidth),
        ('secure_sum_max_input', secure_sum_max_input),
        ('secure_modular_sum_modulus', secure_modular_sum_modulus),
        ('update', update),
    ):
      _check_tensorflow_computation(label, comp)

    prepare_arg_type = prepare.type_signature.parameter

    _check_accepts_tuple('work', work, 2)
    work_2nd_arg_type = work.type_signature.parameter[1]
    prepare_result_type = prepare.type_signature.result
    if not _is_assignable_from_or_both_none(
        work_2nd_arg_type, prepare_result_type
    ):
      raise TypeError(
          'The `work` computation expects an argument tuple with type {} as '
          'the second element (the initial client state from the server), '
          'which does not match the result type {} of `prepare`.'.format(
              work_2nd_arg_type, prepare_result_type
          )
      )

    _check_returns_tuple('work', work, WORK_RESULT_LEN)

    py_typecheck.check_len(accumulate.type_signature.parameter, 2)
    accumulate.type_signature.parameter[0].check_assignable_from(
        zero.type_signature.result
    )
    accumulate_2nd_arg_type = accumulate.type_signature.parameter[1]
    work_client_update_type = work.type_signature.result[WORK_UPDATE_INDEX]
    if not _is_assignable_from_or_both_none(
        accumulate_2nd_arg_type, work_client_update_type
    ):
      raise TypeError(
          'The `accumulate` computation expects a second argument of type {}, '
          'which does not match the expected {} as implied by the type '
          'signature of `work`.'.format(
              accumulate_2nd_arg_type, work_client_update_type
          )
      )
    accumulate.type_signature.parameter[0].check_assignable_from(
        accumulate.type_signature.result
    )

    py_typecheck.check_len(merge.type_signature.parameter, 2)
    merge.type_signature.parameter[0].check_assignable_from(
        accumulate.type_signature.result
    )
    merge.type_signature.parameter[1].check_assignable_from(
        accumulate.type_signature.result
    )
    merge.type_signature.parameter[0].check_assignable_from(
        merge.type_signature.result
    )

    report.type_signature.parameter.check_assignable_from(
        merge.type_signature.result
    )

    expected_update_parameter_type = computation_types.to_type([
        type_signature.parameter[0].member,
        [
            report.type_signature.result,
            # Update takes in the post-summation values of secure aggregation.
            work.type_signature.result[WORK_SECAGG_BITWIDTH_INDEX],
            work.type_signature.result[WORK_SECAGG_MAX_INPUT_INDEX],
            work.type_signature.result[WORK_SECAGG_MODULUS_INDEX],
        ],
    ])
    # The first part of the parameter should align with any initial state that
    # the Computation that the MapReduceForm is based upon should take in as
    # input. Verifying it aligns with a tff.Computation that produces an initial
    # state should be verified outside of the constructor of the MapReduceForm.
    if not _is_assignable_from_or_both_none(
        computation_types.to_type(update.type_signature.parameter),
        expected_update_parameter_type,
    ):
      raise TypeError(
          'The `update` computation expects arguments of type {}, '
          'which does not match the expected {} as implied by the type '
          'signatures of `report` and `work`.'.format(
              computation_types.to_type(update.type_signature.parameter[1:]),
              expected_update_parameter_type,
          )
      )

    _check_returns_tuple('update', update, 2)

    updated_state_type = update.type_signature.result[0]
    if not prepare_arg_type.is_assignable_from(updated_state_type):
      raise TypeError(
          'The `update` computation returns a result tuple whose first element '
          '(the updated state type of the server) is type:\n'
          f'{updated_state_type}\n'
          'which is not assignable to the state parameter type of `prepare`:\n'
          f'{prepare_arg_type}'
      )

    self._type_signature = type_signature
    self._prepare = prepare
    self._work = work
    self._zero = zero
    self._accumulate = accumulate
    self._merge = merge
    self._report = report
    self._secure_sum_bitwidth = secure_sum_bitwidth
    self._secure_sum_max_input = secure_sum_max_input
    self._secure_modular_sum_modulus = secure_modular_sum_modulus
    self._update = update

    parameter_names = structure.name_list_with_nones(type_signature.parameter)
    self._server_state_label, self._client_data_label = parameter_names

  @property
  def type_signature(self) -> computation_types.FunctionType:
    """Returns the TFF type of the equivalent `tff.Computation`."""
    return self._type_signature

  @property
  def prepare(self) -> computation_impl.ConcreteComputation:
    return self._prepare

  @property
  def work(self) -> computation_impl.ConcreteComputation:
    return self._work

  @property
  def zero(self) -> computation_impl.ConcreteComputation:
    return self._zero

  @property
  def accumulate(self) -> computation_impl.ConcreteComputation:
    return self._accumulate

  @property
  def merge(self) -> computation_impl.ConcreteComputation:
    return self._merge

  @property
  def report(self) -> computation_impl.ConcreteComputation:
    return self._report

  @property
  def secure_sum_bitwidth(self) -> computation_impl.ConcreteComputation:
    return self._secure_sum_bitwidth

  @property
  def secure_sum_max_input(self) -> computation_impl.ConcreteComputation:
    return self._secure_sum_max_input

  @property
  def secure_modular_sum_modulus(self) -> computation_impl.ConcreteComputation:
    return self._secure_modular_sum_modulus

  @property
  def update(self) -> computation_impl.ConcreteComputation:
    return self._update

  @property
  def server_state_label(self) -> Optional[str]:
    return self._server_state_label

  @property
  def client_data_label(self) -> Optional[str]:
    return self._client_data_label

  @property
  def securely_aggregates_tensors(self) -> bool:
    """Whether the `MapReduceForm` uses secure aggregation."""
    # Tensors aggregated over `federated_secure_...` are the last three outputs
    # of `work`.
    _, secagg_bitwidth_type, secagg_max_input_type, secagg_modulus_type = (
        self.work.type_signature.result
    )
    for secagg_type in [
        secagg_bitwidth_type,
        secagg_max_input_type,
        secagg_modulus_type,
    ]:
      if type_analysis.contains_tensor_types(secagg_type):
        return True
    return False

  def summary(self, print_fn: Callable[..., None] = print) -> None:
    """Prints a string summary of the `MapReduceForm`.

    Args:
      print_fn: Print function to use. It will be called on each line of the
        summary in order to capture the string summary.
    """
    for label, comp in (
        ('prepare', self.prepare),
        ('work', self.work),
        ('zero', self.zero),
        ('accumulate', self.accumulate),
        ('merge', self.merge),
        ('report', self.report),
        ('secure_sum_bitwidth', self.secure_sum_bitwidth),
        ('secure_sum_max_input', self.secure_sum_max_input),
        ('secure_modular_sum_modulus', self.secure_modular_sum_modulus),
        ('update', self.update),
    ):
      # Add sufficient padding to align first column;
      # len('secure_modular_sum_modulus') == 26
      print_fn(
          '{:<26}: {}'.format(
              label, comp.type_signature.compact_representation()
          )
      )


class DistributeAggregateForm(typed_object.TypedObject):
  """Standard representation of logic deployable to a federated learning system.

  This class docstring describes the purpose of `DistributeAggregateForm` as a
  data structure. For a discussion of how an instance of
  `DistributeAggregateForm` maps to a single federated round, see the [package-
  level docstring](
  https://www.tensorflow.org/federated/api_docs/python/tff/backends/mapreduce).

  This standardized representation can be used to describe a range of
  computations that constitute one round of processing in a federated learning
  system such as the one described in the following paper:

  "Towards Federated Learning at Scale: System Design"
  https://arxiv.org/pdf/1902.01046.pdf

  It should be noted that not every computation that proceeds in synchronous
  rounds is representable as an instance of this class. In particular, this
  representation is not suitable for computations that involve multiple back-
  and-forths between the server and clients.

  Each of the variable constituents of the form are TFF Lambda Computations
  (as defined in `computation.proto`). Systems that cannot run TFF directly can
  convert these TFF Lambda Computations into TensorFlow code using TFF helper
  functions. Generally this class will not be instantiated by a programmer
  directly but instead targeted by a sequence of transformations that take a
  `tff.Computation` and produce the appropriate pieces of logic.
  """

  def __init__(
      self,
      type_signature: computation_types.FunctionType,
      server_prepare: computation_impl.ConcreteComputation,
      server_to_client_broadcast: computation_impl.ConcreteComputation,
      client_work: computation_impl.ConcreteComputation,
      client_to_server_aggregation: computation_impl.ConcreteComputation,
      server_result: computation_impl.ConcreteComputation,
  ):
    """Constructs a representation of a round for a federated learning system.

    Note: All the computations supplied here as arguments must be TFF Lambda
    Computations (as defined in `computation.proto`).

    Args:
      type_signature: The type signature of the corresponding `tff.Computation`
        that is equivalent to the pieces of logic encoded in this data
        structure.
      server_prepare: The computation that prepares the input for the clients
        and computes intermediate server state that will be needed in later
        stages.
      server_to_client_broadcast: The computation that represents broadcasting
        data from the server to the clients.
      client_work: The client-side work computation.
      client_to_server_aggregation: The computation that aggregates the client
        results at the server, potentially using intermediate state generated by
        the server_prepare phase.
      server_result: The computation that combines the aggregated client results
        and the intermediate state for the round to produce the new server state
        as well as server-side output.
    """
    for label, comp in (
        ('server_prepare', server_prepare),
        ('server_to_client_broadcast', server_to_client_broadcast),
        ('client_work', client_work),
        ('client_to_server_aggregation', client_to_server_aggregation),
        ('server_result', server_result),
    ):
      _check_lambda_computation(label, comp)

    # The server_prepare function should take an arbitrary length input that
    # represents the server state and produce 2 results (data to broadcast and
    # temporary state). It should contain only server placements.
    _check_returns_tuple('server_prepare', server_prepare, length=2)
    tree_analysis.check_has_single_placement(
        server_prepare.to_building_block(), placements.SERVER
    )

    # The broadcast function can take an arbitrary number of inputs and produce
    # an arbitrary number of outputs. It should contain a block of locals that
    # are exclusively broadcast-type intrinsics and should return the results of
    # these intrinsics in the order they are computed.
    expected_return_references = []
    for (
        local_name,
        local_value,
    ) in server_to_client_broadcast.to_building_block().result.locals:
      local_value.check_call()
      local_value.function.check_intrinsic()
      if not local_value.function.intrinsic_def().broadcast_kind:
        raise ValueError(
            'Expected only broadcast intrinsics but found {}'.format(
                local_value.function.uri
            )
        )
      _check_flattened_intrinsic_args_are_selections(
          local_value.argument,
          server_to_client_broadcast.to_building_block().parameter_name,
      )
      expected_return_references.append(local_name)
    server_to_client_broadcast.to_building_block().result.result.check_struct()
    return_references = [
        reference.name
        for reference in server_to_client_broadcast.to_building_block().result.result
    ]
    if expected_return_references != return_references:
      raise ValueError(
          'Expected the broadcast function to return references {} but '
          'received {}'.format(expected_return_references, return_references)
      )

    # The client_work function should take 2 inputs (client data and broadcasted
    # data) and produce an output of arbitrary length that represents the data
    # to aggregate. It should contain only CLIENTS placements.
    _check_accepts_tuple('client_work', client_work, length=2)
    tree_analysis.check_has_single_placement(
        client_work.to_building_block(), placements.CLIENTS
    )

    # The client_to_server_aggregation function should take 2 inputs (temporary
    # state and client results) and produce an output of arbitrary length that
    # represents the aggregated data. It should contain a block of locals that
    # are exclusively aggregation-type intrinsics and should return the results
    # of these intrinsics in the order they are computed.
    _check_accepts_tuple(
        'client_to_server_aggregation', client_to_server_aggregation, length=2
    )
    expected_return_references = []
    for (
        local_name,
        local_value,
    ) in client_to_server_aggregation.to_building_block().result.locals:
      local_value.check_call()
      local_value.function.check_intrinsic()
      if not local_value.function.intrinsic_def().aggregation_kind:
        raise ValueError(
            'Expected only aggregation intrinsics but found {}'.format(
                local_value.function.uri
            )
        )
      _check_flattened_intrinsic_args_are_selections(
          local_value.argument,
          client_to_server_aggregation.to_building_block().parameter_name,
      )
      expected_return_references.append(local_name)
    client_to_server_aggregation.to_building_block().result.result.check_struct()
    return_references = [
        reference.name
        for reference in client_to_server_aggregation.to_building_block().result.result
    ]
    if expected_return_references != return_references:
      raise ValueError(
          'Expected the aggregation function to return references {} but '
          'received {}'.format(expected_return_references, return_references)
      )

    # The server_result function should take 2 inputs (temporary state and
    # aggregate client data) and produce 2 outputs (new server state and server
    # output). It should contain only SERVER placements.
    _check_accepts_tuple('server_result', server_result, length=2)
    _check_returns_tuple('server_result', server_result, length=2)
    tree_analysis.check_has_single_placement(
        server_result.to_building_block(), placements.SERVER
    )

    # The broadcast input data types in the 'server_prepare' result and
    # 'server_to_client_broadcast' argument should match.
    if not _is_assignable_from_or_both_none(
        server_to_client_broadcast.type_signature.parameter,
        server_prepare.type_signature.result[0],
    ):
      raise TypeError(
          'The `server_to_client_broadcast` computation expects an argument '
          'type {} that does not match the corresponding result type {} of '
          '`server_prepare`.'.format(
              server_to_client_broadcast.type_signature.parameter,
              server_prepare.type_signature.result[0],
          )
      )

    # The broadcast output data types in the 'server_to_client_broadcast' result
    # and 'client_work' argument should match.
    if not _is_assignable_from_or_both_none(
        client_work.type_signature.parameter[1],
        server_to_client_broadcast.type_signature.result,
    ):
      raise TypeError(
          'The `client_work` computation expects an argument type {} '
          'that does not match the corresponding result type {} of '
          '`server_to_client_broadcast`.'.format(
              client_work.type_signature.parameter[1],
              server_to_client_broadcast.type_signature.result,
          )
      )

    # The aggregation input data types in the 'client_work' result and
    # 'client_to_server_aggregation' argument should match.
    if not _is_assignable_from_or_both_none(
        client_to_server_aggregation.type_signature.parameter[1],
        client_work.type_signature.result,
    ):
      raise TypeError(
          'The `client_to_server_aggregation` computation expects an argument '
          'type {} that does not match the corresponding result type {} of '
          '`client_work`.'.format(
              client_to_server_aggregation.type_signature.parameter[1],
              client_work.type_signature.result,
          )
      )

    # The aggregation output data types in the 'client_to_server_aggregation'
    # result and 'server_result' argument should match.
    if not _is_assignable_from_or_both_none(
        server_result.type_signature.parameter[1],
        client_to_server_aggregation.type_signature.result,
    ):
      raise TypeError(
          'The `server_result` computation expects an argument type {} '
          'that does not match the corresponding result type {} of '
          '`client_to_server_aggregation`.'.format(
              server_result.type_signature.parameter[1],
              client_to_server_aggregation.type_signature.result,
          )
      )

    # The temporary state data types in the 'server_prepare' result,
    # 'client_to_server_aggregation' argument, and 'server_result' argument
    # should match.
    if not _is_assignable_from_or_both_none(
        client_to_server_aggregation.type_signature.parameter[0],
        server_prepare.type_signature.result[1],
    ) or not _is_assignable_from_or_both_none(
        server_result.type_signature.parameter[0],
        server_prepare.type_signature.result[1],
    ):
      raise TypeError(
          'The `client_to_server_aggregation` computation expects an argument '
          'type {} and the `server_result` computation expects an argument '
          'type {} that does not match the corresponding result type {} of '
          '`server_prepare`.'.format(
              client_to_server_aggregation.type_signature.parameter[0],
              server_result.type_signature.parameter[0],
              server_prepare.type_signature.result[1],
          )
      )

    # The server state data types in the original computation argument, the
    # 'server_prepare' argument, the 'server_result' result, and the original
    # computation result should match.
    if (
        not _is_assignable_from_or_both_none(
            server_prepare.type_signature.parameter, type_signature.parameter[0]
        )
        or not _is_assignable_from_or_both_none(
            server_result.type_signature.result[0], type_signature.parameter[0]
        )
        or not _is_assignable_from_or_both_none(
            type_signature.result[0], type_signature.parameter[0]
        )
    ):
      raise TypeError(
          'The original computation argument type {}, '
          'the `server_prepare` computation argument type {}, '
          'the `server_result` computation result type {}, '
          'and the original computation result type {} should all match.'
          .format(
              type_signature.parameter[0],
              server_prepare.type_signature.parameter,
              server_result.type_signature.result[0],
              type_signature.result[0],
          )
      )

    # The data types of the client data in the original computation argument
    # and the 'client_work' argument should match.
    if not _is_assignable_from_or_both_none(
        client_work.type_signature.parameter[0], type_signature.parameter[1]
    ):
      raise TypeError(
          'The `client_work` computation expects an argument type {} '
          'that does not match the original computation argument type {}.'
          .format(
              client_work.type_signature.parameter[0],
              type_signature.parameter[1],
          )
      )

    # The server-side output data types in the original computation result and
    # the 'server_result' result should match.
    if not _is_assignable_from_or_both_none(
        server_result.type_signature.result[1], type_signature.result[1]
    ):
      raise TypeError(
          'The `server_result` computation expects an result type {} '
          'that does not match the original computation result type {}.'.format(
              server_result.type_signature.result[1], type_signature.result[1]
          )
      )

    self._type_signature = type_signature
    self._server_prepare = server_prepare
    self._server_to_client_broadcast = server_to_client_broadcast
    self._client_work = client_work
    self._client_to_server_aggregation = client_to_server_aggregation
    self._server_result = server_result

  @property
  def type_signature(self) -> computation_types.FunctionType:
    """Returns the TFF type of the equivalent `tff.Computation`."""
    return self._type_signature

  @property
  def server_prepare(self) -> computation_impl.ConcreteComputation:
    return self._server_prepare

  @property
  def server_to_client_broadcast(self) -> computation_impl.ConcreteComputation:
    return self._server_to_client_broadcast

  @property
  def client_work(self) -> computation_impl.ConcreteComputation:
    return self._client_work

  @property
  def client_to_server_aggregation(
      self,
  ) -> computation_impl.ConcreteComputation:
    return self._client_to_server_aggregation

  @property
  def server_result(self) -> computation_impl.ConcreteComputation:
    return self._server_result

  def summary(self, print_fn: Callable[..., None] = print) -> None:
    """Prints a string summary of the `DistributeAggregateForm`.

    Args:
      print_fn: Print function to use. It will be called on each line of the
        summary in order to capture the string summary.
    """
    for label, comp in (
        ('server_prepare', self.server_prepare),
        ('server_to_client_broadcast', self.server_to_client_broadcast),
        ('client_work', self.client_work),
        ('client_to_server_aggregation', self.client_to_server_aggregation),
        ('server_result', self.server_result),
    ):
      # Add sufficient padding to align first column;
      # len('client_to_server_aggregation') == 28
      print_fn(
          '{:<28}: {}'.format(
              label, comp.type_signature.compact_representation()
          )
      )
