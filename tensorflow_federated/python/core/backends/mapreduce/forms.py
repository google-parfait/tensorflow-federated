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

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.types import computation_types


def _check_tensorflow_computation(label, comp):
  py_typecheck.check_type(comp, computation_base.Computation, label)
  comp_proto = computation_impl.ComputationImpl.get_proto(comp)
  which_comp = comp_proto.WhichOneof('computation')
  if which_comp != 'tensorflow':
    raise TypeError('Expected all computations supplied as arguments to '
                    'be plain TensorFlow, found {}.'.format(which_comp))


def _is_assignable_from_or_both_none(first, second):
  if first is None:
    return second is None
  return first.is_assignable_from(second)


def _is_two_tuple(t: computation_types.Type) -> bool:
  return t.is_struct() and len(t) == 2


def _check_accepts_two_tuple(label: str, comp: computation_base.Computation):
  param_type = comp.type_signature.parameter
  if not _is_two_tuple(param_type):
    raise TypeError(
        f'The `{label}` computation accepts a parameter of type\n{param_type}\n'
        'that is not a two-tuple.')


def _check_returns_two_tuple(label: str, comp: computation_base.Computation):
  result_type = comp.type_signature.result
  if not _is_two_tuple(result_type):
    raise TypeError(
        f'The `{label}` computation returns a result of type\n{result_type}\n'
        'that is not a two-tuple.')


class BroadcastForm(object):
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

  def __init__(self,
               compute_server_context,
               client_processing,
               server_data_label=None,
               client_data_label=None):
    for label, comp in (
        ('compute_server_context', compute_server_context),
        ('client_processing', client_processing),
    ):
      _check_tensorflow_computation(label, comp)
    _check_accepts_two_tuple('client_processing', client_processing)
    client_first_arg_type = client_processing.type_signature.parameter[0]
    server_context_type = compute_server_context.type_signature.result
    if not _is_assignable_from_or_both_none(client_first_arg_type,
                                            server_context_type):
      raise TypeError(
          'The `client_processing` computation expects an argument tuple with '
          f'type\n{client_first_arg_type}\nas the first element (the context '
          'type from the server), which does not match the result type\n'
          f'{server_context_type}\n of `compute_server_context`.')
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
      print_fn('{:<22}: {}'.format(
          label, comp.type_signature.compact_representation()))


class MapReduceForm(object):
  """Standardized representation of logic deployable to MapReduce-like systems.

  This class docstring describes the purpose of `MapReduceForm` as a data
  structure; for a discussion of the conceptual content of an instance `mrf` of
  `MapReduceForm`, including how precisely it maps to a single federated round,
  see the [package-level docstring](
  https://www.tensorflow.org/federated/api_docs/python/tff/backends/mapreduce).

  This standardized representation can be used to describe a range of iterative
  processes representable as a single round of MapReduce-like processing, and
  deployable to MapReduce-like systems that are only capable of executing plain
  TensorFlow code.

  Non-iterative processes, or processes that do not originate at the server can
  be described by `MapReduceForm`, as well as degenerate cases like computations
  which use exclusively one of the two possible aggregation paths.

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

  The individual TensorFlow computations that constitute an iterative process
  in this form are supplied as constructor arguments. Generally, this class will
  not be instantiated by a programmer directly but targeted by a sequence of
  transformations that take a `tff.templates.IterativeProcess` and produce the
  appropriate pieces of logic.

  """

  def __init__(self,
               initialize,
               prepare,
               work,
               zero,
               accumulate,
               merge,
               report,
               bitwidth,
               update,
               server_state_label=None,
               client_data_label=None):
    """Constructs a representation of a MapReduce-like iterative process.

    Note: All the computations supplied here as arguments must be TensorFlow
    computations, i.e., instances of `tff.Computation` constructed by the
    `tff.tf_computation` decorator/wrapper.

    Args:
      initialize: The computation that produces the initial server state.
      prepare: The computation that prepares the input for the clients.
      work: The client-side work computation.
      zero: The computation that produces the initial state for accumulators.
      accumulate: The computation that adds a client update to an accumulator.
      merge: The computation to use for merging pairs of accumulators.
      report: The computation that produces the final server-side aggregate for
        the top level accumulator (the global update).
      bitwidth: The computation that produces the bitwidth for secure sum.
      update: The computation that takes the global update and the server state
        and produces the new server state, as well as server-side output.
      server_state_label: Optional string label for the server state.
      client_data_label: Optional string label for the client data.

    Raises:
      TypeError: If the Python or TFF types of the arguments are invalid or not
        compatible with each other.
      AssertionError: If the manner in which the given TensorFlow computations
        are represented by TFF does not match what this code is expecting (this
        is an internal error that requires code update).
    """
    for label, comp in (
        ('initialize', initialize),
        ('prepare', prepare),
        ('work', work),
        ('zero', zero),
        ('accumulate', accumulate),
        ('merge', merge),
        ('report', report),
        ('bitwidth', bitwidth),
        ('update', update),
    ):
      _check_tensorflow_computation(label, comp)

    prepare_arg_type = prepare.type_signature.parameter
    init_result_type = initialize.type_signature.result
    if not _is_assignable_from_or_both_none(prepare_arg_type, init_result_type):
      raise TypeError(
          'The `prepare` computation expects an argument of type {}, '
          'which does not match the result type {} of `initialize`.'.format(
              prepare_arg_type, init_result_type))

    _check_accepts_two_tuple('work', work)
    work_2nd_arg_type = work.type_signature.parameter[1]
    prepare_result_type = prepare.type_signature.result
    if not _is_assignable_from_or_both_none(work_2nd_arg_type,
                                            prepare_result_type):
      raise TypeError(
          'The `work` computation expects an argument tuple with type {} as '
          'the second element (the initial client state from the server), '
          'which does not match the result type {} of `prepare`.'.format(
              work_2nd_arg_type, prepare_result_type))

    _check_returns_two_tuple('work', work)

    py_typecheck.check_len(accumulate.type_signature.parameter, 2)
    accumulate.type_signature.parameter[0].check_assignable_from(
        zero.type_signature.result)
    accumulate_2nd_arg_type = accumulate.type_signature.parameter[1]
    work_client_update_type = work.type_signature.result[0]
    if not _is_assignable_from_or_both_none(accumulate_2nd_arg_type,
                                            work_client_update_type):

      raise TypeError(
          'The `accumulate` computation expects a second argument of type {}, '
          'which does not match the expected {} as implied by the type '
          'signature of `work`.'.format(accumulate_2nd_arg_type,
                                        work_client_update_type))
    accumulate.type_signature.parameter[0].check_assignable_from(
        accumulate.type_signature.result)

    py_typecheck.check_len(merge.type_signature.parameter, 2)
    merge.type_signature.parameter[0].check_assignable_from(
        accumulate.type_signature.result)
    merge.type_signature.parameter[1].check_assignable_from(
        accumulate.type_signature.result)
    merge.type_signature.parameter[0].check_assignable_from(
        merge.type_signature.result)

    report.type_signature.parameter.check_assignable_from(
        merge.type_signature.result)

    expected_update_parameter_type = computation_types.to_type([
        initialize.type_signature.result,
        [report.type_signature.result, work.type_signature.result[1]],
    ])
    if not _is_assignable_from_or_both_none(update.type_signature.parameter,
                                            expected_update_parameter_type):
      raise TypeError(
          'The `update` computation expects an argument of type {}, '
          'which does not match the expected {} as implied by the type '
          'signatures of `initialize`, `report`, and `work`.'.format(
              update.type_signature.parameter, expected_update_parameter_type))

    _check_returns_two_tuple('update', update)

    updated_state_type = update.type_signature.result[0]
    if not prepare_arg_type.is_assignable_from(updated_state_type):
      raise TypeError(
          'The `update` computation returns a result tuple whose first element '
          f'(the updated state type of the server) is type:\n'
          f'{updated_state_type}\n'
          f'which is not assignable to the state parameter type of `prepare`:\n'
          f'{prepare_arg_type}')

    self._initialize = initialize
    self._prepare = prepare
    self._work = work
    self._zero = zero
    self._accumulate = accumulate
    self._merge = merge
    self._report = report
    self._bitwidth = bitwidth
    self._update = update

    if server_state_label is not None:
      py_typecheck.check_type(server_state_label, str)
    self._server_state_label = server_state_label
    if client_data_label is not None:
      py_typecheck.check_type(client_data_label, str)
    self._client_data_label = client_data_label

  @property
  def initialize(self):
    return self._initialize

  @property
  def prepare(self):
    return self._prepare

  @property
  def work(self):
    return self._work

  @property
  def zero(self):
    return self._zero

  @property
  def accumulate(self):
    return self._accumulate

  @property
  def merge(self):
    return self._merge

  @property
  def report(self):
    return self._report

  @property
  def bitwidth(self):
    return self._bitwidth

  @property
  def update(self):
    return self._update

  @property
  def server_state_label(self):
    return self._server_state_label

  @property
  def client_data_label(self):
    return self._client_data_label

  @property
  def securely_aggregates_tensors(self) -> bool:
    """Whether the `MapReduceForm` uses secure aggregation."""
    # Tensors aggregated over `federated_secure_sum` are output in the second
    # tuple element from `work()`.
    work_result_type = self.work.type_signature.result
    assert len(work_result_type) == 2
    return not work_result_type[1].is_equivalent_to(
        computation_types.StructType([]))

  def summary(self, print_fn=print):
    """Prints a string summary of the `MapReduceForm`.

    Args:
      print_fn: Print function to use. It will be called on each line of the
        summary in order to capture the string summary.
    """
    for label, comp in (
        ('initialize', self.initialize),
        ('prepare', self.prepare),
        ('work', self.work),
        ('zero', self.zero),
        ('accumulate', self.accumulate),
        ('merge', self.merge),
        ('report', self.report),
        ('bitwidth', self.bitwidth),
        ('update', self.update),
    ):
      # Add sufficient padding to align first column; len('initialize') == 10
      print_fn('{:<10}: {}'.format(
          label, comp.type_signature.compact_representation()))
