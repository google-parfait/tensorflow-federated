# Copyright 2018, The TensorFlow Federated Authors.
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
"""Definitions of all intrinsic for use within the system."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from six import string_types

from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import types


class IntrinsicDef(object):
  """Represents the definition of an intrinsic.

  This class represents the ultimayte source of ground truth about what kinds of
  intrinsics exist and what their types are. To be consuled by all other code
  that deals with intrinsics.
  """

  def __init__(self, name, uri, type_spec):
    """Constructs a definition of an intrinsic.

    Args:
      name: The short human-friendly name of this intrinsic.
      uri: The URI of this intrinsic.
      type_spec: The type of the intrinsic, which must be functional, either
        an instance of types.FunctionType or something convertible to it.
    """
    py_typecheck.check_type(name, string_types)
    py_typecheck.check_type(uri, string_types)
    type_spec = types.to_type(type_spec)
    py_typecheck.check_type(type_spec, types.FunctionType)
    self._name = str(name)
    self._uri = str(uri)
    self._type_signature = type_spec

  # TODO(b/113112885): Add support for an optional type checking function that
  # can verify whether this intrinsic is applicable to given kinds of arguments,
  # e.g., to allow sum-like functions to be applied only to arguments that are
  # composed of tensors as leaves of a possible nested structure.

  @property
  def name(self):
    return self._name

  @property
  def uri(self):
    return self._uri

  @property
  def type_signature(self):
    return self._type_signature

  def __str__(self):
    return self._name

  def __repr__(self):
    return 'IntrinsicDef(\'{}\')'.format(self._uri)


# TODO(b/113112885): Perhaps add a way for these to get auto-registered to
# enable things like lookup by URI, etc., similarly to how it's handled in the
# placement_literals.py.


# TODO(b/113112885): Define the generic equivalents of all operators below,
# i.e., intrinsics that support arbitrary placements, to allow the federated
# operators to be decomposed into expressions that might involve one or more
# layers of intermediate aggregators. The type signatures of these generic
# intrinsics are tentatively defined as follows:
#
# - Compute an aggregate using the 4-part aggregation operator interface:
#
#     generic_aggregate: <{T}@p,U,(<U,T>->U),(<U,U>->U),(U->R),q> -> R@q
#
# - Compute an unweighted average:
#
#     generic_average: <{T}@p,q> -> T@q
#
# - Broadcast an item:
#
#     generic_broadcast: <T@p,q> -> T@q
#
# - Materialize a federated value as a set of sequences at another placement,
#   with the participants at 'q' collecting from disjoint subsets of 'p' that
#   jointly cover all of 'p'.
#
#     generic_partial_collect: <{T}@p,q> -> {T*}@q
#
# - Materialize a federated value as a single sequence:
#
#     generic_collect: <{T}@p,q> -> T*@q
#
# - Pointwise mapping of constituents of a federated value:
#
#     generic_map: <{T}@p,(T->U)> -> {U}@p
#
# - Perform one-stage set reduction of a federated value with a given operator,
#   with the participants at 'q' reducing over disjoint subsets of 'p' that
#   jointly cover all of 'p'.
#
#     generic_partial_reduce: <{T}@p,U,(<U,T>->U),q> -> {U}@q
#
# - Perform complete set reduction of a federated value with a given operator:
#
#     generic_reduce: <{T}@p,U,(<U,T>->U),q> -> U@q
#
# - Select and agree on a single member consistuent of a federated value (this
#   is technically need to project {T}@SERVER to T@SERVER in a manner that is
#   formally consistent; a technicality that we do not expect to surface in the
#   user API).
#
#     generic_only: {T}@p -> T@p
#
# - Compute a partial sum of a value (for values with numeric constituents):
#
#     generic_partial_sum: <{T}@p,q> -> {T}@q
#
# - Compute a simple sum of a value (for values with numeric constituents):
#
#     generic_sum: <{T}@p,q> -> T@q
#
# - Compute an average weighted by a numeric non-complex scalar:
#
#     generic_weighted_average: <{T}@p,{U}@p,q> -> T@q
#
# - Transform a pair of federated values into a federated pair (technicality we
#   expect to bury through implicit convertions, TBD).
#
#     generic_zip: <{T}@p,{U}@p> -> {<T,U>}@p


# Computes an aggregate of client items (the first, {T}@CLIENTS-typed parameter)
# using a multi-stage process in which client items are first partially
# aggregated at an intermediate layer, then the partial aggregates are further
# combined, and finally projected into the result. This multi-stage process is
# parameterized by a four-part aggregation interface that consists of the
# following:
# a) The 'zero' in the algebra used at the initial stage (partial aggregation),
#    This is the second, U-typed parameter.
# b) The operator that accumulates T-typed client items into the U-typed partial
#    aggregates. This is the third, (<U,T>->U)-typed parameter.
# c) The operator that combines pairs of U-typed partial aggregates. This is the
#    fourth, (<U,U>->U)-typed parameter.
# d) The operator that projects the top-level aggregate into the final result.
#    This is the fifth, (U->R)-typed parameter.
#
# Conceptually, given a new literal INTERMEDIATE_AGGREGATORS in a single-layer
# aggregation architecture, one could define this intrinsic in terms of generic
# intrinsics defined above, as follows.
#
# @federated_computation
# def federated_aggregate(x, zero, accumulate, merge, report):
#   a = generic_partial_reduce(x, zero, accumulate, INTERMEDIATE_AGGREGATORS)
#   b = generic_reduce(a, zero, merge, SERVER)
#   c = generic_map(b, report)
#   return c
#
# Actual implementations might vary.
#
# Type signature: <{T}@CLIENTS,U,(U,T->U),(U,U->U),(U->R)> -> R@SERVER
FEDERATED_AGGREGATE = IntrinsicDef(
    'FEDERATED_AGGREGATE',
    'federated_aggregate',
    types.FunctionType(
        [types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
         types.AbstractType('U'),
         types.FunctionType(
             [types.AbstractType('U'), types.AbstractType('T')],
             types.AbstractType('U')),
         types.FunctionType(
             [types.AbstractType('U'), types.AbstractType('U')],
             types.AbstractType('U')),
         types.FunctionType(types.AbstractType('U'), types.AbstractType('R'))],
        types.FederatedType(types.AbstractType('R'), placements.SERVER, True)))


# Computes a simple (equally weighted) average of client items. Only supported
# for numeric tensor types, or composite structures made up of numeric types.
#
# Type signature: {T}@CLIENTS -> T@SERVER
FEDERATED_AVERAGE = IntrinsicDef(
    'FEDERATED_AVERAGE',
    'federated_average',
    types.FunctionType(
        types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
        types.FederatedType(types.AbstractType('T'), placements.SERVER, True)))


# Broadcasts a server item to all clients.
#
# Type signature: T@SERVER -> T@CLIENTS
FEDERATED_BROADCAST = IntrinsicDef(
    'FEDERATED_BROADCAST',
    'federated_broadcast',
    types.FunctionType(
        types.FederatedType(types.AbstractType('T'), placements.SERVER, True),
        types.FederatedType(types.AbstractType('T'), placements.CLIENTS, True)))


# Materializes client items as a sequence on the server.
#
# Type signature: {T}@CLIENTS -> T*@SERVER
FEDERATED_COLLECT = IntrinsicDef(
    'FEDERATED_COLLECT',
    'federated_collect',
    types.FunctionType(
        types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
        types.FederatedType(
            types.SequenceType(types.AbstractType('T')),
            placements.SERVER,
            True)))


# Maps member constituents of a client value pointwise uising a given mapping
# function that operates independently on each client.
#
# Type signature: <{T}@CLIENTS,(T->U)> -> {U}@CLIENTS
FEDERATED_MAP = IntrinsicDef(
    'FEDERATED_MAP',
    'federated_map',
    types.FunctionType(
        [types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
         types.FunctionType(types.AbstractType('T'), types.AbstractType('U'))],
        types.FederatedType(types.AbstractType('U'), placements.CLIENTS)))


# Reduces a set of member constituents of a client value using a given 'zero' in
# the algebra (i.e., the result of reducing an empty set) and a given reduction
# operator with the signature U,T->U that incorporates a single T-typed element
# into a U-typed result of partial reduction. In the special case of T = U, this
# corresponds to the classical notion of reduction of a set using a commutative
# associative binary operator. The generalized reduction operator (with T != U)
# must yield the same results when repeatedly applied on sequences of elements
# in any order.
#
# Type signature: <{T}@CLIENTS,U,(<U,T>->U)> -> U@SERVER
FEDERATED_REDUCE = IntrinsicDef(
    'FEDERATED_REDUCE',
    'federated_reduce',
    types.FunctionType(
        [types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
         types.AbstractType('U'),
         types.FunctionType(
             [types.AbstractType('U'), types.AbstractType('T')],
             types.AbstractType('U'))],
        types.FederatedType(types.AbstractType('U'), placements.SERVER, True)))


# Computes the sum of client values on the server. Only supported for numeric
# types, or nested structures made up of numeric types.
#
# Type signature: {T}@CLIENTS -> T@SERVER
FEDERATED_SUM = IntrinsicDef(
    'FEDERATED_SUM',
    'federated_sum',
    types.FunctionType(
        types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
        types.FederatedType(types.AbstractType('T'), placements.SERVER, True)))


# Computes a weighted average of client items. Only supported for numeric tensor
# types, or composite structures made up of numeric types. Weights must be
# simple scalar numeric (integer or floating point, not complex) tensor types.
# The types of weights and values must be compatible, i.e., multiplying and
# dividing member constituents of the value by weights should yield results of
# the same type as the type of these member constituents being weighted. Thus,
# for example, one may not supply values containing tf.int32 tensors, as the
# result of weighting such values is of a floating-point type. Casting must be
# injected, where appropriate, by the compiler before invoking this intrinsic.
#
# Type signature: <{T}@CLIENTS,{U}@CLIENTS> -> T@SERVER
FEDERATED_WEIGHTED_AVERAGE = IntrinsicDef(
    'FEDERATED_WEIGHTED_AVERAGE',
    'federated_weighted_average',
    types.FunctionType(
        [types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
         types.FederatedType(types.AbstractType('U'), placements.CLIENTS)],
        types.FederatedType(types.AbstractType('T'), placements.SERVER, True)))

# Zips a tuple of two federated types into a federated tuple.
#
# Type signature: <{T}@CLIENTS,{U}@CLIENTS> -> {<T,U>}@CLIENTS
FEDERATED_ZIP = IntrinsicDef(
    'FEDERATED_ZIP',
    'federated_zip',
    types.FunctionType(
        [types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
         types.FederatedType(types.AbstractType('U'), placements.CLIENTS)],
        types.FederatedType(
            [types.AbstractType('T'), types.AbstractType('U')],
            placements.CLIENTS)))
