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
      type_spec: The type of the intrinsic, either an instance of types.Type or
        something convertible to it.
    """
    py_typecheck.check_type(name, string_types)
    py_typecheck.check_type(uri, string_types)
    self._name = str(name)
    self._uri = str(uri)
    self._type_signature = types.to_type(type_spec)

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


FEDERATED_BROADCAST = IntrinsicDef(
    'FEDERATED_BROADCAST',
    'federated_broadcast',
    types.FunctionType(
        types.FederatedType(types.AbstractType('T'), placements.SERVER, True),
        types.FederatedType(types.AbstractType('T'), placements.CLIENTS, True)))


FEDERATED_MAP = IntrinsicDef(
    'FEDERATED_MAP',
    'federated_map',
    types.FunctionType(
        [types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
         types.FunctionType(types.AbstractType('T'), types.AbstractType('U'))],
        types.FederatedType(types.AbstractType('U'), placements.CLIENTS)))


FEDERATED_SUM = IntrinsicDef(
    'FEDERATED_SUM',
    'federated_sum',
    types.FunctionType(
        types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
        types.FederatedType(types.AbstractType('T'), placements.SERVER, True)))


FEDERATED_ZIP = IntrinsicDef(
    'FEDERATED_ZIP',
    'federated_zip',
    types.FunctionType(
        [types.FederatedType(types.AbstractType('T'), placements.CLIENTS),
         types.FederatedType(types.AbstractType('U'), placements.CLIENTS)],
        types.FederatedType(
            [types.AbstractType('T'), types.AbstractType('U')],
            placements.CLIENTS)))
