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
"""Defines utility functions for constructing TFF computations."""

import collections

import attr

from tensorflow_federated.python.common_libs import py_typecheck


def update_state(state, **kwargs):
  """Returns a new `state` with new values for fields in `kwargs`.

  Args:
    state: the structure with named fields to update.
    **kwargs: the list of key-value pairs of fields to update in `state`.

  Raises:
    KeyError: if kwargs contains a field that is not in state.
    TypeError: if state is not a structure with named fields.
  """
  # TODO(b/129569441): Support Struct as well.
  if not (py_typecheck.is_named_tuple(state) or py_typecheck.is_attrs(state) or
          isinstance(state, collections.Mapping)):
    raise TypeError('state must be a structure with named fields (e.g. '
                    'dict, attrs class, collections.namedtuple), '
                    'but found {}'.format(type(state)))
  if py_typecheck.is_named_tuple(state):
    d = state._asdict()
  elif py_typecheck.is_attrs(state):
    d = attr.asdict(state, dict_factory=collections.OrderedDict)
  else:
    for key in kwargs:
      if key not in state:
        raise KeyError(
            'state does not contain a field named "{!s}"'.format(key))
    d = state
  d.update(kwargs)
  if isinstance(state, collections.Mapping):
    return d
  return type(state)(**d)

