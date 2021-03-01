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
from tensorflow_federated.python.common_libs import structure as struct_lib


def update_state(structure, **kwargs):
  """Constructs a new `structure` with new values for fields in `kwargs`.

  This is a helper method for working structured objects in a functional manner.
  This method will create a new structure where the fields named by keys in
  `kwargs` replaced with the associated values.

  NOTE: This method only works on the first level of `structure`, and does not
  recurse in the case of nested structures. A field that is itself a structure
  can be replaced with another structure.

  Args:
    structure: The structure with named fields to update.
    **kwargs: The list of key-value pairs of fields to update in `structure`.

  Returns:
    A new instance of the same type of `structure`, with the fields named
    in the keys of `**kwargs` replaced with the associated values.

  Raises:
    KeyError: If kwargs contains a field that is not in structure.
    TypeError: If structure is not a structure with named fields.
  """
  if not (py_typecheck.is_named_tuple(structure) or
          py_typecheck.is_attrs(structure) or
          isinstance(structure, (struct_lib.Struct, collections.abc.Mapping))):
    raise TypeError('`structure` must be a structure with named fields (e.g. '
                    'dict, attrs class, collections.namedtuple, '
                    'tff.structure.Struct), but found {}'.format(
                        type(structure)))
  if isinstance(structure, struct_lib.Struct):
    elements = [(k, v) if k not in kwargs else (k, kwargs.pop(k))
                for k, v in struct_lib.iter_elements(structure)]
    if kwargs:
      raise KeyError(f'`structure` does not contain fields named {kwargs}')
    return struct_lib.Struct(elements)
  elif py_typecheck.is_named_tuple(structure):
    # In Python 3.8 and later `_asdict` no longer return OrdereDict, rather a
    # regular `dict`, so we wrap here to get consistent types across Python
    # version.s
    d = collections.OrderedDict(structure._asdict())
  elif py_typecheck.is_attrs(structure):
    d = attr.asdict(structure, dict_factory=collections.OrderedDict)
  else:
    for key in kwargs:
      if key not in structure:
        raise KeyError(
            'structure does not contain a field named "{!s}"'.format(key))
    d = structure
  d.update(kwargs)
  if isinstance(structure, collections.abc.Mapping):
    return d
  return type(structure)(**d)
