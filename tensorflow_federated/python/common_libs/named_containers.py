# Copyright 2022, The TensorFlow Federated Authors.
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
# limitations under the License.
"""Shared utilities for container manipulation."""

import collections
import dataclasses
from typing import Any

import attr

from tensorflow_federated.python.common_libs import py_typecheck


def dataclass_to_odict(dataclass_obj: Any) -> collections.OrderedDict[str, Any]:
  """Shallow-copies a dataclass instance to an ordered dict."""
  py_typecheck.check_dataclass(dataclass_obj)
  # dataclasses guarantee field ordering.
  fields = dataclasses.fields(dataclass_obj)
  odict = collections.OrderedDict()
  for field in fields:
    odict[field.name] = getattr(dataclass_obj, field.name)
  return odict


def attrs_class_to_odict(
    attr_class_obj: Any,
) -> collections.OrderedDict[Any, Any]:
  """Shallow-copies an attr-class object to an ordered dict."""
  py_typecheck.check_attrs(attr_class_obj)
  odict = attr.asdict(
      attr_class_obj, dict_factory=collections.OrderedDict, recurse=False
  )
  return odict  # pytype:disable=bad-return-type
