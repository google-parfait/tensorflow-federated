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
"""Defines errors used by multiple modules in the `tff.templates` package."""


class TemplateInitFnParamNotEmptyError(TypeError):
  """`TypeError` for `initialize_fn` having arguments."""
  pass


class TemplateStateNotAssignableError(TypeError):
  """`TypeError` for `state` not being assignable to expected `state`."""
  pass


class TemplateNotMeasuredProcessOutputError(TypeError):
  """`TypeError` for output of `next_fn` not being a `MeasuredProcessOutput`."""
  pass


class TemplateNextFnNumArgsError(TypeError):
  """`TypeError` for `next_fn` not having expected number of input arguments."""


class TemplateNotFederatedError(TypeError):
  """`TypeError` for template functions not being federated."""


class TemplatePlacementError(TypeError):
  """`TypeError` for template types not being placed as expected."""
