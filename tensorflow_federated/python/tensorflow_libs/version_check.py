# Lint as: python3
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
"""Methods for conditioning code on the Tensorflow version at runtime."""

import semantic_version


def is_tensorflow_version_newer(version: str, tf_module):
  """Determines the Tensorflow module is newer than a specified version.

  Args:
    version: a `str` in the semantic versioning format 'major.minor.patch'.
    tf_module: the TensorFlow module to assert the version against.

  Returns:
    `True` iff `tf_module` is versions at or newer than `version` string, or
  the is determined to be a nightly pre-release version of TensorFlow.
  """
  tf_version = semantic_version.Version(tf_module.version.VERSION)
  if tf_version.prerelease:
    # tf-nightly uses versions like MAJOR.MINOR.PATCH-devYYYYMMDD
    if tf_version.prerelease[0].startswith('dev2020'):
      return True
  version_spec = semantic_version.Spec('>={v}'.format(v=version))
  return version_spec.match(tf_version)
