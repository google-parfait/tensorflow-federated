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

  This function should be used when we wish to take dependency on some behavior
  present in newer but not older versions of TensorFlow. When adding a usage
  of this, please perform the following steps:

  1. File a bug to clean up the usage of this check--once the latest released
     TensorFlow version contains this behavior, we want to remove it.

  2. Log in the true and false cases, and ensure that the default behavior is
     the one you expect.

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
