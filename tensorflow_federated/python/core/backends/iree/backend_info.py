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
"""Bundles of information about particular kinds of IREE backends."""

from tensorflow_federated.python.common_libs import py_typecheck


class BackendInfo(object):
  """A bundle of information about a particular kind of IREE backend."""

  def __init__(self, driver_name, target_name):
    """Creates an instance of this class.

    Args:
      driver_name: The name of the IREE driver to use, e.g., 'vulkan'. This must
        be a valid paameter to the IREE runtime config constructor.
      target_name: The name of the IREE compilation target, compatible with the
        selected driver. This must be a valid element of the list of
        `target_backends` supplied to an IREE compiler module's `compile()`.
    """
    # TODO(b/153499219): It would be great if we could somehow cheaply confirm
    # that these arguments are legitimate and compatible with each other.
    py_typecheck.check_type(driver_name, str)
    py_typecheck.check_type(target_name, str)
    self._driver_name = driver_name
    self._target_name = target_name

  @property
  def driver_name(self):
    return self._driver_name

  @property
  def target_name(self):
    return self._target_name


VMLA = BackendInfo('vmla', 'vmla')

VULKAN_SPIRV = BackendInfo('vulkan', 'vulkan-spirv')
