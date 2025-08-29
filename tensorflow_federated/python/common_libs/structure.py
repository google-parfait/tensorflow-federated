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
"""Container for structures with named and/or unnamed fields."""

from federated_language.common_libs import structure
from tensorflow_federated.python.common_libs import deprecation


Struct = deprecation.deprecated(
    "`tff.structure.Struct` is deprecated, use a Python container instead."
)(structure.Struct)
_to_elements = deprecation.deprecated(
    "`tff.structure.to_elements` is deprecated, use a Python container instead."
)(structure.to_elements)
to_odict_or_tuple = deprecation.deprecated(
    "`tff.structure.to_odict_or_tuple` is deprecated, use a Python container"
    " instead."
)(structure.to_odict_or_tuple)
flatten = deprecation.deprecated(
    "`tff.structure.flatten` is deprecated, use a Python container instead."
)(structure.flatten)
pack_sequence_as = deprecation.deprecated(
    "`tff.structure.pack_sequence_as` is deprecated, use a Python container"
    " instead."
)(structure.pack_sequence_as)
_map_structure = deprecation.deprecated(
    "`tff.structure.map_structure` is deprecated, use a Python container"
    " instead."
)(structure.map_structure)
_from_container = deprecation.deprecated(
    "`tff.structure.from_container` is deprecated, use a Python container"
    " instead."
)(structure.from_container)
_update_struct = deprecation.deprecated(
    "`tff.structure.update_struct` is deprecated, use a Python container"
    " instead."
)(structure.update_struct)
