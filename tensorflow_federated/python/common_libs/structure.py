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

Struct = structure.Struct
name_list = structure.name_list
to_elements = structure.to_elements
to_odict = structure.to_odict
to_odict_or_tuple = structure.to_odict_or_tuple
flatten = structure.flatten
pack_sequence_as = structure.pack_sequence_as
map_structure = structure.map_structure
from_container = structure.from_container
has_field = structure.has_field
update_struct = structure.update_struct
