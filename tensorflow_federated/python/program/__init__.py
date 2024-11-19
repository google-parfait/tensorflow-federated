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
"""Libraries for creating federated programs."""

import federated_language
# pylint: disable=g-importing-member
from tensorflow_federated.python.program.client_id_data_source import ClientIdDataSource
from tensorflow_federated.python.program.client_id_data_source import ClientIdDataSourceIterator

FederatedDataSource = federated_language.program.FederatedDataSource
FederatedDataSourceIterator = (
    federated_language.program.FederatedDataSourceIterator
)
from tensorflow_federated.python.program.dataset_data_source import DatasetDataSource
from tensorflow_federated.python.program.dataset_data_source import DatasetDataSourceIterator

check_in_federated_context = (
    federated_language.program.check_in_federated_context
)
ComputationArg = federated_language.program.ComputationArg
contains_only_server_placed_data = (
    federated_language.program.contains_only_server_placed_data
)
FederatedContext = federated_language.program.FederatedContext
from tensorflow_federated.python.program.file_program_state_manager import FileProgramStateManager
from tensorflow_federated.python.program.file_release_manager import CSVFileReleaseManager
from tensorflow_federated.python.program.file_release_manager import CSVKeyFieldnameNotFoundError
from tensorflow_federated.python.program.file_release_manager import CSVSaveMode
from tensorflow_federated.python.program.file_release_manager import SavedModelFileReleaseManager

LoggingReleaseManager = federated_language.program.LoggingReleaseManager
MemoryReleaseManager = federated_language.program.MemoryReleaseManager
from tensorflow_federated.python.program.native_platform import NativeFederatedContext
from tensorflow_federated.python.program.native_platform import NativeValueReference

ProgramStateExistsError = federated_language.program.ProgramStateExistsError
ProgramStateManager = federated_language.program.ProgramStateManager
ProgramStateNotFoundError = federated_language.program.ProgramStateNotFoundError
ProgramStateStructure = federated_language.program.ProgramStateStructure
ProgramStateValue = federated_language.program.ProgramStateValue
DelayedReleaseManager = federated_language.program.DelayedReleaseManager
FilteringReleaseManager = federated_language.program.FilteringReleaseManager
GroupingReleaseManager = federated_language.program.GroupingReleaseManager
NotFilterableError = federated_language.program.NotFilterableError
PeriodicReleaseManager = federated_language.program.PeriodicReleaseManager
ReleasableStructure = federated_language.program.ReleasableStructure
ReleasableValue = federated_language.program.ReleasableValue
ReleaseManager = federated_language.program.ReleaseManager
from tensorflow_federated.python.program.tensorboard_release_manager import TensorBoardReleaseManager

MaterializableStructure = federated_language.program.MaterializableStructure
MaterializableTypeSignature = (
    federated_language.program.MaterializableTypeSignature
)
MaterializableValue = federated_language.program.MaterializableValue
MaterializableValueReference = (
    federated_language.program.MaterializableValueReference
)
materialize_value = federated_language.program.materialize_value
MaterializedStructure = federated_language.program.MaterializedStructure
MaterializedValue = federated_language.program.MaterializedValue
# pylint: enable=g-importing-member
