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

from tensorflow_federated.python.program.client_id_data_source import ClientIdDataSource
from tensorflow_federated.python.program.client_id_data_source import ClientIdDataSourceIterator
from tensorflow_federated.python.program.data_source import Capability
from tensorflow_federated.python.program.data_source import FederatedDataSource
from tensorflow_federated.python.program.data_source import FederatedDataSourceIterator
from tensorflow_federated.python.program.dataset_data_source import DatasetDataSource
from tensorflow_federated.python.program.dataset_data_source import DatasetDataSourceIterator
from tensorflow_federated.python.program.federated_context import check_in_federated_context
from tensorflow_federated.python.program.federated_context import contains_only_server_placed_data
from tensorflow_federated.python.program.federated_context import FederatedContext
from tensorflow_federated.python.program.file_program_state_manager import FileProgramStateManager
from tensorflow_federated.python.program.file_release_manager import CSVFileReleaseManager
from tensorflow_federated.python.program.file_release_manager import CSVSaveMode
from tensorflow_federated.python.program.file_release_manager import FileReleaseManagerIncompatibleFileError
from tensorflow_federated.python.program.file_release_manager import FileReleaseManagerPermissionDeniedError
from tensorflow_federated.python.program.file_release_manager import SavedModelFileReleaseManager
from tensorflow_federated.python.program.logging_release_manager import LoggingReleaseManager
from tensorflow_federated.python.program.memory_release_manager import MemoryReleaseManager
from tensorflow_federated.python.program.native_platform import AwaitableValueReference
from tensorflow_federated.python.program.native_platform import NativeFederatedContext
from tensorflow_federated.python.program.prefetching_data_source import FetchedValue
from tensorflow_federated.python.program.prefetching_data_source import PrefetchingDataSource
from tensorflow_federated.python.program.prefetching_data_source import PrefetchingDataSourceIterator
from tensorflow_federated.python.program.program_state_manager import ProgramStateManager
from tensorflow_federated.python.program.program_state_manager import ProgramStateManagerStateAlreadyExistsError
from tensorflow_federated.python.program.program_state_manager import ProgramStateManagerStateNotFoundError
from tensorflow_federated.python.program.program_state_manager import ProgramStateManagerStructureError
from tensorflow_federated.python.program.release_manager import FilteringReleaseManager
from tensorflow_federated.python.program.release_manager import GroupingReleaseManager
from tensorflow_federated.python.program.release_manager import ReleaseManager
from tensorflow_federated.python.program.tensorboard_release_manager import TensorBoardReleaseManager
from tensorflow_federated.python.program.value_reference import MaterializablePythonType
from tensorflow_federated.python.program.value_reference import MaterializableTffType
from tensorflow_federated.python.program.value_reference import MaterializableValueReference
from tensorflow_federated.python.program.value_reference import materialize_value
