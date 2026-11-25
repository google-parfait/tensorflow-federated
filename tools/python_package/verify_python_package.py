# Copyright 2026, The TensorFlow Federated Authors.
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
"""Verifies built TFF wheel metadata and installed C++ pybind bindings.

Usage:
  verify_python_package.py wheel <wheel_path>
  verify_python_package.py bindings
"""

from collections.abc import Sequence
import importlib.util
import os
import platform
import sys
import zipfile

_REQUIRED_PYTHON_SPECS = frozenset({'>=3.12', '<3.14'})
_REQUIRED_SHARED_LIBRARY_PREFIXES = (
    'tensorflow_federated/cc/core/impl/executors/executor_bindings',
    'tensorflow_federated/cc/core/impl/executors/tensorflow_bindings',
)
_FORBIDDEN_WHEEL_SUBSTRINGS = (
    'worker_binary',
    'cc/core/impl/aggregation/',
    'cc/core/impl/executor_stacks/executor_stack_bindings',
    'executor_test_utils_bindings',
)
_CC_PACKAGE_PREFIX = 'tensorflow_federated/cc/'
_WHEEL_METADATA_SUFFIX = '.dist-info/WHEEL'
_PACKAGE_METADATA_SUFFIX = '.dist-info/METADATA'
_LINUX_PLATFORM_PREFIX = 'manylinux_2_28_'

_INVALID_VALUE_REF = 999999
_NOT_FOUND_STATUS_CODE = 5
_UNAVAILABLE_STATUS_CODE = 14
_SINGLE_CLIENT = 1


def verify_wheel(wheel_path: str) -> None:
  """Verifies wheel filename tags, metadata headers, and archive contents."""
  wheel_filename = os.path.basename(wheel_path)
  py_tag = f'cp{sys.version_info[0]}{sys.version_info[1]}'
  expected_abi_tag = f'{py_tag}-{py_tag}-'

  assert expected_abi_tag in wheel_filename, (
      f'Expected wheel filename {wheel_filename!r} to contain '
      f'{expected_abi_tag!r}'
  )
  expected_plat_tag = None
  if platform.system() == 'Linux':
    expected_plat_tag = f'{_LINUX_PLATFORM_PREFIX}{platform.machine()}'
    assert wheel_filename.endswith(f'-{expected_plat_tag}.whl'), (
        f'Expected Linux wheel filename {wheel_filename!r} to end with '
        f'-{expected_plat_tag}.whl'
    )

  with zipfile.ZipFile(wheel_path) as whl:
    names = whl.namelist()

    wheel_meta_files = [
        name for name in names if name.endswith(_WHEEL_METADATA_SUFFIX)
    ]
    assert (
        len(wheel_meta_files) == 1
    ), f'Expected 1 {_WHEEL_METADATA_SUFFIX} entry, found {wheel_meta_files}'
    wheel_meta = whl.read(wheel_meta_files[0]).decode('utf-8')
    wheel_meta_lines = wheel_meta.splitlines()
    assert (
        'Root-Is-Purelib: false' in wheel_meta_lines
    ), f'Expected Root-Is-Purelib: false in WHEEL metadata:\n{wheel_meta}'
    expected_tag_prefix = f'Tag: {py_tag}-{py_tag}-'
    if expected_plat_tag is not None:
      expected_tag_line = f'{expected_tag_prefix}{expected_plat_tag}'
      assert (
          expected_tag_line in wheel_meta_lines
      ), f'Expected {expected_tag_line!r} in WHEEL metadata:\n{wheel_meta}'
    else:
      assert any(
          line.startswith(expected_tag_prefix) for line in wheel_meta_lines
      ), (
          f'Expected line starting with {expected_tag_prefix!r} in '
          f'WHEEL:\n{wheel_meta}'
      )

    pkg_meta_files = [
        name for name in names if name.endswith(_PACKAGE_METADATA_SUFFIX)
    ]
    assert (
        len(pkg_meta_files) == 1
    ), f'Expected 1 {_PACKAGE_METADATA_SUFFIX} entry, found {pkg_meta_files}'
    pkg_meta = whl.read(pkg_meta_files[0]).decode('utf-8')
    req_python_lines = [
        line.split(':', 1)[1].strip()
        for line in pkg_meta.splitlines()
        if line.startswith('Requires-Python:')
    ]
    assert len(req_python_lines) == 1, (
        'Expected 1 Requires-Python header in METADATA, found '
        f'{req_python_lines}'
    )
    req_python_parts = {
        part.strip() for part in req_python_lines[0].split(',') if part.strip()
    }
    assert req_python_parts == _REQUIRED_PYTHON_SPECS, (
        f'Expected Requires-Python {_REQUIRED_PYTHON_SPECS!r}, got '
        f'{req_python_lines[0]!r}'
    )

  for prefix in _REQUIRED_SHARED_LIBRARY_PREFIXES:
    assert any(
        name.startswith(prefix) and name.endswith('.so') for name in names
    ), f'Missing shared library matching {prefix}*.so in wheel'

  for name in names:
    for forbidden in _FORBIDDEN_WHEEL_SUBSTRINGS:
      assert (
          forbidden not in name
      ), f'Unexpected forbidden entry in wheel: {name}'
    if name.startswith(_CC_PACKAGE_PREFIX):
      assert not name.endswith('.h'), f'Unexpected C++ header in wheel: {name}'
      assert not name.endswith(
          '__init__.py'
      ), f'Unexpected __init__.py under cc/ in wheel: {name}'

  print('Verified wheel tags, metadata, and contents.')


def verify_bindings() -> None:
  """Verifies installed pybind11 extensions and status_casters error types."""
  # Import installed package dependencies lazily so `wheel` stays stdlib-only.
  # pylint: disable=g-import-not-at-top
  import federated_language
  import portpicker
  from tensorflow_federated.cc.core.impl.executors import executor_bindings
  from tensorflow_federated.cc.core.impl.executors import tensorflow_bindings
  from tensorflow_federated.python.core.impl.executor_stacks import executor_stack_bindings
  from tensorflow_federated.python.core.impl.executors import executors_errors
  # pylint: enable=g-import-not-at-top

  assert 'pybind11_abseil' not in sys.modules
  assert importlib.util.find_spec('pybind11_abseil') is None
  try:
    stack_spec = importlib.util.find_spec(
        'tensorflow_federated.cc.core.impl.executor_stacks'
        '.executor_stack_bindings'
    )
  except ModuleNotFoundError:
    stack_spec = None
  assert stack_spec is None
  assert executor_bindings.__name__.endswith('executor_bindings')
  assert tensorflow_bindings.__name__.endswith('tensorflow_bindings')

  # Check that C++ absl::Status errors expose `.status` in pip wheels and work
  # with `executors_errors.is_absl_status_retryable_error`.
  executor = tensorflow_bindings.create_tensorflow_executor()
  try:
    executor.materialize(_INVALID_VALUE_REF)
  except Exception as exc:  # pylint: disable=broad-except
    assert 'pybind11_abseil' not in sys.modules
    # Check the exact builtins.Exception fallback, not a StatusNotOk subclass.
    assert (
        type(exc) is Exception  # pylint: disable=unidiomatic-typecheck
    ), f'Expected Exception fallback, got {type(exc)!r}'
    assert hasattr(exc, 'status'), f'Missing .status attribute on {exc!r}'
    assert hasattr(exc.status, 'code_int'), 'Missing .status.code_int()'
    assert hasattr(exc.status, 'message'), 'Missing .status.message()'
    assert hasattr(exc.status, 'to_string'), 'Missing .status.to_string()'
    assert exc.status.code_int() == _NOT_FOUND_STATUS_CODE, (
        f'Expected NOT_FOUND ({_NOT_FOUND_STATUS_CODE}), got '
        f'{exc.status.code_int()}'
    )
    assert (
        isinstance(exc.status.message(), str) and exc.status.message()
    ), f'Expected non-empty status message, got {exc.status.message()!r}'
    assert (
        isinstance(exc.status.to_string(), str) and exc.status.to_string()
    ), f'Expected non-empty status string, got {exc.status.to_string()!r}'
    assert not executors_errors.is_absl_status_retryable_error(exc)
  else:
    raise AssertionError('Expected NOT_FOUND exception from materialize()')

  # Check that retryable C++ absl::Status errors (UNAVAILABLE, code 14) are
  # classified as retryable.
  unused_port = portpicker.pick_unused_port()
  channel = executor_bindings.create_insecure_grpc_channel(
      f'localhost:{unused_port}'
  )
  try:
    executor_stack_bindings.create_remote_executor_stack(
        [channel], {federated_language.CLIENTS: _SINGLE_CLIENT}
    )
  except Exception as exc:  # pylint: disable=broad-except
    # Check the exact builtins.Exception fallback, not a StatusNotOk subclass.
    assert (
        type(exc) is Exception  # pylint: disable=unidiomatic-typecheck
    ), f'Expected Exception fallback, got {type(exc)!r}'
    assert hasattr(exc, 'status'), f'Missing .status attribute on {exc!r}'
    assert exc.status.code_int() == _UNAVAILABLE_STATUS_CODE, (
        f'Expected UNAVAILABLE ({_UNAVAILABLE_STATUS_CODE}), got '
        f'{exc.status.code_int()}'
    )
    assert executors_errors.is_absl_status_retryable_error(
        exc
    ), f'Expected UNAVAILABLE error to be retryable: {exc!r}'
  else:
    raise AssertionError(
        'Expected UNAVAILABLE exception from create_remote_executor_stack()'
    )

  print('Verified pybind bindings and status_casters in pip package.')


def main(argv: Sequence[str]) -> None:
  if len(argv) == 3 and argv[1] == 'wheel':
    verify_wheel(argv[2])
  elif len(argv) == 2 and argv[1] == 'bindings':
    verify_bindings()
  else:
    raise SystemExit(f'usage: {argv[0]} (wheel <wheel_path> | bindings)')


if __name__ == '__main__':
  main(sys.argv)
