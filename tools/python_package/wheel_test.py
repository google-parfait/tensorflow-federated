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
"""Hermetic Bazel tests for the TensorFlow Federated Python wheel."""

from email import message
from email import parser
import os
import platform
import re
import subprocess
import sys
import tomllib
import zipfile

from absl import flags
from absl.testing import absltest
import tensorflow as tf

_WHEEL = flags.DEFINE_string(
    'wheel',
    default='',
    help='Path to the built .whl archive.',
)
_PYPROJECT = flags.DEFINE_string(
    'pyproject',
    default='',
    help='Path to pyproject.toml.',
)
_README = flags.DEFINE_string(
    'readme',
    default='',
    help='Path to README.md.',
)
_VERSION_FILE = flags.DEFINE_string(
    'version_file',
    default='',
    help='Path to version.py.',
)
_TEST_COMPUTATION = flags.DEFINE_string(
    'test_computation',
    default='',
    help='Path to test_computation.py.',
)
_VERIFY_SCRIPT = flags.DEFINE_string(
    'verify_script',
    default='',
    help='Path to verify_python_package.py.',
)

_MAX_WHEEL_BYTES = 20_000_000
_METADATA_SUFFIX = '.dist-info/METADATA'
_LICENSE_SUFFIX = '.dist-info/LICENSE'
_CONTENT_TYPE_MARKDOWN = 'text/markdown'

_PRIVACY_REQUIRED_MEMBER = (
    'tensorflow_federated/python/aggregators/privacy/query.py'
)
_PRIVACY_FORBIDDEN_MEMBERS = (
    'tensorflow_federated/python/aggregators/privacy/__init__.py',
    'tensorflow_federated/python/aggregators/privacy/test_utils.py',
)

_TF_BINDINGS_REL_DIR = 'tensorflow_federated/cc/core/impl/executors'
_STACK_BINDINGS_REL_DIR = 'tensorflow_federated/cc/core/impl/executor_stacks'
_REQUIRED_RPATH_PATTERN = (
    r'\((RUNPATH|RPATH)\).*\$ORIGIN/\.\./\.\./\.\./\.\./\.\./tensorflow'
)
_REQUIRED_NEEDED_LIBS = (
    'libtensorflow_framework.so.2',
    'libtensorflow_cc.so.2',
)
_REQUIRED_DEFINED_INIT_SYMBOLS = (
    ('executor_bindings', 'PyInit_executor_bindings'),
    ('tensorflow_bindings', 'PyInit_tensorflow_bindings'),
)
_FORBIDDEN_DEFINED_SYMBOL_PATTERN = r'tensorflow_federated::|google::protobuf::'
_MAX_LEAKED_SYMBOLS_REPORTED = 20
_REQUIRED_UNDEF_HASH_SEED_PATTERN = r'MixingHashState.*kSeed'
_FORBIDDEN_UNDEF_DESCRIPTOR_POOL = 'DescriptorPool::generated_pool'
_SYMTAB_SECTION_PATTERN = r'\s\.symtab\s'


def parse_version_py(version_path: str) -> str:
  """Reads __version__ from version.py."""
  with open(version_path, encoding='utf-8') as f:
    for line in f:
      match = re.search(r"^\s*__version__\s*=\s*['\"]([^'\"]+)['\"]", line)
      if match:
        return match.group(1)
  raise ValueError(f'Could not parse __version__ from {version_path!r}')


def read_wheel_metadata(
    wheel_path: str,
) -> tuple[message.Message, list[str]]:
  """Returns parsed .dist-info/METADATA and all archive member names."""
  with zipfile.ZipFile(wheel_path) as whl:
    names = whl.namelist()
    meta_entries = [name for name in names if name.endswith(_METADATA_SUFFIX)]
    if len(meta_entries) != 1:
      raise AssertionError(
          f'Expected 1 {_METADATA_SUFFIX} entry, found {meta_entries}'
      )
    raw_meta = whl.read(meta_entries[0]).decode('utf-8')
  return parser.Parser().parsestr(raw_meta), names


def parse_project_urls(meta: message.Message) -> dict[str, str]:
  """Parses RFC 822 Project-URL headers into a label-to-URL mapping."""
  urls = {}
  for header in meta.get_all('Project-URL') or ():
    label, url = header.split(',', 1)
    urls[label.strip()] = url.strip()
  return urls


def build_isolated_wheel_env(
    extracted_dir: str,
    tf_pkg_dir: str,
) -> dict[str, str]:
  """Links TensorFlow next to extracted wheel and builds isolated PYTHONPATH."""
  tf_link = os.path.join(extracted_dir, 'tensorflow')
  if not os.path.lexists(tf_link):
    os.symlink(tf_pkg_dir, tf_link)

  filtered_paths = []
  extracted_real = os.path.realpath(extracted_dir)
  for entry in sys.path:
    if not entry:
      continue
    entry_real = os.path.realpath(entry)
    if entry_real == extracted_real:
      continue
    if os.path.isdir(os.path.join(entry, 'tensorflow_federated')):
      continue
    filtered_paths.append(entry)

  env = os.environ.copy()
  env['PYTHONPATH'] = os.pathsep.join([extracted_dir, *filtered_paths])
  env['PYTHONDONTWRITEBYTECODE'] = '1'
  return env


def _find_single_so(directory: str, prefix: str) -> str:
  matches = [
      os.path.join(directory, name)
      for name in sorted(os.listdir(directory))
      if name.startswith(prefix) and name.endswith('.so')
  ]
  if len(matches) != 1:
    raise AssertionError(
        f'Expected 1 {prefix}*.so in {directory}, found {matches}'
    )
  return matches[0]


def verify_linux_elf_in_extracted_wheel(extracted_dir: str) -> None:
  """Verifies Linux ELF RPATH, NEEDED, visibility, and stripped .symtab."""
  exec_dir = os.path.join(extracted_dir, _TF_BINDINGS_REL_DIR)
  stack_dir = os.path.join(extracted_dir, _STACK_BINDINGS_REL_DIR)
  if os.path.isdir(stack_dir):
    stack_sos = [n for n in os.listdir(stack_dir) if n.endswith('.so')]
    if stack_sos:
      raise AssertionError(f'Unexpected executor_stack .so files: {stack_sos}')

  tf_so = _find_single_so(exec_dir, 'tensorflow_bindings')
  exec_so = _find_single_so(exec_dir, 'executor_bindings')

  dynamic_info = subprocess.check_output(['readelf', '-d', tf_so], text=True)
  if not re.search(_REQUIRED_RPATH_PATTERN, dynamic_info):
    raise AssertionError(
        f'{tf_so} missing expected TensorFlow RPATH/RUNPATH:\n{dynamic_info}'
    )
  for needed_lib in _REQUIRED_NEEDED_LIBS:
    if needed_lib not in dynamic_info:
      raise AssertionError(
          f'{tf_so} missing NEEDED {needed_lib}:\n{dynamic_info}'
      )

  tf_undef = subprocess.check_output(
      ['nm', '-D', '-C', '--undefined-only', tf_so], text=True
  )
  if not re.search(_REQUIRED_UNDEF_HASH_SEED_PATTERN, tf_undef):
    raise AssertionError(f'{tf_so} missing undefined MixingHashState::kSeed')
  if _FORBIDDEN_UNDEF_DESCRIPTOR_POOL in tf_undef:
    raise AssertionError(
        f'{tf_so} unexpectedly has undefined {_FORBIDDEN_UNDEF_DESCRIPTOR_POOL}'
    )

  so_by_prefix = {
      'executor_bindings': exec_so,
      'tensorflow_bindings': tf_so,
  }
  for prefix, init_symbol in _REQUIRED_DEFINED_INIT_SYMBOLS:
    so_path = so_by_prefix[prefix]
    defined = subprocess.check_output(
        ['nm', '-D', '-C', '--defined-only', so_path], text=True
    )
    if init_symbol not in defined:
      raise AssertionError(f'{so_path} missing exported {init_symbol}')
    leaked = [
        line
        for line in defined.splitlines()
        if re.search(_FORBIDDEN_DEFINED_SYMBOL_PATTERN, line)
    ]
    if leaked:
      sample = '\n'.join(leaked[:_MAX_LEAKED_SYMBOLS_REPORTED])
      raise AssertionError(f'{so_path} leaks hidden C++ symbols:\n{sample}')

    sections = subprocess.check_output(
        ['readelf', '-S', '-W', so_path], text=True
    )
    if re.search(_SYMTAB_SECTION_PATTERN, sections):
      raise AssertionError(
          f'{so_path} was not stripped (.symtab still present)'
      )


class WheelTest(absltest.TestCase):

  def _require_flag(self, flag_holder: flags.FlagHolder[str]) -> str:
    value = flag_holder.value
    self.assertNotEmpty(value, f'Required flag --{flag_holder.name} is empty.')
    self.assertTrue(os.path.isfile(value), f'File not found: {value}')
    return value

  def test_wheel_size_and_verify_script(self):
    wheel_path = self._require_flag(_WHEEL)
    verify_script = self._require_flag(_VERIFY_SCRIPT)

    wheel_size = os.path.getsize(wheel_path)
    self.assertLess(wheel_size, _MAX_WHEEL_BYTES)

    subprocess.run(
        [sys.executable, verify_script, 'wheel', wheel_path],
        check=True,
    )

  def test_wheel_metadata_matches_pyproject_version_and_readme(self):
    wheel_path = self._require_flag(_WHEEL)
    pyproject_path = self._require_flag(_PYPROJECT)
    readme_path = self._require_flag(_README)
    version_path = self._require_flag(_VERSION_FILE)

    with open(pyproject_path, 'rb') as f:
      project = tomllib.load(f)['project']
    with open(readme_path, encoding='utf-8') as f:
      expected_readme = f.read().strip()
    expected_version = parse_version_py(version_path)

    meta, names = read_wheel_metadata(wheel_path)

    self.assertEqual(meta['Name'], project['name'])
    self.assertEqual(meta['Version'], expected_version)
    self.assertEqual(meta['Summary'], project['description'])
    self.assertEqual(meta['Requires-Python'], project['requires-python'])
    self.assertEqual(meta['Author'], project['authors'][0]['name'])
    self.assertEqual(meta['Author-email'], project['authors'][0]['email'])
    self.assertEqual(
        meta['Description-Content-Type'],
        _CONTENT_TYPE_MARKDOWN,
    )

    actual_keywords = [
        kw.strip() for kw in (meta['Keywords'] or '').split(',') if kw.strip()
    ]
    self.assertEqual(actual_keywords, project['keywords'])
    self.assertCountEqual(
        meta.get_all('Requires-Dist') or [],
        project['dependencies'],
    )
    self.assertCountEqual(
        meta.get_all('Classifier') or [],
        project['classifiers'],
    )
    self.assertDictEqual(parse_project_urls(meta), project['urls'])

    payload = meta.get_payload()
    self.assertIsInstance(payload, str)
    self.assertEqual(payload.strip(), expected_readme)

    license_entries = [n for n in names if n.endswith(_LICENSE_SUFFIX)]
    self.assertLen(license_entries, 1)
    with zipfile.ZipFile(wheel_path) as whl:
      self.assertNotEmpty(whl.read(license_entries[0]).strip())

  def test_namespace_package_privacy_layout(self):
    wheel_path = self._require_flag(_WHEEL)
    with zipfile.ZipFile(wheel_path) as whl:
      names = set(whl.namelist())

    self.assertIn(_PRIVACY_REQUIRED_MEMBER, names)
    for forbidden in _PRIVACY_FORBIDDEN_MEMBERS:
      self.assertNotIn(forbidden, names)

  def test_linux_elf_rpath_symbols_and_stripped(self):
    if platform.system() != 'Linux':
      self.skipTest('Linux ELF checks apply only on Linux.')
    wheel_path = self._require_flag(_WHEEL)
    extracted_dir = self.create_tempdir().full_path
    with zipfile.ZipFile(wheel_path) as whl:
      whl.extractall(extracted_dir)
    verify_linux_elf_in_extracted_wheel(extracted_dir)

  def test_runtime_import_computation_and_bindings(self):
    wheel_path = self._require_flag(_WHEEL)
    test_computation = self._require_flag(_TEST_COMPUTATION)
    verify_script = self._require_flag(_VERIFY_SCRIPT)

    extracted_dir = self.create_tempdir().full_path
    with zipfile.ZipFile(wheel_path) as whl:
      whl.extractall(extracted_dir)

    tf_pkg_dir = os.path.dirname(os.path.realpath(tf.__file__))
    env = build_isolated_wheel_env(extracted_dir, tf_pkg_dir)

    origin_check = subprocess.check_output(
        [
            sys.executable,
            '-P',
            '-c',
            (
                'import os, tensorflow_federated as tff; '
                'print(os.path.realpath(tff.__file__))'
            ),
        ],
        env=env,
        text=True,
    ).strip()
    self.assertTrue(
        origin_check.startswith(os.path.realpath(extracted_dir) + os.sep),
        f'Expected import from {extracted_dir}, got {origin_check}',
    )

    subprocess.run(
        [sys.executable, '-P', test_computation], env=env, check=True
    )
    subprocess.run(
        [sys.executable, '-P', verify_script, 'bindings'],
        env=env,
        check=True,
    )


if __name__ == '__main__':
  absltest.main()
