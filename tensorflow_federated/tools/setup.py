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
"""TensorFlow Federated is an open source federated learning framework.

TensorFlow Federated is a framework for implementing federated learning
algorithms and other computations on privacy-sensitive decentralized data.

TODO(b/112658581): Add TFF high level overview.
"""

import sys
import setuptools

DOCLINES = __doc__.split('\n')

# Set when releasing a new version of TensorFlow Federated (e.g. 1.0.0).
_VERSION = '0.0.0'

project_name = 'tensorflow_federated'

# Set when building the pip package
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)

REQUIRED_PACKAGES = [
    'h5py',
    'mock',
    'numpy',
    'six',
    'tensorflow>=1.13.0rc0',
]

setuptools.setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    packages=setuptools.find_packages(exclude=('tools')),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    url='http://tensorflow.org/federated',
    download_url='https://github.com/tensorflow/federated/tags',
    install_requires=REQUIRED_PACKAGES,
    # PyPI package information.
    classifiers=(
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license='Apache 2.0',
    keywords='tensorflow federated machine learning',
)
