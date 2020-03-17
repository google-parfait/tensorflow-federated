# Copyright 2019, The TensorFlow Federated Authors.
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
#
# This Dockerfile is used to create a Docker image of the TensorFlow Federated
# (TFF) remote executor service from a locally-built TFF .whl file. It requires
# a whl named like tensorflow_federated-*.whl to be present in the directory
# where docker build is run.
FROM python:3.6-buster

RUN python3 --version

COPY "tensorflow_federated/runtime/remote/remote_executor_service.py" /
COPY "tensorflow_federated-"*".whl" /

RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir --upgrade "/tensorflow_federated-"*".whl"
RUN pip3 freeze

EXPOSE 8000

CMD ["python3", "/remote_executor_service.py"]
