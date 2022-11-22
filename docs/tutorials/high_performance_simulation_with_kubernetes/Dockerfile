# Copyright 2022, The TensorFlow Federated Authors.
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
FROM python:3.9-buster

ARG WORKER_SRC_PATH="./worker_service.py"

ENV RUN_DIR="/root/worker"
# File path to the sqlite database storing a partition of the EMNIST dataset.
ENV EMNIST_PARTITION_PATH="$RUN_DIR/data/emnist_partition.sqlite"

WORKDIR $RUN_DIR

COPY $WORKER_SRC_PATH $RUN_DIR

RUN python3 --version

RUN pip3 install --no-cache-dir --upgrade "pip"
RUN pip3 install --no-cache-dir --upgrade "tensorflow-federated"

EXPOSE 8000

CMD python3 $RUN_DIR/worker_service.py
