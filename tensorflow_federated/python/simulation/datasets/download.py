# Copyright 2021, The TensorFlow Federated Authors.
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
"""Module for utilites for downloading datasets and caching them locally."""

import lzma
import os
from typing import Optional
import urllib
import urllib.parse

import tensorflow as tf
import tqdm


def url_basename(origin: str) -> str:
  origin_path = urllib.parse.urlparse(origin).path
  return origin_path.rsplit('/', maxsplit=1)[-1]


def _fetch_lzma_file(origin: str, filename: str):
  """Fetches a LZMA compressed file and decompresses on the fly."""
  # Read and decompress in approximately megabyte chunks.
  chunk_size = 2**20
  decompressor = lzma.LZMADecompressor()
  with urllib.request.urlopen(origin) as in_stream, tf.io.gfile.GFile(
      filename, 'wb'
  ) as out_stream:
    length = in_stream.headers.get('content-length')
    if length is not None:
      total_size = int(length)
    else:
      total_size = None
    download_chunk = in_stream.read(chunk_size)
    with tqdm.tqdm(
        total=total_size, desc=f'Downloading {url_basename(origin)}'
    ) as progbar:
      while download_chunk:
        progbar.update(len(download_chunk))
        out_stream.write(decompressor.decompress(download_chunk))
        download_chunk = in_stream.read(chunk_size)


def get_compressed_file(origin: str, cache_dir: Optional[str] = None) -> str:
  """Downloads and caches an LZMA compressed file from a URL.

  Args:
    origin: The URL source of the file to fetch.
    cache_dir: An optional alternative path for caching the downloaded and
      extracted file. If `None`, defaults to a `.tff` sub folder in the user's
      home directory.

  Returns:
    A `str` path to the uncompressed file in the cache directory.
  """
  if cache_dir is None:
    cache_dir = os.path.join(os.path.expanduser('~'), '.tff')
  filename = url_basename(origin)
  local_filename = os.path.join(cache_dir, filename)
  extracted_filename, ext = os.path.splitext(local_filename)
  if ext != '.lzma':
    raise ValueError(
        'Only decompressing LZMA files is supported. If the file '
        'is LZMA compressed, rename the origin to have a .lzma suffix.'
    )
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.makedirs(cache_dir)
  if tf.io.gfile.exists(extracted_filename):
    return extracted_filename
  _fetch_lzma_file(origin, extracted_filename)
  return extracted_filename
