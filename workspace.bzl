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
"""TEST."""

def _tensorflow_pip_impl(ctx):
    """TEST."""
    library_path = ctx.execute([
        "python3",
        "-c",
        "import tensorflow; print(tensorflow.sysconfig.get_lib())",
    ])
    if library_path.return_code != 0:
        fail("Failed to find library path. Did you remember to pip install " +
             "tensorflow?: %s" % library_path.stderr)

    include_path = ctx.execute([
        "python3",
        "-c",
        "import tensorflow; print(tensorflow.sysconfig.get_include())",
    ])

    if include_path.return_code != 0:
        fail("Failed to find include path. Did you remember to pip install " +
             "tensorflow?: %s" % include_path.stderr)

    if "mac" in ctx.os.name:
        library_filename = "libtensorflow_framework.dylib"
    else:
        library_filename = "libtensorflow_framework.so.2"

    ctx.symlink(
        "/".join([library_path.stdout.strip(), library_filename]),
        library_filename,
    )
    ctx.symlink(include_path.stdout.strip(), "include")
    ctx.file("BUILD", """
cc_library(
    name = "libtensorflow_framework",
    srcs = ["{}"],
    hdrs = glob(["include/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""".format(library_filename))

tensorflow_pip = repository_rule(
    implementation = _tensorflow_pip_impl,
)
