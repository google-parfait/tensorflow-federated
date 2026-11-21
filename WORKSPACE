# Dependencies for the TensorFlow Federated bazel environment.
#
# If you add a new or update a repository, follow these guidelines:
#
# *   Repositories must be deterministic (i.e., not a branch).
# *   Prefer to use an `http_archive` rule with a `sha256` parameter.
# *   Prefer for the repository to be a released product (i.e., a tag).
# *   Configuration must be documented when a commit is used instead of a
#     released product.

workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#
# Direct dependencies
#

# A hermetic build system is designed to produce completely reproducible builds for C++.
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "5e56be4646bdff06e0129a1824ee1a326d8bf231d11a2709f1237ba46bb2fe97",
    strip_prefix = "rules_ml_toolchain-0.4.0-rc2",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/releases/download/0.4.0-rc2/rules_ml_toolchain-0.4.0-rc2.tar.gz",
    ],
)

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64")

http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
)

http_archive(
    name = "rules_cc",
    sha256 = "b8b918a85f9144c01f6cfe0f45e4f2838c7413961a8ff23bc0c6cdf8bb07a3b6",
    strip_prefix = "rules_cc-0.1.5",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.1.5/rules_cc-0.1.5.tar.gz",
)

# Protobuf (version aligned with TF 2.21: v6.31.1)
# Must be pinned BEFORE any macros (cc_toolchain_deps, py_repositories,
# pip_parse, grpc_deps) that might register their own protobuf version.
http_archive(
    name = "com_google_protobuf",
    sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
    strip_prefix = "protobuf-6.31.1",
    url = "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip",
    repo_mapping = {
        "@abseil-cpp": "@com_google_absl",
    },
)

http_archive(
    name = "com_github_grpc_grpc",
    patch_args = ["-p1"],
    patches = [
        "//third_party/grpc:grpc.patch",
    ],
    sha256 = "dd6a2fa311ba8441bbefd2764c55b99136ff10f7ea42954be96006a2723d33fc",
    strip_prefix = "grpc-1.74.0",
    url = "https://github.com/grpc/grpc/archive/refs/tags/v1.74.0.tar.gz",
)

http_archive(
    name = "com_google_benchmark",
    sha256 = "6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f36ac7b06b0d9e9ce",
    strip_prefix = "benchmark-1.8.3",
    url = "https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz",
)

http_archive(
    name = "com_google_cc_differential_privacy",
    sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
    strip_prefix = "differential-privacy-3.0.0/cc",
    url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
)

http_archive(
    name = "com_google_differential_privacy",
    sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
    strip_prefix = "differential-privacy-3.0.0",
    url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
)


http_archive(
    name = "eigen",
    build_file = "//third_party:eigen.BUILD",
    sha256 = "1a432ccbd597ea7b9faa1557b1752328d6adc1a3db8969f6fe793ff704be3bf0",
    strip_prefix = "eigen-4c38131a16803130b66266a912029504f2cf23cd",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/4c38131a16803130b66266a912029504f2cf23cd/eigen-4c38131a16803130b66266a912029504f2cf23cd.tar.gz",
        "https://gitlab.com/libeigen/eigen/-/archive/4c38131a16803130b66266a912029504f2cf23cd/eigen-4c38131a16803130b66266a912029504f2cf23cd.tar.gz",
    ],
)

http_archive(
    name = "federated_language",
    patches = [
        "//third_party/federated_language:numpy.patch",
        "//third_party/federated_language:proto_library_loads.patch",
        "//third_party/federated_language:python_deps.patch",
        # Must come after `python_deps.patch`, this patches the output of `python_deps.patch`.
        "//third_party/federated_language:structure_visibility.patch",
        "//third_party/federated_language:remove_compile_pip_requirements.patch",
    ],
    repo_mapping = {
        "@protobuf": "@com_google_protobuf",
        "@federated_language_pypi": "@pypi",
    },
    sha256 = "d99f3a4cc88f816391777fc53cdc094d9abd72bdd7a6fe2bdccfa43235f351db",
    strip_prefix = "federated-language-0.5.4",
    url = "https://github.com/google-parfait/federated-language/archive/refs/tags/v0.5.4.tar.gz",
)

http_archive(
    name = "pybind11_bazel",
    sha256 = "cae680670bfa6e82703c03f2a3c995408cdcbf43616d7bdd198ef45d3c327731",
    strip_prefix = "pybind11_bazel-2.13.6",
    url = "https://github.com/pybind/pybind11_bazel/archive/refs/tags/v2.13.6.tar.gz",
)

http_archive(
    name = "rules_license",
    sha256 = "8c1155797cb5f5697ea8c6eac6c154cf51aa020e368813d9d9b949558c84f2da",
    strip_prefix = "rules_license-0.0.8",
    url = "https://github.com/bazelbuild/rules_license/archive/refs/tags/0.0.8.tar.gz",
)

#
# Inlined transitive dependencies, grouped by direct dependency.
#

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    urls = [
        "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
    ],
)

# Required by pybind11_abseil and pybind11_protobuf.
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
    sha256 = "efc901aa0aab439a3fea6efeaf930b5a349fb06394bf845c64ce15a9cf8f0240",
    strip_prefix = "pybind11-2.13.4",
    url = "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.4.tar.gz",
)

# Abseil C++ (version aligned with TF 2.21: LTS 20250814.1)
http_archive(
    name = "com_google_absl",
    sha256 = "d1abe9da2003e6cbbd7619b0ced3e52047422f4f4ac6c66a9bef5d2e99fea837",
    strip_prefix = "abseil-cpp-d38452e1ee03523a208362186fd42248ff2609f6",
    url = "https://github.com/abseil/abseil-cpp/archive/d38452e1ee03523a208362186fd42248ff2609f6.tar.gz",
)

# pybind11_abseil (version aligned with TF 2.21.0)
http_archive(
    name = "pybind11_abseil",
    sha256 = "0223b647b8cc817336a51e787980ebc299c8d5e64c069829bf34b69d72337449",
    strip_prefix = "pybind11_abseil-2c4932ed6f6204f1656e245838f4f5eae69d2e29",
    url = "https://github.com/pybind/pybind11_abseil/archive/2c4932ed6f6204f1656e245838f4f5eae69d2e29.tar.gz",
)

# pybind11_protobuf (version aligned with TF 2.21.0)
http_archive(
    name = "pybind11_protobuf",
    sha256 = "3cf7bf0f23954c5ce6c37f0a215f506efa3035ca06e3b390d67f4cbe684dce23",
    strip_prefix = "pybind11_protobuf-f02a2b7653bc50eb5119d125842a3870db95d251",
    url = "https://github.com/pybind/pybind11_protobuf/archive/f02a2b7653bc50eb5119d125842a3870db95d251.zip",
)

#
# rules_python and pip dependencies
#

http_archive(
    name = "rules_python",
    sha256 = "9c6e26911a79fbf510a8f06d8eedb40f412023cf7fa6d1461def27116bff022c",
    strip_prefix = "rules_python-1.1.0",
    url = "https://github.com/bazel-contrib/rules_python/releases/download/1.1.0/rules_python-1.1.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_register_toolchains(
    name = "python_3_12",
    ignore_root_user_error = True,
    python_version = "3.12",
)

load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

# Create Bazel cc_library targets for NumPy headers.
# https://github.com/bazel-contrib/rules_python/issues/259
numpy_annotation = package_annotation(
    additive_build_content = """\
cc_library(
    name = "numpy_headers_2",
    hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]),
    strip_include_prefix="site-packages/numpy/_core/include/",
)
cc_library(
    name = "numpy_headers_1",
    hdrs = glob(["site-packages/numpy/core/include/**/*.h"]),
    strip_include_prefix="site-packages/numpy/core/include/",
)
cc_library(
    name = "numpy_headers",
    deps = [":numpy_headers_2", ":numpy_headers_1"],
    # For the layering check to work we need to re-export the headers from the
    # dependencies.
    hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]) +
           glob(["site-packages/numpy/core/include/**/*.h"]),
    visibility = ["//visibility:public"],
)
""",
)

# Create Bazel cc_library targets for TensorFlow C++ headers and framework
# shared library from the pip package. This replaces the need for
# building TensorFlow from source for C++ compilation.
tensorflow_annotation = package_annotation(
    additive_build_content = """\
cc_library(
    name = "tf_header_lib",
    hdrs = glob(["site-packages/tensorflow/include/**/*"]),
    strip_include_prefix = "site-packages/tensorflow/include/",
    visibility = ["//visibility:public"],
)

# Force `libtensorflow_framework` before `absl/hash` so TFF and TensorFlow
# share one `MixingHashState::kSeed` while protobuf stays private.
cc_library(
    name = "libtensorflow_framework",
    srcs = select({
        "@platforms//os:linux": ["site-packages/tensorflow/libtensorflow_framework.so.2"],
        "@platforms//os:macos": ["site-packages/tensorflow/libtensorflow_framework.2.dylib"],
    }),
    deps = ["@com_google_absl//absl/hash"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtensorflow_cc",
    srcs = select({
        "@platforms//os:linux": ["site-packages/tensorflow/libtensorflow_cc.so.2"],
        "@platforms//os:macos": ["site-packages/tensorflow/libtensorflow_cc.2.dylib"],
    }),
    deps = [":libtensorflow_framework"],
    visibility = ["//visibility:public"],
)
""",
)

pip_parse(
    name = "pypi",
    annotations = {
        "numpy": numpy_annotation,
        "tensorflow": tensorflow_annotation,
    },
    extra_pip_args = [
        "--index-url",
        "https://pypi.org/simple",
    ],
    python_interpreter_target = "@python_3_12_host//:python",
    requirements_lock = "//:requirements_lock_3_12.txt",
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

#
# CUDA and GPU dependencies
#

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")



# Register rules_proto 6.0.2 explicitly before grpc_deps() so protobuf and gRPC
# rules resolve against a consistent rules_proto archive.
http_archive(
    name = "rules_proto",
    sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
    strip_prefix = "rules_proto-6.0.2",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/6.0.2.tar.gz",
    ],
)


load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

# Define @system_python before grpc_extra_deps() / protobuf_deps() so
# @com_google_protobuf//python:proto_api (used by pybind11_protobuf) compiles
# against hermetic Python headers instead of host /usr/include/python*.
load("//third_party:system_python.bzl", "hermetic_system_python")

hermetic_system_python(name = "system_python")

# grpc_deps() registers bazel_features 1.14.0 but doesn't call its deps
# function. rules_proto 6.0.2 needs bazel_features to be fully initialized.
load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()
