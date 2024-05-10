workspace(name = "org_tensorflow_federated")

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "git_repository",
    "new_git_repository",
)
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#
# Direct dependencies
#

git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "1.3.0",
)

git_repository(
    name = "com_github_grpc_grpc",
    remote = "https://github.com/grpc/grpc.git",
    tag = "v1.50.0",
)

git_repository(
    name = "com_google_benchmark",
    remote = "https://github.com/google/benchmark.git",
    tag = "v1.8.3",
)

http_archive(
  name = "com_google_cc_differential_privacy",
  url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
  sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
  strip_prefix = "differential-privacy-3.0.0/cc",
)

http_archive(
  name = "com_google_differential_privacy",
  url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
  sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
  strip_prefix = "differential-privacy-3.0.0",
)

git_repository(
    name = "com_google_absl",
    commit = "fb3621f4f897824c0dbe0615fa94543df6192f30",
    remote = "https://github.com/abseil/abseil-cpp.git",
)

git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.12.1",
)

git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/protocolbuffers/protobuf.git",
    tag = "v3.21.9",
)

# TODO: b/333391041 - Temporarily disable the direct dependency on
# `eigen`, for now we pick this dependency up from the TensorFlow workspace.
# new_git_repository(
#     name = "eigen",
#     tag = "3.4.0",
#     remote = "https://gitlab.com/libeigen/eigen.git",
#     build_file = "//third_party/eigen:eigen.BUILD",
#     repo_mapping = {
#         "@eigen": "@eigen_archive",
#     },
# )

git_repository(
    name = "org_tensorflow",
    # The version of this dependency should match the version in
    # https://github.com/tensorflow/federated/blob/main/requirements.txt.
    patches = [
        # Depending on restricted visibility BUILD target om external git
        # repository does not seem to be supported.
        # E.g. issue: https://github.com/bazelbuild/bazel/issues/3744
        # TODO: b/263201501 - Make DTensor C++ API public and remove this patch.
        "//third_party/tensorflow:dtensor_internal_visibility.patch",
        "//third_party/tensorflow:internal_visibility.patch",
        "//third_party/tensorflow:python_toolchain.patch",
        "//third_party/tensorflow:tf2xla_visibility.patch",
    ],
    remote = "https://github.com/tensorflow/tensorflow.git",
    tag = "v2.14.0",
)

git_repository(
    name = "pybind11_abseil",
    commit = "38111ef06d426f75bb335a3b58aa0342f6ce0ce3",
    remote = "https://github.com/pybind/pybind11_abseil.git",
)

git_repository(
    name = "pybind11_bazel",
    remote = "https://github.com/pybind/pybind11_bazel.git",
    tag = "v2.11.1",
)

git_repository(
    name = "pybind11_protobuf",
    commit = "80f3440cd8fee124e077e2e47a8a17b78b451363",
    remote = "https://github.com/pybind/pybind11_protobuf.git",
)

git_repository(
    name = "rules_license",
    remote = "https://github.com/bazelbuild/rules_license.git",
    tag = "0.0.8",
)

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.23.0",
)

#
# Inlined transitive dependencies, grouped by direct dependency.
#

# Required by pybind11_abseil and pybind11_protobuf.
new_git_repository(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    remote = "https://github.com/pybind/pybind11.git",
    tag = "v2.9.2",
)

# Required by com_github_grpc_grpc. This commit is determined by
# https://github.com/grpc/grpc/blob/v1.50.0/bazel/grpc_deps.bzl#L344.
git_repository(
    name = "upb",
    remote = "https://github.com/protocolbuffers/upb.git",
    commit = "e4635f223e7d36dfbea3b722a4ca4807a7e882e2",
)

#
# Transitive dependencies, grouped by direct dependency.
#

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Unable to load the `grpc_extra_deps` from gRPC because they conflict with the
# dependencies required by TensorFlow.
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()
