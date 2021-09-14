workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

#
# Direct dependencies
#

git_repository(
    name = "absl_py",
    remote = "https://github.com/abseil/abseil-py.git",
    tag = "pypi-v0.9.0",
)

# git_repository(
#     name = "bazel_skylib",
#     remote = "https://github.com/bazelbuild/bazel-skylib.git",
#     tag = "1.0.3",
# )

git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp.git",
    tag = "20210324.2",
)

git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.11.0",
)

git_repository(
    name = "pybind11_abseil",
    remote = "https://github.com/pybind/pybind11_abseil.git",
    commit = "d9614e4ea46b411d02674305245cba75cd91c1c6",
)

git_repository(
    name = "pybind11_bazel",
    remote = "https://github.com/pybind/pybind11_bazel.git",
    commit = "26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
)

git_repository(
    name = "pybind11_protobuf",
    remote = "https://github.com/pybind/pybind11_protobuf.git",
    commit = "b2e7ae2294b8520f01842e4da8e6d030d58c4a27",
)

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.2.0",
)

#
# Inlined transitive dependencies, grouped by direct dependency.
#

# Required by pybind11_bazel
new_git_repository(
    name = "pybind11",
    remote = "https://github.com/pybind/pybind11.git",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    tag = "v2.7.1",
)

# load("@pybind11_bazel//:python_configure.bzl", "python_configure")
# python_configure(name = "local_config_python")

#
# Other dependencies
#

git_repository(
    name = "org_tensorflow",
    remote = "https://github.com/tensorflow/tensorflow.git",
    tag = "v2.6.0",
    patches = ["//third_party/tensorflow:internal_visibility.patch"],
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/protocolbuffers/protobuf.git",
    tag = "v3.17.3",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

git_repository(
    name = "com_github_grpc_grpc",
    remote = "https://github.com/grpc/grpc.git",
    tag = "v1.38.1",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

#
# Transitive dependencies, grouped by direct dependency.
#
