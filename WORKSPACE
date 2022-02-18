workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

#
# Direct dependencies
#

git_repository(
    name = "absl_py",
    remote = "https://github.com/abseil/abseil-py.git",
    tag = "v1.0.0",
)

git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "1.0.3",
)

# TODO(b/218935430): Temporarily disable the direct dependency on Abseil C++
# Common Libraries, for now we pick this dependency up via TensorFlows
# workspace.
# git_repository(
#     name = "com_google_absl",
#     remote = "https://github.com/abseil/abseil-cpp.git",
#     tag = "20211102.0",
# )

git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.11.0",
)

git_repository(
    name = "com_google_protobuf",
    # Patched to give visibility into private targets to pybind11_protobuf
    patches = ["//third_party/protobuf:com_google_protobuf_build.patch"],
    remote = "https://github.com/protocolbuffers/protobuf.git",
    tag = "v3.18.0-rc1",
)

git_repository(
    name = "io_bazel_rules_go",
    remote = "https://github.com/bazelbuild/rules_go.git",
    tag = "v0.29.0",
)

git_repository(
    name = "org_tensorflow",
    patches = ["//third_party/tensorflow:internal_visibility.patch"],
    remote = "https://github.com/tensorflow/tensorflow.git",
    tag = "v2.8.0",
)

git_repository(
    name = "pybind11_abseil",
    commit = "d9614e4ea46b411d02674305245cba75cd91c1c6",
    remote = "https://github.com/pybind/pybind11_abseil.git",
)

git_repository(
    name = "pybind11_bazel",
    commit = "72cbbf1fbc830e487e3012862b7b720001b70672",
    remote = "https://github.com/pybind/pybind11_bazel.git",
)

git_repository(
    name = "pybind11_protobuf",
    commit = "f20537c3285636716ed86a7ef96baa19a788c944",
    remote = "https://github.com/pybind/pybind11_protobuf.git",
)

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.5.0",
)

#
# Inlined transitive dependencies, grouped by direct dependency.
#

# Required by pybind11_bazel
new_git_repository(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    remote = "https://github.com/pybind/pybind11.git",
    tag = "v2.7.1",
)

# Required by pybind11_bazel
# load("@pybind11_bazel//:python_configure.bzl", "python_configure")
# python_configure(name = "local_config_python")

# Required by absl_py
http_archive(
    name = "six",
    build_file = "//third_party:six.BUILD",
    sha256 = "30639c035cdb23534cd4aa2dd52c3bf48f06e5f4a941509c8bafd8ce11080259",
    strip_prefix = "six-1.15.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.15.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.15.0.tar.gz",
    ],
)

#
# Transitive dependencies, grouped by direct dependency.
#

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.17.1")

git_repository(
    name = "bazel_gazelle",
    remote = "https://github.com/bazelbuild/bazel-gazelle.git",
    tag = "v0.24.0",
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

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

git_repository(
    name = "com_github_grpc_grpc",
    remote = "https://github.com/grpc/grpc.git",
    tag = "v1.38.1",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()
