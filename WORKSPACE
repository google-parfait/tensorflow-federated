workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.2.0",
)

git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/protocolbuffers/protobuf.git",
    tag = "v3.17.0",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Required by com_google_protobuf
bind(
    name = "grpc_python_plugin",
    actual = "@com_github_grpc_grpc//src/compiler:grpc_python_plugin",
)

git_repository(
    name = "com_github_grpc_grpc",
    remote = "https://github.com/grpc/grpc.git",
    tag = "v1.38.1",
)

# load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
# grpc_deps()

# load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
# grpc_extra_deps()

# Required by com_github_grpc_grpc
git_repository(
    name = "build_bazel_rules_swift",
    remote = "https://github.com/bazelbuild/rules_swift.git",
    tag = "0.12.1",
)

# Required by com_github_grpc_grpc
git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "1.0.0",
)

# Required by com_github_grpc_grpc
git_repository(
    name = "build_bazel_apple_support",
    remote = "https://github.com/bazelbuild/apple_support.git",
    tag = "0.7.1",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@upb//bazel:workspace_deps.bzl", "upb_deps")
upb_deps()

git_repository(
    name = "pybind11_bazel",
    remote = "https://github.com/pybind/pybind11_bazel.git",
    commit = "26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
)

new_git_repository(
    name = "pybind11",
    remote = "https://github.com/pybind/pybind11.git",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    tag = "v2.6.2",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
