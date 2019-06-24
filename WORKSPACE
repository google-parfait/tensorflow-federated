workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "com_google_protobuf",
    commit = "54288a01cebfd0bfa62ca581dd07ffd6f9c77f2c",
    remote = "https://github.com/protocolbuffers/protobuf.git",
    shallow_since = "2019-06-01",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

#Required by com_google_protobuf
git_repository(
    name = "bazel_skylib",
    commit = "3721d32c14d3639ff94320c780a60a6e658fb033",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
)

#Required by com_google_protobuf
git_repository(
    name = "grpc",
    commit = "51d641669195b0e044a3cda1a17e5740197b8658",
    remote = "https://github.com/grpc/grpc.git",
    shallow_since = "2019-06-01",
)

new_git_repository(
    name = "benjaminp_six",
    build_file = "//third_party:six.BUILD",
    commit = "d927b9e27617abca8dbf4d66cc9265ebbde261d6",
    remote = "https://github.com/benjaminp/six.git",
)

#Required by com_google_protobuf
bind(
    name = "six",
    actual = "@benjaminp_six//:six",
)

bind(
    name = "grpc_python_plugin",
    actual = "@grpc//:grpc_python_plugin",
)

# Needed by gRPC
bind(
    name = "protobuf_clib",
    actual = "@com_google_protobuf//:protoc_lib",
)

# Needed by gRPC
bind(
    name = "protobuf_headers",
    actual = "@com_google_protobuf//:protobuf_headers",
)
