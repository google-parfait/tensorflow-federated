workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_google_protobuf",
    sha256 = "39c7a5e7e557b24fc324bec3a73054d277ed9b9b320b273564e04f862131e679",
    strip_prefix = "protobuf-3.10.1",
    urls = ["https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protobuf-python-3.10.1.tar.gz"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Required by com_google_protobuf
bind(
    name = "grpc_python_plugin",
    actual = "@com_github_grpc_grpc//src/compiler:grpc_python_plugin",
)

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "ffbe61269160ea745e487f79b0fd06b6edd3d50c6d9123f053b5634737cf2f69",
    strip_prefix = "grpc-1.25.0",
    urls = ["https://github.com/grpc/grpc/archive/v1.25.0.tar.gz"],
)

# Required by com_github_grpc_grpc
http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "18cd4df4e410b0439a4935f9ca035bd979993d42372ba79e7f2d4fafe9596ef0",
    urls = ["https://github.com/bazelbuild/rules_swift/releases/download/0.12.1/rules_swift.0.12.1.tar.gz"],
)

# Required by com_github_grpc_grpc
http_archive(
    name = "bazel_skylib",
    sha256 = "e72747100a8b6002992cc0bf678f6279e71a3fd4a88cab3371ace6c73432be30",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.0/bazel-skylib-1.0.0.tar.gz"],
)

# Required by com_github_grpc_grpc
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "122ebf7fe7d1c8e938af6aeaee0efe788a3a2449ece5a8d6a428cb18d6f88033",
    urls = ["https://github.com/bazelbuild/apple_support/releases/download/0.7.1/apple_support.0.7.1.tar.gz"],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@upb//bazel:workspace_deps.bzl", "upb_deps")

upb_deps()
