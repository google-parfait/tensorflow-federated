workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load(":version_check.bzl", "check_bazel_version_equal")

check_bazel_version_equal(bazel_version = "0.26.1")

http_archive(
    name = "com_google_protobuf",
    sha256 = "69d4d1fa02eab7c6838c8f11571cfd5509afa661b3944b3f7d24fef79a18d49d",
    strip_prefix = "protobuf-6a59a2ad1f61d9696092f79b6d74368b4d7970a3",
    urls = [
        "http://mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/6a59a2ad1f61d9696092f79b6d74368b4d7970a3.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/6a59a2ad1f61d9696092f79b6d74368b4d7970a3.tar.gz",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Required by com_google_protobuf
http_archive(
    name = "bazel_skylib",
    sha256 = "6b6ef4f707252c55b6109f02f4322f5219c7467b56bff8587876681ad067e57b",
    strip_prefix = "bazel-skylib-3721d32c14d3639ff94320c780a60a6e658fb033",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/archive/3721d32c14d3639ff94320c780a60a6e658fb033.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/archive/3721d32c14d3639ff94320c780a60a6e658fb033.tar.gz",
    ],
)

# Required by com_google_protobuf
http_archive(
    name = "grpc",
    sha256 = "0200a58dc9fb7372d0c181e502080d4985857f6c7bf8d37c45bd2d1374767449",
    strip_prefix = "grpc-51d641669195b0e044a3cda1a17e5740197b8658",
    urls = [
        "http://mirror.tensorflow.org/github.com/grpc/grpc/archive/51d641669195b0e044a3cda1a17e5740197b8658.tar.gz",
        "https://github.com/grpc/grpc/archive/51d641669195b0e044a3cda1a17e5740197b8658.tar.gz",
    ],
)

http_archive(
    name = "benjaminp_six",
    build_file = "//third_party:six.BUILD",
    sha256 = "0c8a18a365fbe4fca9f6bdfc1e64f34d527c8690d717e0b0488456b7f871d05e",
    urls = [
        "http://mirror.tensorflow.org/github.com/benjaminp/six/archive/d927b9e27617abca8dbf4d66cc9265ebbde261d6.tar.gz",
        "https://github.com/benjaminp/six/archive/d927b9e27617abca8dbf4d66cc9265ebbde261d6.tar.gz",
    ],
)

# Required by com_google_protobuf
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
