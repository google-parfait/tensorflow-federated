workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "com_google_protobuf",
    commit = "5902e759108d14ee8e6b0b07653dac2f4e70ac73",
    remote = "https://github.com/protocolbuffers/protobuf.git",
    shallow_since = "2019-04-01",
)

#Required by com_google_protobuf
git_repository(
    name = "bazel_skylib",
    commit = "3721d32c14d3639ff94320c780a60a6e658fb033",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
)

git_repository(
    name = "tensorflow_privacy",
    commit = "17fefb38958c749f0294bf457dadf8891f2ab49a",
    remote = "https://github.com/tensorflow/privacy.git",
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

http_archive(
    name = "zlib_archive",
    build_file = "//third_party:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "http://mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
)

#Required by com_google_protobuf
bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
)
