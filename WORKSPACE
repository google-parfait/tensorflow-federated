workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "1.0.3",
)

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.2.0",
)

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

git_repository(
    name = "absl_py",
    remote = "https://github.com/abseil/abseil-py.git",
    tag = "pypi-v0.9.0",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "six_archive",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    build_file = "@absl_py//third_party:six.BUILD",
)

http_archive(
    name = "mock_archive",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/a2/52/7edcd94f0afb721a2d559a5b9aae8af4f8f2c79bc63fdbe8a8a6c9b23bbe/mock-1.0.1.tar.gz",
        "https://pypi.python.org/packages/a2/52/7edcd94f0afb721a2d559a5b9aae8af4f8f2c79bc63fdbe8a8a6c9b23bbe/mock-1.0.1.tar.gz",
    ],
    sha256 = "b839dd2d9c117c701430c149956918a423a9863b48b09c90e30a6013e7d2f44f",
    strip_prefix = "mock-1.0.1",
    build_file = "@absl_py//third_party:mock.BUILD",
)

http_archive(
    name = "enum34_archive",
    urls = [
        "https://mirror.bazel.build/pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz",
        "https://pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz"
    ],
    sha256 = "8ad8c4783bf61ded74527bffb48ed9b54166685e4230386a9ed9b1279e2df5b1",
    build_file = "@absl_py//third_party:enum34.BUILD"
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

new_git_repository(
    name = "pybind11",
    remote = "https://github.com/pybind/pybind11.git",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    tag = "v2.6.2",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
