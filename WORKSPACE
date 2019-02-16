workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "org_tensorflow",
    sha256 = "c3c35cd1e69557fa4862dff9469b3f99946f8cf62bfe61be831b2c0222428215",
    strip_prefix = "tensorflow-c865ec5621c013a7f8a4a26d380782e63117224f",
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/c865ec5621c013a7f8a4a26d380782e63117224f.tar.gz",
        "https://github.com/tensorflow/tensorflow/archive/c865ec5621c013a7f8a4a26d380782e63117224f.tar.gz",
    ],
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.19.2")

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
)

# gRPC wants the existence of a cares dependence but its contents are not
# actually important since we have set GRPC_ARES=0 in .bazelrc
bind(
    name = "cares",
    actual = "@grpc//third_party/nanopb:nanopb",
)
