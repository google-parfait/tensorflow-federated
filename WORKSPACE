workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Commit hash found from git tag `v1.13.0-rc0`:
#  https://github.com/tensorflow/tensorflow/tree/v1.13.0-rc0
http_archive(
    name = "org_tensorflow",
    sha256 = "1677e8e5bf9d5a9041aa5fc12ea6ff82eecc8603f6fd1292f0759b90e7096c21",
    strip_prefix = "tensorflow-a8e5c41c5bbe684a88b9285e07bd9838c089e83b",
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/a8e5c41c5bbe684a88b9285e07bd9838c089e83b.tar.gz",
        "https://github.com/tensorflow/tensorflow/archive/a8e5c41c5bbe684a88b9285e07bd9838c089e83b.tar.gz",
    ],
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
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
tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

# gRPC wants the existence of a cares dependence but its contents are not
# actually important since we have set GRPC_ARES=0 in .bazelrc
native.bind(
    name = "cares",
    actual = "@grpc//third_party/nanopb:nanopb",
)
