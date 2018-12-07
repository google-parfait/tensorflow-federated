workspace(name = "org_tensorflow_federated")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# v1.12.0
http_archive(
    name = "org_tensorflow",
    sha256 = "5aae44f967556dc524e6b184643bbd33a2d78184bc654b72c09f041f266b89a2",
    strip_prefix = "tensorflow-a6d8ffae097d0132989ae4688d224121ec6d8f35",
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/a6d8ffae097d0132989ae4688d224121ec6d8f35.tar.gz",
        "https://github.com/tensorflow/tensorflow/archive/a6d8ffae097d0132989ae4688d224121ec6d8f35.tar.gz",
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

# Please add all new TensorFlow Federated dependencies in workspace.bzl.
load("//tensorflow_federated:workspace.bzl", "tf_federated_workspace")
tf_federated_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("0.15.0")
