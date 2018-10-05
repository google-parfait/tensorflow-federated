"""TensorFlow Federated dependencies that can be loaded in WORKSPACE files."""

# load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def tf_federated_workspace():
    """TensorFlow Federated dependencies."""

    # Tensorflow initializes some dependencies that Tensorflow Federated is
    # directly dependent on, there is no reason to re-import them:
    #   six_archive
    tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

    # gRPC wants the existence of a cares dependence but its contents are not
    # actually important since we have set GRPC_ARES=0 in .bazelrc
    native.bind(
        name = "cares",
        actual = "@grpc//third_party/nanopb:nanopb",
    )
