"""TensorFlow Federated dependencies that can be loaded in WORKSPACE files."""

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def tf_federated_workspace():
    """TensorFlow Federated dependencies."""
    tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

    # gRPC wants the existence of a cares dependence but its contents are not
    # actually important since we have set GRPC_ARES=0 in .bazelrc
    native.bind(
        name = "cares",
        actual = "@grpc//third_party/nanopb:nanopb",
    )
