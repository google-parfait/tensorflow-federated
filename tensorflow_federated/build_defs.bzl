"""TensorFlow Federated build definitions."""

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

def federated_py_proto_library(name, srcs = [], deps = [], **kwargs):
    """Generates proto targets for Python.

    Args:
      name: Name for proto_library.
      srcs: Same as py_proto_library deps.
      deps: Unused, provided for compatibility only.
      **kwargs: proto_library arguments.
    """
    _ = (deps)  # Unused.
    py_proto_library(
        name = name,
        srcs = srcs,
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        **kwargs
    )
