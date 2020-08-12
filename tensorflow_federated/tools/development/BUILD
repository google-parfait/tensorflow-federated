package(default_visibility = ["//tensorflow_federated/tools:__subpackages__"])

licenses(["notice"])

# The data labels are needed otherwise Bazel will not know about the
# `__init__.py` files; this is because the modules are imported directory,
# rather than the packages, so in the Bazel dependency hierarchy, nothing is
# dependent on the `__init__.py` files.
sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        ":setup",
        "//tensorflow_federated",
        "//tensorflow_federated/proto",
        "//tensorflow_federated/proto/v0",
        "//tensorflow_federated/python",
        "//tensorflow_federated/python/common_libs",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/core/api",
        "//tensorflow_federated/python/core/impl",
        "//tensorflow_federated/python/core/impl/compiler",
        "//tensorflow_federated/python/core/impl/context_stack",
        "//tensorflow_federated/python/core/impl/executors",
        "//tensorflow_federated/python/core/impl/federated_context",
        "//tensorflow_federated/python/core/impl/tensorflow_context",
        "//tensorflow_federated/python/core/impl/types",
        "//tensorflow_federated/python/core/impl/utils",
        "//tensorflow_federated/python/core/impl/wrappers",
        "//tensorflow_federated/python/tensorflow_libs",
    ],
)

filegroup(
    name = "setup",
    srcs = ["setup.py"],
    tags = ["ignore_srcs"],
)

sh_binary(
    name = "publish_pip_package",
    srcs = ["publish_pip_package.sh"],
)

sh_binary(
    name = "test_pip_package",
    srcs = ["test_pip_package.sh"],
)
