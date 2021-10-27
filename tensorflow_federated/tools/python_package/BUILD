package(default_visibility = ["//visibility:private"])

package_group(
    name = "python_package_tool",
    packages = ["//tensorflow_federated/tools/python_package/..."],
)

licenses(["notice"])

# The data labels are needed otherwise Bazel will not know about the
# `__init__.py` files; this is because the modules are imported directory,
# rather than the packages, so in the Bazel dependency hierarchy, nothing is
# dependent on the `__init__.py` files.
sh_binary(
    name = "build_python_package",
    srcs = ["build_python_package.sh"],
    data = [
        ":common",
        ":setup",
        "//tensorflow_federated",
        "//tensorflow_federated/experimental/python",
        "//tensorflow_federated/experimental/python/core",
        "//tensorflow_federated/experimental/python/core/api",
        "//tensorflow_federated/experimental/python/core/impl",
        "//tensorflow_federated/experimental/python/core/impl/jax_context",
        "//tensorflow_federated/experimental/python/core/impl/wrappers",
        "//tensorflow_federated/proto",
        "//tensorflow_federated/proto/v0",
        "//tensorflow_federated/python",
        "//tensorflow_federated/python/analytics/hierarchical_histogram",
        "//tensorflow_federated/python/common_libs",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/core/api",
        "//tensorflow_federated/python/core/impl",
        "//tensorflow_federated/python/core/impl/compiler",
        "//tensorflow_federated/python/core/impl/computation",
        "//tensorflow_federated/python/core/impl/context_stack",
        "//tensorflow_federated/python/core/impl/execution_contexts",
        "//tensorflow_federated/python/core/impl/executors",
        "//tensorflow_federated/python/core/impl/federated_context",
        "//tensorflow_federated/python/core/impl/tensorflow_context",
        "//tensorflow_federated/python/core/impl/types",
        "//tensorflow_federated/python/core/impl/utils",
        "//tensorflow_federated/python/core/impl/wrappers",
        "//tensorflow_federated/python/core/impl/xla_context",
        "//tensorflow_federated/python/tensorflow_libs",
    ],
)

sh_binary(
    name = "common",
    srcs = ["common.sh"],
)

filegroup(
    name = "setup",
    srcs = ["setup.py"],
    tags = ["ignore_srcs"],
)

sh_binary(
    name = "publish_python_package",
    srcs = ["publish_python_package.sh"],
    data = [
        ":common",
    ],
)

sh_binary(
    name = "test_python_package",
    srcs = ["test_python_package.sh"],
    data = [
        ":common",
    ],
)
