package(default_visibility = ["//visibility:private"])

package_group(
    name = "python_package_tool",
    packages = ["//third_party/tensorflow_federated/tools/python_package/..."],
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
        "//third_party/tensorflow_federated",
        "//third_party/tensorflow_federated/experimental/python",
        "//third_party/tensorflow_federated/experimental/python/core",
        "//third_party/tensorflow_federated/experimental/python/core/api",
        "//third_party/tensorflow_federated/experimental/python/core/impl",
        "//third_party/tensorflow_federated/experimental/python/core/impl/jax_context",
        "//third_party/tensorflow_federated/experimental/python/core/impl/wrappers",
        "//third_party/tensorflow_federated/proto",
        "//third_party/tensorflow_federated/proto/v0",
        "//third_party/tensorflow_federated/python",
        "//third_party/tensorflow_federated/python/analytics/hierarchical_histogram",
        "//third_party/tensorflow_federated/python/common_libs",
        "//third_party/tensorflow_federated/python/core",
        "//third_party/tensorflow_federated/python/core/api",
        "//third_party/tensorflow_federated/python/core/impl",
        "//third_party/tensorflow_federated/python/core/impl/compiler",
        "//third_party/tensorflow_federated/python/core/impl/computation",
        "//third_party/tensorflow_federated/python/core/impl/context_stack",
        "//third_party/tensorflow_federated/python/core/impl/execution_contexts",
        "//third_party/tensorflow_federated/python/core/impl/executors",
        "//third_party/tensorflow_federated/python/core/impl/federated_context",
        "//third_party/tensorflow_federated/python/core/impl/tensorflow_context",
        "//third_party/tensorflow_federated/python/core/impl/types",
        "//third_party/tensorflow_federated/python/core/impl/utils",
        "//third_party/tensorflow_federated/python/core/impl/wrappers",
        "//third_party/tensorflow_federated/python/core/impl/xla_context",
        "//third_party/tensorflow_federated/python/tensorflow_libs",
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
