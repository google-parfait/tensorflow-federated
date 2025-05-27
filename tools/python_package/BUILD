package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "python_package_tool",
    packages = ["//tools/python_package/..."],
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
        "//:LICENSE",
        "//:README.md",
        "//:pyproject_toml",
        "//tensorflow_federated",
        "//tensorflow_federated/data:worker_binary",
        "//tensorflow_federated/proto",
        "//tensorflow_federated/proto/v0",
        "//tensorflow_federated/python",
        "//tensorflow_federated/python/analytics/hierarchical_histogram",
        "//tensorflow_federated/python/common_libs",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/core/environments",
        "//tensorflow_federated/python/core/environments/tensorflow_backend",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend",
        "//tensorflow_federated/python/core/impl",
        "//tensorflow_federated/python/core/impl/compiler",
        "//tensorflow_federated/python/core/impl/execution_contexts",
        "//tensorflow_federated/python/core/impl/executor_stacks",
        "//tensorflow_federated/python/core/impl/executors",
    ],
)

sh_binary(
    name = "test_python_package",
    srcs = ["test_python_package.sh"],
)
