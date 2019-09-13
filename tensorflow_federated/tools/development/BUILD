package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

# The data labels are needed otherwise Bazel will not know about the
# `__init__.py` files; this is because the modules are imported directory,
# rather than the packages, so in the Bazel dependency hierarchy, nothing is
# dependent on the `__init__.py` files.
sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        ":pip_package_files",
        "//tensorflow_federated",
        "//tensorflow_federated/proto",
        "//tensorflow_federated/proto/v0",
        "//tensorflow_federated/python",
        "//tensorflow_federated/python/common_libs",
        "//tensorflow_federated/python/core/impl",
        "//tensorflow_federated/python/core/impl/compiler",
        "//tensorflow_federated/python/core/impl/utils",
        "//tensorflow_federated/python/tensorflow_libs",
    ],
)

filegroup(
    name = "pip_package_files",
    srcs = ["setup.py"],
)

sh_binary(
    name = "publish_pip_package",
    srcs = ["publish_pip_package.sh"],
)

sh_binary(
    name = "test_pip_package",
    srcs = ["test_pip_package.sh"],
)
