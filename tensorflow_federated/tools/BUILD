package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

# The py library data labels are needed otherwise bazel doesn't generate
# the __init__.py files, since the modules in the package are imported
# directory, rather than the package itself.
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
        "//tensorflow_federated/python/tensorflow_libs",
    ],
)

filegroup(
    name = "pip_package_files",
    srcs = ["setup.py"],
)
