package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "//tensorflow_federated",
        "//tensorflow_federated/tools:pip_package_files",
    ],
)

filegroup(
    name = "pip_package_files",
    srcs = ["setup.py"],
)
