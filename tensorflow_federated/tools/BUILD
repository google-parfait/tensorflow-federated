package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        ":pip_package_files",
        "//tensorflow_federated",
    ],
)

filegroup(
    name = "pip_package_files",
    srcs = ["setup.py"],
)
