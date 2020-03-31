package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

filegroup(
    name = "dockerfile_file",
    srcs = ["local.Dockerfile"],
)

sh_binary(
    name = "build_image",
    srcs = ["build_image.sh"],
    data = [
        ":dockerfile_file",
        "//tensorflow_federated/tools/development:build_pip_package",
        "//tensorflow_federated/tools/runtime:remote_executor_service",
    ],
)
