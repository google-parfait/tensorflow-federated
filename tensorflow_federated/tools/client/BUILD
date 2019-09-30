package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

sh_binary(
    name = "build_image",
    srcs = ["build_image.sh"],
    data = [
        ":dockerfile_file",
        "//tensorflow_federated/tools/development:build_pip_package",
    ],
)

filegroup(
    name = "dockerfile_file",
    srcs = ["Dockerfile"],
)

sh_binary(
    name = "publish_image",
    srcs = ["publish_image.sh"],
)

filegroup(
    name = "test",
    srcs = ["test.py"],
    tags = ["ignore_srcs"],
)
