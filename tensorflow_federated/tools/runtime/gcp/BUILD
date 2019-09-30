package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

sh_binary(
    name = "build_image",
    srcs = ["build_image.sh"],
    data = [
        ":dockerfile_file",
        "//tensorflow_federated/python/simulation:worker",
        "//tensorflow_federated/tools/development:build_pip_package",
    ],
)

filegroup(
    name = "dockerfile_file",
    srcs = ["Dockerfile"],
)

sh_binary(
    name = "deploy_endpoint",
    srcs = ["deploy_endpoint.sh"],
    data = [
        ":worker_configuration_file",
        "//tensorflow_federated/proto/v0:proto_files",
    ],
)

sh_binary(
    name = "publish_image",
    srcs = ["publish_image.sh"],
)

filegroup(
    name = "worker_configuration_file",
    srcs = ["worker.yaml"],
)
