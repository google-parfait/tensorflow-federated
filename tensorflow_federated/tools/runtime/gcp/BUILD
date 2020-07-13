package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

sh_binary(
    name = "deploy_endpoint",
    srcs = ["deploy_endpoint.sh"],
    data = [
        ":worker_configuration_file",
        "//tensorflow_federated/proto/v0:proto_files",
    ],
)

filegroup(
    name = "worker_configuration_file",
    srcs = ["worker.yaml"],
)
