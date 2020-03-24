package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "remote_executor_service",
    srcs = ["remote_executor_service.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/runtime/container:__pkg__"],
    deps = ["//tensorflow_federated"],
)
