load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_binary(
    name = "emnist_fedavg_main",
    srcs = ["emnist_fedavg_main.py"],
    deps = [
        ":stateful_fedavg_tf",
        ":stateful_fedavg_tff",
        "//tensorflow_federated",
    ],
)

py_library(
    name = "stateful_fedavg_tf",
    srcs = ["stateful_fedavg_tf.py"],
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "stateful_fedavg_tff",
    srcs = ["stateful_fedavg_tff.py"],
    deps = [
        ":stateful_fedavg_tf",
        "//tensorflow_federated",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "stateful_fedavg_test",
    size = "medium",
    srcs = ["stateful_fedavg_test.py"],
    deps = [
        ":stateful_fedavg_tf",
        ":stateful_fedavg_tff",
        "//tensorflow_federated",
    ],
)
