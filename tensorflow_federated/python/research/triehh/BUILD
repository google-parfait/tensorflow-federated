package(default_visibility = ["//tensorflow_federated/python/research/triehh:__subpackages__"])

licenses(["notice"])

py_library(
    name = "triehh_tf",
    srcs = ["triehh_tf.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "triehh_tff",
    srcs = ["triehh_tff.py"],
    srcs_version = "PY3",
    deps = [
        ":triehh_tf",
        "//tensorflow_federated",
    ],
)

py_test(
    name = "triehh_test",
    srcs = ["triehh_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":triehh_tf",
        ":triehh_tff",
        "//tensorflow_federated",
    ],
)
