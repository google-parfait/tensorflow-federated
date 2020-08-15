package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "triehh_tf",
    srcs = ["triehh_tf.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/analytics/heavy_hitters:heavy_hitters_utils",
    ],
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
    name = "triehh_tff_test",
    srcs = ["triehh_tff_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    deps = [
        ":triehh_tf",
        ":triehh_tff",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/analytics/heavy_hitters:heavy_hitters_testcase",
    ],
)

py_test(
    name = "triehh_tf_test",
    srcs = ["triehh_tf_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    deps = [
        ":triehh_tf",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/analytics/heavy_hitters:heavy_hitters_testcase",
    ],
)
