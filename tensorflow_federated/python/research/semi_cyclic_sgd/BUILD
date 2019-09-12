package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "cyclic_bag_log_reg",
    srcs = ["cyclic_bag_log_reg.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":sentiment_util"],
)

py_binary(
    name = "preprocess_sentiment140",
    srcs = ["preprocess_sentiment140.py"],
    python_version = "PY3",
    srcs_version = "PY3",
)

py_library(
    name = "sentiment_util",
    srcs = ["sentiment_util.py"],
    srcs_version = "PY3",
)
