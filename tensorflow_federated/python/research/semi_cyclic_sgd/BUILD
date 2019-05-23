package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "cyclic_bag_log_reg",
    srcs = ["cyclic_bag_log_reg.py"],
    deps = [":sentiment_util"],
)

py_binary(
    name = "preprocess_sentiment140",
    srcs = ["preprocess_sentiment140.py"],
)

py_library(
    name = "sentiment_util",
    srcs = ["sentiment_util.py"],
)
