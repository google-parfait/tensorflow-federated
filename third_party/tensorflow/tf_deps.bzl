"""TensorFlow dependency helpers for TensorFlow Federated."""

def tff_tf_deps(source_deps):
    # copybara:comment_begin(oss-only)
    return select({
        "//third_party/tensorflow:tff_pip_build": [
            "@pypi_tensorflow//:libtensorflow_framework",
            "@pypi_tensorflow//:libtensorflow_cc",
            "@pypi_tensorflow//:headers_lib",
        ],
        "//conditions:default": source_deps,
    })
    # copybara:comment_end
